import copy

import torch
import torch.nn.functional as F
from torch import device, nn

from aule.assigners.hungarian_assigner import HungarianAssigner
from aule.groundingdino.models.GroundingDINO.utils import (bbox_cxcywh_to_xyxy,
                                                           bbox_xyxy_to_cxcywh,
                                                           reduce_mean)
from aule.groundingdino.util import box_ops
from aule.losses.focal_loss import FocalLoss
from aule.losses.iou_loss import GIoULoss
from aule.losses.smooth_l1_loss import L1Loss
from aule.misc import multi_apply

from .matcher import HungarianMatcher


def create_positive_map(tokenized, tokens_positive, cat_list, caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, label in enumerate(tokens_positive):

        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map


class GroundingDinoCriterion(nn.Module):

    def __init__(
        self,
        matcher=HungarianAssigner(),
        weight_dict=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
        max_text_len: int = 512,
        num_classes: int = 256,
    ):
        super().__init__()
        self.matcher = matcher
        if weight_dict is None:
            weight_dict = {
                "loss_bbox": 5.0,
                "loss_ce": 1.0,
                "loss_giou": 2.0,
            }
        self.weight_dict = weight_dict

        self.giou_loss = GIoULoss(loss_weight=weight_dict["loss_giou"])
        self.bbox_loss = L1Loss(loss_weight=weight_dict["loss_bbox"])
        self.ce_loss = FocalLoss(
            loss_weight=weight_dict["loss_ce"], gamma=focal_gamma, alpha=focal_alpha
        )

        self.max_text_len = max_text_len

        self.num_classes = num_classes

    @staticmethod
    def split_outputs_all_layers(
        all_layers_cls_scores: torch.Tensor,
        all_layers_bbox_preds: torch.Tensor,
        dn_meta: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta["num_denoising_queries"]
        if dn_meta is not None:
            all_layers_denoising_cls_scores = all_layers_cls_scores[
                :, :, :num_denoising_queries, :
            ]
            all_layers_denoising_bbox_preds = all_layers_bbox_preds[
                :, :, :num_denoising_queries, :
            ]
            all_layers_matching_cls_scores = all_layers_cls_scores[
                :, :, num_denoising_queries:, :
            ]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[
                :, :, num_denoising_queries:, :
            ]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
        )

    def split_aux_outputs(self, outputs: dict):
        logits, boxes = (
            outputs["aux_outputs"]["pred_logits"],
            outputs["aux_outputs"]["pred_boxes"],
        )
        if outputs.get("dn_meta", None) is not None:
            logits, boxes, denoising_logits, denoising_boxes = (
                self.split_outputs_all_layers(logits, boxes, dn_meta=outputs["dn_meta"])
            )
            outputs["denoising_logits"] = denoising_logits
            outputs["denoising_boxes"] = denoising_boxes
        outputs["pred_logits"] = logits[-1]
        outputs["pred_boxes"] = boxes[-1]
        outputs["pre_logits"] = logits[:-1]
        outputs["pre_boxes"] = boxes[:-1]

    def _get_dn_targets_single(
        self, target: dict, positive_maps, dn_meta, **kwargs
    ) -> tuple:
        gt_labels = target["labels"]
        num_groups = dn_meta["num_denoising_groups"]
        num_denoising_queries = dn_meta["num_denoising_queries"]
        num_queries_each_group = int(num_denoising_queries / num_groups)

        factor = target["size"].repeat(1, 2).float().to(target["boxes"].device)

        gt_bboxes = bbox_cxcywh_to_xyxy(target["boxes"]) * factor
        device = gt_bboxes.device
        positive_maps = positive_maps.to(device)

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2

        # label targets
        labels = gt_bboxes.new_full(
            (num_denoising_queries, self.max_text_len), 0, dtype=torch.float32
        )
        labels[pos_inds] = positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_dn_targets(self, targets, positive_maps, dn_meta):
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_dn_targets_single, targets, positive_maps, dn_meta=dn_meta
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _denoising_loss_single(
        self,
        dn_cls_scores,
        dn_bbox_preds,
        targets: list[dict],
        text_token_mask,
        positive_maps,
        dn_meta,
    ):
        cls_reg_targets = self.get_dn_targets(targets, positive_maps, dn_meta)
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        text_masks = text_token_mask.new_zeros(
            (text_token_mask.size(0), self.max_text_len)
        )
        text_masks[:, : text_token_mask.size(1)] = text_token_mask
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.ce_loss(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor
            )
        else:
            loss_cls = torch.zeros(1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = torch.cat(
            [
                t["size"].repeat(1, 2).float().repeat(dn_bbox_preds[idx].size(0), 1)
                for idx, t in enumerate(targets)
            ],
            0,
        ).to(dn_bbox_preds.device)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.giou_loss(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.bbox_loss(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        return loss_cls, loss_iou, loss_bbox

    def denoising_loss(
        self,
        all_layers_denoising_cls_scores,
        all_layers_denoising_bbox_preds,
        targets,
        dn_meta,
        text_token_mask,
        positive_maps,
    ):
        losses_cls, losses_iou, losses_bbox = multi_apply(
            self._denoising_loss_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            targets=targets,
            dn_meta=dn_meta,
            text_token_mask=text_token_mask,
            positive_maps=positive_maps,
        )
        return [
            {
                "loss_cls": cls,
                "loss_iou": iou,
                "loss_bbox": bbox,
            }
            for cls, iou, bbox in zip(losses_cls, losses_iou, losses_bbox)
        ]

    def _get_targets_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        target: dict,
        positive_maps: dict,
        text_token_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        target["boxes"] = target["boxes"].to(bbox_pred.device)
        target["labels"] = target["labels"].to(cls_score.device)
        factor = target["size"].repeat(1, 2).float().to(bbox_pred.device)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor
        
        positive_maps = positive_maps.to(bbox_pred.device)

        img_meta = target["size"]
        img_meta = {"img_shape": (img_meta[0].item(), img_meta[1].item())}

        boxes = bbox_cxcywh_to_xyxy(target["boxes"]) * factor
        
        assign_result = self.matcher.assign(
            {"pred_bboxes": bbox_pred, "scores": cls_score},
            dict(
                boxes=boxes,
                labels=target["labels"],
            ),
            positive_maps=positive_maps,
            text_token_mask=text_token_mask,
            img_meta=img_meta,
        )
        
        pos_inds = (
            torch.nonzero(assign_result["gt_inds"] > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result["gt_inds"] == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        
        pos_assigned_gt_inds = assign_result["gt_inds"][pos_inds] - 1
        pos_gt_bboxes = boxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = boxes.new_full((num_bboxes, self.max_text_len), 0, dtype=torch.float32)
        labels[pos_inds] = positive_maps[pos_assigned_gt_inds]

        label_weights = boxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=boxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=boxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        targets,
        positive_maps,
        text_token_mask,
        **kwargs,
    ):
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            targets,
            positive_maps,
            text_token_mask,
            **kwargs,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def prediction_loss(
        self, cls_scores, bbox_preds, targets: list[dict], text_token_mask, positive_maps,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        with torch.no_grad():
            cls_reg_targets = self.get_targets(
                cls_scores_list,
                bbox_preds_list,
                targets,
                text_token_mask=text_token_mask,
                positive_maps=positive_maps,
            )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        text_masks = text_token_mask.new_zeros(
            (text_token_mask.size(0), self.max_text_len)
        )
        text_masks[:, : text_token_mask.size(1)] = text_token_mask
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)

        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()
        
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[..., None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0
        cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
                
        loss_cls = self.ce_loss(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = torch.cat(
            [
                t["size"].repeat(1, 2).float().repeat(bbox_preds[idx].size(0), 1)
                for idx, t in enumerate(targets)
            ],
            0,
        ).to(bbox_preds.device)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.giou_loss(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.bbox_loss(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
                
        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_iou": loss_iou,
        }

    def forward(self, outputs: dict, targets: list[dict]):
        """
        :param outputs:
        :param targets:
        :return:
        """
        if "aux_outputs" in outputs:
            self.split_aux_outputs(outputs)

        losses = {}

        prediction_losses = self.prediction_loss(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            targets,
            text_token_mask=outputs["text_token_mask"],
            positive_maps=outputs["positive_maps"],
        )
        losses.update(prediction_losses)

        enc_losses = self.prediction_loss(
            outputs["interm_outputs"]["pred_logits"],
            outputs["interm_outputs"]["pred_boxes"],
            targets,
            text_token_mask=outputs["text_token_mask"],
            positive_maps=outputs["positive_maps"],
        )
        losses.update({f"enc_{k}": v for k, v in enc_losses.items()})

        if "pre_logits" in outputs:
            for idx, (logit, boxes) in enumerate(
                zip(outputs["pre_logits"], outputs["pre_boxes"])
            ):
                pre_loss = self.prediction_loss(
                    logit,
                    boxes,
                    targets,
                    text_token_mask=outputs["text_token_mask"],
                    positive_maps=outputs["positive_maps"],
                )
                losses.update({f"d{idx}.{k}": v for k, v in pre_loss.items()})

        if "denoising_logits" in outputs:
            denoising_losses = self.denoising_loss(
                outputs["denoising_logits"],
                outputs["denoising_boxes"],
                targets,
                text_token_mask=outputs["text_token_mask"],
                dn_meta=outputs["dn_meta"],
                positive_maps=outputs["positive_maps"],
            )
            losses.update({f"dn_{k}": v for k, v in denoising_losses[-1].items()})
            for idx, denoising_single_loss in enumerate(denoising_losses[:-1]):
                losses.update(
                    {f"d{idx}.dn_{k}": v for k, v in denoising_single_loss.items()}
                )

        return losses
