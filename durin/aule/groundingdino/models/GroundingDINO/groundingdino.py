import copy
import logging
import warnings
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchEncoding

from aule.groundingdino.models.GroundingDINO.utils import bbox_cxcywh_to_xyxy
from aule.groundingdino.util import get_tokenlizer
from aule.groundingdino.util.misc import (NestedTensor, inverse_sigmoid,
                                          nested_tensor_from_tensor_list,
                                          split_by_mask)

from .backbone import build_backbone
from .bertwarper import (BertModelWarper,
                         generate_masks_with_special_tokens_and_transfer_map)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DefaultBackboneArguments:
    backbone: str = "swin_B_384_22k"
    backbone_freeze_keywords: tuple[str] | None = None
    use_checkpoint: bool = True
    dilation: bool = True
    return_interm_indices: list[int] = field(default_factory=lambda: [1,2,3])
    hidden_dim: int = 256
    position_embedding: str = "sine"
    pe_temperatureH: int = 20
    pe_temperatureW: int = 20

@dataclass
class DefaultTransformerArgument:
    enc_layers: int = 6
    dec_layers: int = 6
    pre_norm: bool = False
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.0
    nheads: int = 8
    num_queries: int = 900
    query_dim: int = 4
    num_patterns: int = 0
    num_feature_levels: int = 4
    enc_n_points: int = 4
    dec_n_points: int = 4
    two_stage_type: str = "standard"
    transformer_activation: str = "relu"
    embed_init_tgt: bool = True
    use_fusion_layer: bool = True
    use_text_enhancer: bool = True
    use_checkpoint: bool = True
    use_transformer_ckpt: bool = True
    use_text_cross_attention: bool = True
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries: int = 900,
        aux_loss: bool = False,
        query_dim: int = 4,
        num_feature_levels: int = 4,
        nheads: int = 8,
        # two stage
        two_stage_type="standard",  # ['no', 'standard']
        dec_pred_bbox_embed_share=False,
        two_stage_class_embed_share: bool = False,
        two_stage_bbox_embed_share: bool = False,
        num_patterns: int = 0,
        dn_number: int = 256,
        dn_box_noise_scale: float = 1.0,
        dn_label_noise_ratio: float = 0.5,
        dn_labelbook_size: int = 2000,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=512,
        force_training: bool = False,
        freeze_language_model: bool = True,
        sep_token: str = ". ",
        config_file_language_model: str | None = None,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of queries to be used in the transformer
            aux_loss: whether to use auxiliary decoding losses
            query_dim: dimension of the query
            num_feature_levels: number of feature levels to be used in the transformer
            nheads: number of heads to be used in the transformer
            two_stage_type: Either 'no' or 'standard'.
            dec_pred_bbox_embed_share: whether to share the bbox embedding between decoder layers
            two_stage_class_embed_share: whether to share the class embedding between decoder layers
            two_stage_bbox_embed_share: whether to share the bbox embedding between encoder and decoder
            num_patterns: number of patterns to be used in the transformer
            dn_number: number of denoising queries to be used in the transformer
            dn_box_noise_scale: scale of the box noise to be used in the transformer
            dn_label_noise_ratio: scale of the label noise to be used in the transformer
            dn_labelbook_size: size of the label book to be used in the transformer
            text_encoder_type: type of the text encoder to be used. See get_tokenlizer.py
            sub_sentence_present: whether to use sub-sentence present in the transformer, meaning if the text is
                tokenized into sub-sentences (classes, e.g. dog running . cat dancing .)
            max_text_len: maximum length of the text to be used in the transformer
            force_training: whether to force training of the model even if model.eval() is called
            freeze_language_model: whether to freeze the language model
            sep_token: token used to separate phrases in the text
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type, config_file=config_file_language_model)
        try:
            self.bert.pooler.dense.weight.requires_grad_(False)
            self.bert.pooler.dense.bias.requires_grad_(False)
        except AttributeError:
            pass
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(
            self.bert.config.hidden_size, self.hidden_dim, bias=True
        )
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        self.special_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            if self.num_feature_levels > num_backbone_outs:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed(max_text_len=max_text_len)

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [
            copy.deepcopy(_class_embed) for _ in range(transformer.num_decoder_layers)
        ]
        self.transformer.decoder.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.transformer.decoder.class_embed = nn.ModuleList(class_embed_layerlist)

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

        self.sep_token = sep_token
        self.force_training = force_training
        self.freeze_language_model = freeze_language_model

    def _reset_parameters(self) -> None:
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries: int) -> None:
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def generate_positive_maps(
        self, caption: str, sep_token: str | None = None
    ) -> list[tuple[int, int]]:
        """
        Generates a list where each element is a phrase between self.sep_token.
        Example:
        "car . dog . cat ."

        tokens_positive = [[0, 4], [6, 10], [12, 15]]

        :param caption: string of phrases seperated by self.sep_token
        :param sep_token: token used to separate phrases, by default self.sep_token
        :return: list of tuples indicating start and end of each phrase
        """
        caption_string = ""
        if sep_token is None:
            sep_token = self.sep_token
        tokens_positive = []
        caption = caption.strip(self.sep_token.strip())
        for idx, word in enumerate(caption.split(sep_token)):
            tokens_positive.append(
                [[len(caption_string), len(caption_string) + len(word)]]
            )
            caption_string += word
            caption_string += sep_token
        return tokens_positive

    def create_positive_map(
        self,
        tokenized: BatchEncoding,
        tokens_positive: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        """
        Positive map is a torch tensor of shape:
        number of phrases x maximum number of tokens.
        Each row belongs to a phrase and each and is filled with 1.0 on each token position
        that belongs to a phrase, else 0.
        So if there is a caption like 'dog running .' that consists of 3 tokens.
        Let's say when tokenizing this caption we get [SOS; DOG_TOKEN, RUN_TOKEN, ING_TOKEN, ._TOKEN, EOS]
        tokens_positive will be [[0, 11]].
        beg_pos will be where the character of index 0 (dog) is mapped when text is tokenized
        (for character d it belongs to the token DOG_TOKEN so beg_pos will be 1)
        And character ' ' belong to the ING_TOKEN so end_pos will be 3.

        :param tokenized: tokenized input (contains input_ids and other values)
        :param tokens_positive: got from self.generate_positive_maps, list of tuples
         indicating start and end of each phrase
        :return:
        """
        positive_map = torch.zeros(
            (len(tokens_positive), self.max_text_len), dtype=torch.float
        )

        for j, tok_list in enumerate(tokens_positive):
            for beg, end in tok_list:
                try:
                    beg_pos = tokenized.char_to_token(beg)
                    end_pos = tokenized.char_to_token(end - 1)
                except Exception as e:
                    logger.error("Can't get the correct token in create_positive_map")
                    raise e
                if beg_pos is None:
                    try:
                        beg_pos = tokenized.char_to_token(beg + 1)
                        if beg_pos is None:
                            beg_pos = tokenized.char_to_token(beg + 2)
                    except:
                        beg_pos = None
                if end_pos is None:
                    try:
                        end_pos = tokenized.char_to_token(end - 2)
                        if end_pos is None:
                            end_pos = tokenized.char_to_token(end - 3)
                    except Exception as _:
                        end_pos = None
                if beg_pos is None or end_pos is None:
                    continue

                assert beg_pos is not None and end_pos is not None
                positive_map[j, beg_pos : end_pos + 1].fill_(1)
        return positive_map / (positive_map.sum(-1)[:, None] + 1e-6).bool().float()

    def backbone_forward(
        self, samples: NestedTensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Extract features from the backbone and project them to the hidden dimension.
        """
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
            
        if self.num_feature_levels > len(srcs):
            src = self.input_proj[-1](features[-1].tensors)
            m = samples.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                torch.bool
            )[0]
            pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src)
            masks.append(mask)
            poss.append(pos_l)
        return srcs, masks, poss

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
            
        tokenized = self.tokenizer(
            captions,
            padding="longest",
            return_tensors="pt",
            max_length=self.max_text_len,
        ).to(samples.device)

        targets = [
            {
                k: v.to(samples.device) if isinstance(v, torch.Tensor) else v
                for k, v in tg.items()
            }
            for tg in targets
        ]
        
        positive_map = []
        if targets[0].get("positive_token_indices", None) is not None:
            for idx, (t, caption) in enumerate(zip(targets, captions)):
                positive_token_indices = t[
                    "positive_token_indices"
                ]  # this only has the phrases that have a bounding box
                pos_tokens = self.generate_positive_maps(caption)
                positive_token_indices = [
                    pos_tokens[pos] for pos in positive_token_indices
                ]  # only take tokens from the phrases that are positive
                positive_map.append(
                    self.create_positive_map(tokenized[idx], positive_token_indices)
                )  # torch tensor of shape (number of positive phrases, max number of tokens) where each row
                # is filled with 1.0 on each token position that belongs to a phrase, else 0.

        one_hot_token = tokenized

        (text_self_attention_masks, position_ids, special_tokens_mask) = (
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, self.special_tokens
            )
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.max_text_len
            ]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        if self.freeze_language_model:
            with torch.no_grad():
                bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        else:
            bert_output = self.bert(**tokenized_for_encoder)

        encoded_text = self.feat_map(
            bert_output["last_hidden_state"]
        )  # bs, 195, d_model

        text_token_mask = tokenized.attention_mask.bool()  # bs, 195, padding mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195, padding mask
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195 which token attends to which token
        }

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        srcs, masks, poss = self.backbone_forward(samples)
        
        input_query_bbox = input_query_label = attn_mask = dn_meta = (
            None  # only when training, used for denoising loss
        )
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )
                
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.transformer.decoder.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(
                    self.transformer.decoder.class_embed, hs
                )
            ]
        )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        if dn_meta is not None:
            out["dn_meta"] = dn_meta

        # Used to calculate losses
        bs, len_td = text_dict["text_token_mask"].shape
        out["text_mask"] = torch.zeros(bs, self.max_text_len, dtype=torch.bool)
        for b in range(bs):
            for j in range(len_td):
                if text_dict["text_token_mask"][b][j] == True:
                    out["text_mask"][b][j] = True

        out["text_token_mask"] = text_dict["text_token_mask"]
        # for intermediate outputs
        if self.aux_loss and (self.training or self.force_training):
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)
        out["token"] = one_hot_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

        out["special_tokens_mask"] = special_tokens_mask
        out["positive_maps"] = positive_map
        out["embeddings"] = hs[-1]

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }

    @torch.no_grad()
    def predict(
        self,
        samples: NestedTensor,
        targets: List = None,
        box_threshold: float = 0.05,
        text_threshold: float = 0.05,
        **kw,
    ):
        outputs = self(samples, targets, **kw)

        special_tokens_mask = outputs["special_tokens_mask"]
        logits = outputs["pred_logits"].sigmoid()

        boxes = bbox_cxcywh_to_xyxy(outputs["pred_boxes"])
        embeddings = outputs["embeddings"]

        results = []

        for logit, box, mask, tg, embs in zip(logits, boxes, special_tokens_mask, targets, embeddings):
            cap_list = tg.get(
                "cap_list",
                [
                    cap.strip(".").strip()
                    for cap in tg["caption"].strip(".").split(".")
                    if cap.strip(".").strip("")
                ],
            )
            filter_mask = logit.max(dim=1)[0] > box_threshold
            logit = logit[filter_mask]
            boxes = box[filter_mask]
            embs = embs[filter_mask]

            split_logits = [split_by_mask(l[: len(mask)], mask) for l in logit]
            res = defaultdict(list)
            for idx, logit in enumerate(split_logits):
                max_arg = logit.argmax().item()
                if logit[max_arg].item() < text_threshold:
                    continue
                res["boxes"].append(boxes[idx])
                res["scores"].append(logit[max_arg].item())
                res["labels"].append(max_arg)
                res["label_names"].append(cap_list[max_arg])
                res["embeddings"].append(embs[idx])

            # Get the indices that would sort 'scores' in descending order
            if res["boxes"]:
                res["boxes"] = torch.stack(res["boxes"]).cpu()
                res["scores"] = torch.tensor(res["scores"]).cpu()
                res["labels"] = torch.tensor(res["labels"]).cpu()
                res["embeddings"] = torch.stack(res["embeddings"]).cpu()
            else:
                res["boxes"] = torch.tensor([])
                res["scores"] = torch.tensor([])
                res["labels"] = torch.tensor([])
                res["embeddings"] = torch.tensor([])

            results.append(res.copy())

        return results

def build_predictor(config_file_language_model: str | None = None) -> GroundingDINO:
    backbone = build_backbone(DefaultBackboneArguments())
    transformer = build_transformer(DefaultTransformerArgument())
    
    return GroundingDINO(backbone=backbone, transformer=transformer, config_file_language_model=config_file_language_model)