from torchvision.ops import box_iou, box_area
import torch

def filter_ovelapping_boxes(
    bbox: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold=0.2,
    overlap_threshold = 0.6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filters out detections that completely overlap with others or that have iou bigger than iou_threshold.
    Only leaves the detection with the highest score.

    Args:
        bbox (torch.Tensor): Bounding boxes of shape (N, 4).
        scores (torch.Tensor): Scores of shape (N,).

    Returns:
        tuple: Filtered bounding boxes and scores.
    """
    original_indices = torch.arange(bbox.shape[0], device=bbox.device)
    scores, sorted_indices = torch.sort(scores, descending=True)

    bbox = bbox[sorted_indices]
    ious = box_iou(bbox, bbox)
    sorted_original_indices = original_indices[sorted_indices]
    
    bbox_areas = box_area(bbox)
            
    keep = torch.ones(scores.shape[0], dtype=torch.bool, device=bbox.device)
    for i in range(scores.shape[0]):
        if not keep[i]:
            continue
        for j in range(i + 1, scores.shape[0]):
            if not keep[j]:
                continue
            if ious[i][j] > iou_threshold:
                keep[j] = False
                continue
            
            intersection = ((bbox_areas[i] + bbox_areas[j])* ious[i][j])/(1+ ious[i][j])
            if intersection/min(bbox_areas[i], bbox_areas[j]) > overlap_threshold:
                keep[j] = False
                continue

    return sorted_original_indices[keep].tolist()
            

def filter_detections(
    bbox: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_threshold: float = 0.1,
    iou_threshold: float = 0.5,
    overlap_threshold: float = 0.75,
) -> torch.Tensor:
    """
    Filters the detections based on score and overlap.
    Filters out detections with scores below the score threshold.
    Filters out detections that overlap with others above the IoU threshold, but only detections from the same class.
    Filters out detections that completely overlap (if two detections have the same class and one is completely inside the other, the one with the lower score is removed).

    Args:
        bbox (torch.Tensor): Bounding boxes of shape (N, 4).
        scores (torch.Tensor): Scores of shape (N,).
        labels (torch.Tensor): Labels of shape (N,).
        score_threshold (float): Minimum score to keep a detection.
        iou_threshold (float): IoU threshold for NMS.
        overlap_threshold (float): Overlap threshold for filtering.

    Returns:
        torch.Tensor: indices of the kept detections.
    """
    bbox = bbox.clone()
    scores = scores.clone()
    labels = labels.clone()
    
    # group by labels
    label_groups = {}
    for i, label in enumerate(labels):
        label = label.item()
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)
        
    keep_indices = []
    for label, indices in label_groups.items():
        if len(indices) > 1:
            ids = filter_ovelapping_boxes(bbox[indices], scores[indices], iou_threshold, overlap_threshold)
            indices = [indices[i] for i in ids]
            keep_indices.extend(indices)
        else:
            keep_indices.append(indices[0])
    
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=scores.device)
    scores_to_keep = scores[keep_indices] > score_threshold
    # bbox = bbox[keep_indices[scores_to_keep]]
    # scores = scores[keep_indices[scores_to_keep]]
    
    # ids = filter_ovelapping_boxes(bbox, scores, iou_threshold_different_classes, overlap_threshold_different_classes)
    # keep_indices = [keep_indices[scores_to_keep][idx] for idx in ids]
            
    return keep_indices[scores_to_keep]
            

    