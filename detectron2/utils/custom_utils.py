import torch
from torch import nn
import copy
import kornia

import detectron2
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.structures import ROIMasks, Instances
from typing import Dict, List, Optional, Tuple

def deep_merge(_dict1: dict, _dict2: dict) -> dict:
    dict1, dict2 = copy.deepcopy(_dict1), copy.deepcopy(_dict2)
    def _val(v1, v2):
        if isinstance(v1, dict) and isinstance(v2, dict):
            return deep_merge(v1, v2)
        elif isinstance(v1, list) and isinstance(v2, list):
            return v1 + v2
        return v2 or v1
    return {k: _val(dict1.get(k), dict2.get(k)) for k in dict1.keys() | dict2.keys()}

def same_type_conv(dict_1, dict_2):
    for k in dict_1:
        dict_1[k] = type(dict_2[k])(dict_1[k])
    if isinstance(dict_1[k], dict): same_type_conv(dict_1[k], dict_2[k])

def RMSELoss(x, y):
    eps = 1e-6
    loss = torch.sqrt(nn.functional.mse_loss(x, y) + eps)
    return loss

def expand_box(bb_tensor, max_width, max_height, ratio = 0.3):
    for bb in bb_tensor:
        x0, y0, x1, y1 = bb
        width, height = x1 - x0, y1 - y0
        bb[0] = (x0 - width * ratio) if (x0 - width * ratio) > 0 else 0
        bb[1] = (y0 - height * ratio) if (y0 - height * ratio) > 0 else 0
        bb[2] = x1 + (width * ratio) if x1 + (width * ratio) < max_width else max_width
        bb[3] = y1 + (height * ratio) if y1 + (height * ratio) < max_height else max_height
    return bb_tensor

def get_iou(a, b, type="xyxy", epsilon=1e-5):
    if type == "xyxy":
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    elif type == "xywh":
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2] + a[0], b[2] + b[0]), min(a[3] + a[1], b[3] + b[1])
    width, height = (x2 - x1), (y2 - y1)
    if (width<0) or (height <0): return 0.0
    area_overlap = width * height
    if type == "xyxy":
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
    elif type == "xywh":
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]

    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

def calculate_center(bb, xyxy = True):
    if xyxy: return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]
    else: return [bb[0] + (bb[2]/2), bb[1] + (bb[3]/2)]

def select_hands_proposals(proposals: List[Instances], hand_label: int) -> Tuple[List[Instances], List[torch.Tensor]]:
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    hands_proposals = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes == hand_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        hands_proposals.append(proposals_per_image[fg_idxs])
    return hands_proposals

#### INPUT ARRAY OF INSTANCES [B, I] OUTPUT TENSOR [B, N, OUTPUT_SIZE_H, OUTPUT_SIZE_W]
def extract_masks_and_resize(preds, output_size, class_id):
    output_masks = []
    for pred in preds:
        pred = pred[pred.gt_classes == class_id] if "gt_classes" in pred._fields else pred[pred.pred_classes == class_id]
        if len(pred) == 0: continue
        original_size =  pred.image_size
        roi_masks = ROIMasks(pred.pred_masks[:, 0, :, :])
        masks = roi_masks.to_bitmasks(pred.proposal_boxes, original_size[0], original_size[1], 0.5).tensor if "proposal_boxes" in pred._fields else roi_masks.to_bitmasks(pred.pred_boxes, original_size[0], original_size[1], 0.5).tensor
        masks = masks.float()
        res_masks = kornia.geometry.transform.resize(masks, output_size, align_corners = True)
        output_masks.append(res_masks)
    return output_masks