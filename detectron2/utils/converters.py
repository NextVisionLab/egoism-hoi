import numpy as np
import torch
from torchvision.ops.boxes import nms
import copy
from pycocotools.coco import COCO
import math
from abc import abstractmethod

from detectron2.structures import BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import ROIMasks
from detectron2.utils.custom_utils import get_iou, calculate_center
import cv2

class Converter:
    def __init__(self, cfg, metadata) -> None:
        self._cfg = cfg
        self._metadata = metadata
        self._thing_classes = self._metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self._thresh_objs = cfg.ADDITIONAL_MODULES.THRESH_OBJS
        self._nms_thresh = cfg.ADDITIONAL_MODULES.NMS_THRESH

    def convert_instances_to_coco(self, instances, img_id, convert_boxes_xywh_abs = False):
        boxes = instances.pred_boxes.tensor.detach().clone()
        if convert_boxes_xywh_abs: boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        results = []
        for k in range(len(instances)):
            result = {"image_id": img_id, "category_id": classes[k], "bbox": boxes[k], "score": scores[k]}
            results.append(result)
        return results
    
    def convert_coco_to_coco_target_object(self, coco_hands, coco_all):
        coco = COCO()
        tmp_dateset = {"annotations": [], "images": copy.deepcopy(coco_all.dataset["images"]), "info": [], "licenses": []}
        tmp_dateset["categories"] = copy.deepcopy(coco_all.dataset["categories"])
        tmp_dateset["categories"].pop(self._id_hand)
        tmp_dateset["annotations"] = [{"id": x["id"], "image_id": x["image_id"], "category_id": x["category_id_obj"], "area": x["area_obj"], "bbox": x["bbox_obj"], "iscrowd": 0} for x in coco_hands.anns.values() if x["contact_state"] == 1]
        coco.dataset = tmp_dateset
        return coco

    @abstractmethod
    def generate_predictions(self):
        pass
    
    def generate_confident_instances(self, instances):
        confident_instances = instances[instances.scores >= self._thresh_objs]
        return self._nms(confident_instances)

    def _nms(self, confident_instances):
        confident_instances.to(torch.device("cpu"))
        if "pred_boxes" in confident_instances.get_fields():
            keep = nms(confident_instances.pred_boxes.tensor, confident_instances.scores.float(), self._nms_thresh)
        elif "boxes" in confident_instances.get_fields():
            keep = nms(confident_instances.boxes, confident_instances.scores.float(), self._nms_thresh)
        else:
            assert False
        confident_instances = confident_instances[keep]
        return confident_instances

class MMEhoiNetConverterv1(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)
        self._diag = math.sqrt((math.pow(int(cfg.UTILS.TARGET_SHAPE_W), 2) + math.pow(int(cfg.UTILS.TARGET_SHAPE_H), 2)))
        self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR

    def match_object(self, obj_dets, hand_bb, hand_dxdymag):
        object_cc_list = np.array([calculate_center(bbox) for bbox in obj_dets]) # object center list
        magn =  hand_dxdymag[2] / self._scale_factor
        hand_cc = np.array(calculate_center(hand_bb)) # hand center points
        point_cc = np.array([(hand_cc[0] + hand_dxdymag[0] * magn * self._diag), (hand_cc[1] + hand_dxdymag[1] * magn * self._diag)])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist) # find the nearest 
        return dist_min

    def generate_predictions(self, image_id, confident_instances, instances_hand):
        results = []
        results_target = []
        objs = self.convert_instances_to_coco(confident_instances[confident_instances.pred_classes != self._id_hand], image_id)
            
        for idx_hand in range(len(instances_hand)):
            instance_hand = instances_hand[idx_hand]
            bbox_hand = instance_hand.boxes.numpy()[0]
            x0_hand, y0_hand, x1_hand, y1_hand = bbox_hand
            width_hand, heigth_hand = x1_hand - x0_hand, y1_hand - y0_hand
            dxdymag_v = instance_hand.dxdymagn_hand.numpy()[0]
            contact_state = instance_hand.contact_states.item()
            score = instance_hand.scores.item()
            side = instance_hand.sides.item()

            element = {
                "image_id": image_id, 
                "category_id": 0, 
                "bbox": [x0_hand, y0_hand, width_hand, heigth_hand], 
                "score": score, 
                "hand_side": side, 
                "contact_state": contact_state, 
                "bbox_obj": [], 
                "category_id_obj": -1, 
                "dx":dxdymag_v[0],
                "dy": dxdymag_v[1],
                "magnitude": dxdymag_v[2] / self._scale_factor * self._diag
            }
            
            objs_iou, bbox_objs = [], []
            for obj in objs:
                if get_iou(obj["bbox"], bbox_hand) > 0:
                    objs_iou.append(obj)
                    bbox_objs.append(obj["bbox"])
            
            if contact_state and len(bbox_objs): 
                idx_closest_obj = self.match_object(bbox_objs, bbox_hand, dxdymag_v)
                x0, y0, x1, y1 = np.array(bbox_objs[idx_closest_obj]).astype(float)
                width, heigth = x1 - x0, y1 - y0                
                element["bbox_obj"] = [x0, y0, width, heigth]
                element["category_id_obj"] = int(objs_iou[idx_closest_obj]["category_id"])
                element["score_obj"] = objs_iou[idx_closest_obj]["score"]
                results_target.append({"image_id": image_id, "category_id": element["category_id_obj"], "bbox": [x0, y0, width, heigth], "score": element["score_obj"]})
            results.append(element)

        return results, results_target

class MMEhoiNetConverterv2(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)

    def generate_predictions(self, image_id, confident_instances, instances_hand):
        results = []
        results_target = []
            
        for idx_hand in range(len(instances_hand)):
            instance_hand = instances_hand[idx_hand]
            bbox_hand = instance_hand.boxes.numpy()[0]
            x0_hand, y0_hand, x1_hand, y1_hand = bbox_hand
            width_hand, heigth_hand = x1_hand - x0_hand, y1_hand - y0_hand
            interaction_box = instance_hand.pred_interaction_boxes.numpy()[0]
            x0_ib, y0_ib, x1_ib, y1_ib = interaction_box
            w_ib, h_ib = x1_ib - x0_ib, y1_ib - y0_ib
            target_object_box = instance_hand.pred_target_object_boxes.numpy()[0]
            x0_to, y0_to, x1_to, y1_to = target_object_box
            w_to, h_to = x1_to - x0_to, y1_to - y0_to
            target_object_cls = instance_hand.pred_target_object_cls.item()
            contact_state = instance_hand.contact_states.item()
            score = instance_hand.scores.item()
            side = instance_hand.sides.item()
            pred_mask_hand = instance_hand.pred_masks_hand
            pred_mask_target_object = instance_hand.pred_masks_target_object

            element = {
                "image_id": image_id, 
                "category_id": 0, 
                "bbox": [x0_hand, y0_hand, width_hand, heigth_hand], 
                "score": score, 
                "hand_side": side, 
                "contact_state": contact_state, 
                "bbox_obj": [x0_to, y0_to, w_to, h_to], 
                "category_id_obj": target_object_cls, 
                "interaction_box": [x0_ib, y0_ib, w_ib, h_ib],
                "mask_hand": pred_mask_hand,
                "mask_target_object": pred_mask_target_object
            }

            if contact_state: results_target.append({"image_id": image_id, "category_id": element["category_id_obj"], "bbox": element["bbox_obj"], "score": 100})
            results.append(element)

        return results, results_target

class MMEhoiNetConverterv1DepthMask(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)
        self._diag = math.sqrt((math.pow(int(cfg.UTILS.TARGET_SHAPE_W), 2) + math.pow(int(cfg.UTILS.TARGET_SHAPE_H), 2)))
        self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR

    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5):
        results = Instances(size, **results.get_fields())
        if results.has("pred_masks"):
            if isinstance(results.pred_masks, ROIMasks): roi_masks = results.pred_masks
            else: roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
            results.pred_masks = roi_masks.to_bitmasks(results.pred_boxes, size[0], size[1], mask_threshold).tensor
        return results.pred_masks

    def resize_depth(self, depth):
        depth = torch.squeeze(depth[:, :, :]).to("cpu")
        depth = cv2.resize(depth.numpy(), (self._cfg.UTILS.TARGET_SHAPE_W, self._cfg.UTILS.TARGET_SHAPE_H), interpolation=cv2.INTER_CUBIC)
        if not np.isfinite(depth).all():
            depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*1))-1
        depth = max_val * (depth - depth_min) / (depth_max - depth_min) if depth_max - depth_min > np.finfo("float").eps else np.zeros(depth.shape, dtype=depth.dtype)
        return depth

    def mean_pxl_depth(self, depth, mask):
        locs = np.where(mask)
        pixels = depth[locs]
        mean = np.mean(pixels)
        return mean

    def match_object(self, obj_dets, hand_bb, hand_dxdymag):
        object_cc_list = np.array([calculate_center(bbox) for bbox in obj_dets]) # object center list
        magn =  hand_dxdymag[2] / self._scale_factor
        hand_cc = np.array(calculate_center(hand_bb)) # hand center points
        point_cc = np.array([(hand_cc[0] + hand_dxdymag[0] * magn * self._diag), (hand_cc[1] + hand_dxdymag[1] * magn * self._diag)])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist) # find the nearest 
        return dist_min

    def generate_predictions(self, image_id, confident_instances, instances_hand, depth_map: torch.tensor = None, image: torch.tensor = None):
        results = []
        results_target = []
        depth_map = self.resize_depth(depth_map)
        objs = self.convert_instances_to_coco(confident_instances[confident_instances.pred_classes != self._id_hand], image_id)
        instances_hand = self.generate_confident_instances(instances_hand)
        masks_hand = self._mask_postprocess(confident_instances[confident_instances.pred_classes == self._id_hand], confident_instances.image_size)
        
        if len(instances_hand):
            masks_objs = self._mask_postprocess(confident_instances[confident_instances.pred_classes != self._id_hand], confident_instances.image_size)
            mean_objs_pxls = [self.mean_pxl_depth(depth_map, mask_obj) for mask_obj in masks_objs]

        for idx_hand in range(len(instances_hand)):
            instance_hand = instances_hand[idx_hand]
            bbox_hand = instance_hand.boxes.numpy()[0]
            x0_hand, y0_hand, x1_hand, y1_hand = bbox_hand
            width_hand, heigth_hand = x1_hand - x0_hand, y1_hand - y0_hand
            dxdymag_v = instance_hand.dxdymagn_hand.numpy()[0]
            contact_state = instance_hand.contact_states.item()
            score = instance_hand.scores.item()
            side = instance_hand.sides.item()

            mean_pxl_hand = self.mean_pxl_depth(depth_map, masks_hand[idx_hand])
            best_match_idx, best_mean = -1, 256
            for idx_obj, mean_pxl_obj in enumerate(mean_objs_pxls):
                if np.abs(mean_pxl_hand - mean_pxl_obj) < best_mean:
                    best_mean = np.abs(mean_pxl_hand - mean_pxl_obj)
                    best_match_idx = idx_obj

            contact_state = 1 if best_mean < 20 else contact_state
            contact_state = 0 if best_mean > 100 else contact_state

            element = {
                "image_id": image_id, 
                "category_id": 0, 
                "bbox": [x0_hand, y0_hand, width_hand, heigth_hand], 
                "score": score, 
                "hand_side": side, 
                "contact_state": contact_state, 
                "bbox_obj": [], 
                "category_id_obj": -1, 
                "dx":dxdymag_v[0],
                "dy": dxdymag_v[1],
                "magnitude": dxdymag_v[2] / self._scale_factor * self._diag
            }

            if contact_state and best_match_idx != -1:
                x0, y0, x1, y1 = objs[best_match_idx]["bbox"]
                width, heigth = x1 - x0, y1 - y0                
                element["bbox_obj"] = [x0, y0, width, heigth]
                element["category_id_obj"] = int(objs[best_match_idx]["category_id"])
                element["score_obj"] = objs[best_match_idx]["score"]
                results_target.append({"image_id": image_id, "category_id": element["category_id_obj"], "bbox": element["bbox_obj"], "score": element["score_obj"]})
            
            results.append(element)

        return results, results_target