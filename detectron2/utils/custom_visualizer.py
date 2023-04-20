from abc import abstractmethod
import numpy as np
import cv2
import copy
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures.masks import ROIMasks

from detectron2.utils.converters import Converter

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class BaseEhoiVisualizer:
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        self._thing_classes = metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self.cfg = cfg
        self.class_names = {v: metadata.thing_classes[k] for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        self._input_size = (cfg.UTILS.TARGET_SHAPE_W, cfg.UTILS.TARGET_SHAPE_H)

        ###PARAMS
        self._converter = converter
        self._draw_ehoi = self.cfg.UTILS.VISUALIZER.DRAW_EHOI
        self._draw_masks = self.cfg.UTILS.VISUALIZER.DRAW_MASK
        self._draw_objs = self.cfg.UTILS.VISUALIZER.DRAW_OBJS
        self._draw_depth = self.cfg.UTILS.VISUALIZER.DRAW_DEPTH and cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE
        self.create_colors()

    def create_colors(self):
        colors = np.array([(190, 209, 18), (108, 233, 247), (255, 188, 73), (221, 149, 42), (191, 80, 61), (144, 183, 3), (14, 160, 41), (75, 229, 96), (78, 80, 183), (35, 33, 150), (103, 252, 103), (38, 116, 193), (72, 52, 153), (51, 198, 154), (191, 70, 22), (160, 14, 29), (150, 242, 101), (214, 17, 30), (11, 229, 142), (190, 234, 32)], np.uint8 )
        self._colors_classes = {k: v for k, v in enumerate(colors)}
            
    @abstractmethod
    def _draw_masks_f(self, *args, **kwargs):
        pass

    @abstractmethod
    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5, *args, **kwargs):
        pass

    def _draw_ehoi_f(self, image, predictions, additional_outputs):
        predictions_hands, _ = self._converter.generate_predictions("", predictions, additional_outputs)
        if not len(predictions_hands): return
        
        annotations_active_objs = [x for x in copy.deepcopy(predictions_hands) if x["contact_state"] and x["category_id_obj"] != -1]
        for element in annotations_active_objs:
            x,y,w,h = np.array(element['bbox_obj'], int)
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)   

        for element in predictions_hands:
            x,y,w,h = np.array(element['bbox'], int)
            hand_side = element["hand_side"]
            hand_state = element["contact_state"]
    
            if hand_state == 1: color = (0, 255, 0)
            else: color = (0, 0, 255)    
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), color, 2)
            cv2.rectangle(image, (x, y), ((x+w), y+15), (255,255,255), -1)

            if hand_side == 1: cv2.putText(image, f'Right Hand {round(element["score"] * 100, 2)} %', (x + 5, y + 11), 1,  1, (0, 0, 0), 1, cv2.LINE_AA)        
            else: cv2.putText(image, f'Left Hand {round(element["score"] * 100, 2)} %', (x + 5, y + 11), 1,  1, (0, 0, 0), 1, cv2.LINE_AA)         
    
            if hand_state and element["category_id_obj"] != -1:
                obj_box = np.array(element['bbox_obj'], int)
                hand_cc = (x + w//2, y + h//2)
                point_cc = (obj_box[0] + obj_box[2]//2, obj_box[1] + obj_box[3]//2)
                cv2.line(image, hand_cc, point_cc, (0, 255, 0), 4)
                cv2.circle(image, hand_cc, 4, (0, 0, 255), -1)
                cv2.circle(image, point_cc, 4, (0, 255, 0), -1)
        return image

    def _draw_objs_f(self, image, predictions):
        predictions_obj = predictions[predictions.pred_classes != self._id_hand]
        predictions_objs = self._converter.convert_instances_to_coco(predictions_obj, "", convert_boxes_xywh_abs=True)
        for element in predictions_objs:
            x,y,w,h = np.array(element['bbox'], int)
            class_name = self.class_names[element['category_id']] + " " + str(round(element["score"] * 100, 2)) + " %"
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (128,128,128), 1)        
            cv2.rectangle(image, (x, y), ((x+w), y+15), (255,255,255), -1) 
            cv2.putText(image, class_name, (x + 5, y + 11), 1,  1, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    def _draw_depth_f(self, image, outputs, **kwargs):
        depth = cv2.resize(outputs["depth_map"].detach().to("cpu").numpy().transpose(1, 2, 0), self._input_size)
        depth = np.array(depth).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        if "save_depth_map" in kwargs.keys() and kwargs["save_depth_map"]: cv2.imwrite(kwargs["save_depth_map_path"], depth)
        image =  np.concatenate((image, depth), axis=0)
        return image

    def draw_results(self, image_, outputs, **kwargs):
        image = cv2.resize(image_, self._input_size)
        predictions = outputs["instances"].to("cpu")
        confident_instances = self._converter.generate_confident_instances(predictions)
        additional_outputs = outputs["additional_outputs"].to("cpu")
        confident_additional_outputs = self._converter.generate_confident_instances(additional_outputs)

        if self._draw_masks and predictions.has("pred_masks"): 
            image = self._draw_masks_f(image, confident_instances, confident_additional_outputs, **kwargs)
        if self._draw_ehoi: 
            image = self._draw_ehoi_f(image, confident_instances, confident_additional_outputs)
        if self._draw_objs: 
            image = self._draw_objs_f(image, confident_instances)
        if self._draw_depth:
            image = self._draw_depth_f(image, outputs, **kwargs)

        return image

class EhoiVisualizerv1(BaseEhoiVisualizer):
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        super().__init__(cfg, metadata, converter, **kwargs)
        ###PARAMS
        self._draw_keypoints = self.cfg.UTILS.VISUALIZER.DRAW_KEYPOINTS

    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5):
        results = Instances(size, **results.get_fields())
        if results.has("pred_masks"):
            if isinstance(results.pred_masks, ROIMasks): roi_masks = results.pred_masks
            else: roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
            results.pred_masks = roi_masks.to_bitmasks(results.pred_boxes, size[0], size[1], mask_threshold).tensor
        return results.pred_masks

    def _draw_masks_f(self, image, predictions, additional_outputs, save_masks = False, save_masks_path = "./masks.png", **kwargs):
        masks = self._mask_postprocess(predictions, predictions.image_size)
        masked_img = np.zeros(image.shape)

        for idx, mask in enumerate(masks):
            id_class = predictions[idx].pred_classes.item()
            masked_img = np.where(mask[...,None], self._colors_classes[id_class], masked_img)
            
        if save_masks: cv2.imwrite(save_masks_path, masked_img)
        masked_img = np.where(masked_img != 0, masked_img, image)
        image = cv2.addWeighted(image, 0.4, np.asarray(masked_img, np.uint8), 0.6, 0)
        return image

    def _draw_keypoints_f(self, image):
        raise NotImplementedError

    def _draw_vector(self, image, annotations_hands):
        for element in annotations_hands:
            x,y,w,h = np.array(element['bbox'], int)
            dx, dy, magn = float(element['dx']), float(element['dy']), float(element['magnitude'])
            hand_cc = np.array([x + w//2, y + h//2])
            point_cc = np.array([(hand_cc[0] + dx * magn), (hand_cc[1] + dy * magn)])
            cv2.line(image, tuple(hand_cc.astype(int)), tuple(point_cc.astype(int)), (255, 0, 0), 4)
            cv2.circle(image, tuple(hand_cc.astype(int)), 4, (255, 0, 255), -1)
        return image

    def _draw_ehoi_f(self, image, predictions, additional_outputs):
        predictions_hands, _ = self._converter.generate_predictions("", predictions, additional_outputs)
        if not len(predictions_hands): return image
        image = self._draw_vector(image, predictions_hands)
        return super()._draw_ehoi_f(image, predictions, additional_outputs)