import contextlib
import torch
import os
import math
import numpy as np
import copy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.custom_handside_cocoeval import CustomHandSideCOCOeval
from pycocotools.custom_handstate_cocoeval import CustomHandContactStateCOCOeval
from pycocotools.custom_hand_target_object_w_classification import CustomHandTargetCOCOeval
from pycocotools.custom_hand_all_cocoeval_w_classification import CustomHandAllCOCOeval

from detectron2.data import MetadataCatalog
from detectron2.utils.custom_utils import get_iou
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from torchvision.ops.boxes import nms

class EHOIEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, converter):
        self._cfg = cfg
        self._output_dir = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)

        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_classes = self._metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self._class_names_objs = copy.deepcopy(self._metadata.as_dict()["thing_classes"])
        self._class_names_objs.pop(self._id_hand)

		###converter
        self._converter = converter 

        self._coco_gt = COCO(PathManager.get_local_path(self._metadata.coco_gt_hands))
        self._coco_gt_all = COCO(PathManager.get_local_path(self._metadata.json_file))
        self._coco_gt_targets = self._converter.convert_coco_to_coco_target_object(self._coco_gt, self._coco_gt_all)
        self._coco_gt_targets.createIndex()

    def reset(self):
        self._predictions = []
        self._predictions_all = []
        self._predictions_targets = []
        
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(torch.device("cpu"))
            confident_instances = self._converter.generate_confident_instances(instances)
            predictions, predictions_target = self._converter.generate_predictions(image_id, confident_instances, output["additional_outputs"])
            self._predictions += predictions
            self._predictions_targets+= predictions_target
            confident_instances_all = self._converter._nms(instances)
            self._predictions_all.extend(self._converter.convert_instances_to_coco(confident_instances_all, image_id, convert_boxes_xywh_abs = True))

    def evaluate(self):
        cocoPreds = self._coco_gt.loadRes(self._predictions)
        cocoPreds_all = self._coco_gt_all.loadRes(self._predictions_all)
        if(len(self._predictions_targets)):
            cocoPreds_target = self._coco_gt_targets.loadRes(self._predictions_targets)

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions.pth"), "wb") as f:
                torch.save(self._predictions, f)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions_all.pth"), "wb") as f:
                torch.save(self._predictions_all, f)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions_targets.pth"), "wb") as f:
                torch.save(self._predictions_targets, f)

        annType = 'bbox'
        coco_results = {}

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):    
            ##### HAND
            cocoEval = COCOeval(self._coco_gt_all, cocoPreds_all, annType)
            cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            cocoEval.params.iouThrs = np.array([0.5])
            cocoEval.params.catIds = [self._id_hand]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_results["AP Hand"] = round(cocoEval.stats[0] * 100, 2)

            ##### OBJECTS
            cocoEval = COCOeval(self._coco_gt_all, cocoPreds_all, annType)
            cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            cocoEval.params.iouThrs = np.array([0.5])
            cocoEval.params.catIds = [x["id"] for x in self._coco_gt_all.cats.values() if x["id"] != self._id_hand]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_results["mAP Objects"] = round(cocoEval.stats[0] * 100, 2)

            ##### TARGET OBJECTS
            if len(self._predictions_targets):
                cocoEval = COCOeval(self._coco_gt_targets, cocoPreds_target, annType)
                cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
                cocoEval.params.iouThrs = np.array([0.5])
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                coco_results["mAP Target Objects"] = round(cocoEval.stats[0] * 100, 2)

            ##### HAND + SIDE
            customHandSideCOCOeval = CustomHandSideCOCOeval(self._coco_gt, cocoPreds, annType)
            customHandSideCOCOeval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            customHandSideCOCOeval.params.iouThrs = np.array([0.5])
            customHandSideCOCOeval.evaluate()
            customHandSideCOCOeval.accumulate()
            customHandSideCOCOeval.summarize()
            coco_results["AP Hand + Side"] = round(customHandSideCOCOeval.stats[0] * 100, 2)

            ##### HAND + CONTACT_STATE
            customHandContactStateCOCOeval = CustomHandContactStateCOCOeval(self._coco_gt, cocoPreds, annType)
            customHandContactStateCOCOeval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            customHandContactStateCOCOeval.params.iouThrs = np.array([0.5])
            customHandContactStateCOCOeval.evaluate()
            customHandContactStateCOCOeval.accumulate()
            customHandContactStateCOCOeval.summarize()
            coco_results["AP Hand + State"] = round(customHandContactStateCOCOeval.stats[0] * 100, 2)
            
            ###COPY GT AND PRED
            cocoGT_filtred = copy.deepcopy(self._coco_gt)
            cocoPreds_filtred = copy.deepcopy(cocoPreds)

            ##### mAP HAND + CONTACT_STATE
            tmp_results = {}

            for idx_class, class_name in enumerate(self._class_names_objs):  
                cocoGT_filtred.dataset["annotations"] = [ann for _, ann in enumerate(self._coco_gt.dataset["annotations"]) if ann["category_id_obj"] == idx_class]
                cocoGT_filtred.createIndex()
                new_anns = []
                for ann_gt in cocoGT_filtred.anns.values():
                    new_anns += [cocoPreds.anns[ann_pred_id] for ann_pred_id in cocoPreds.getAnnIds(imgIds=ann_gt["image_id"]) if get_iou(ann_gt["bbox"], cocoPreds.anns[ann_pred_id]["bbox"], type = "xywh") >= 0.5]

                cocoPreds_filtred.dataset["annotations"] = new_anns
                cocoPreds_filtred.createIndex()
                cocoEval = CustomHandContactStateCOCOeval(cocoGT_filtred, cocoPreds_filtred, annType)
                cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
                cocoEval.params.iouThrs = np.array([0.5])
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                tmp_results[class_name] = round(cocoEval.stats[0] * 100, 2) if round(cocoEval.stats[0] * 100, 2) > 0 else 0

            cocoGT_filtred.dataset["annotations"] = [ann for ann_idx, ann in enumerate(self._coco_gt.dataset["annotations"]) if ann["contact_state"] == 0]
            cocoGT_filtred.createIndex()
            new_anns = []
            for ann_gt in cocoGT_filtred.anns.values():
                new_anns += [cocoPreds.anns[ann_pred_id] for ann_pred_id in cocoPreds.getAnnIds(imgIds=ann_gt["image_id"]) if get_iou(ann_gt["bbox"], cocoPreds.anns[ann_pred_id]["bbox"], type = "xywh") >= 0.5]

            cocoPreds_filtred.dataset["annotations"] = new_anns
            cocoPreds_filtred.createIndex()
            cocoEval = CustomHandContactStateCOCOeval(cocoGT_filtred, cocoPreds_filtred, annType)
            cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            cocoEval.params.iouThrs = np.array([0.5])
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            tmp_results["no_contact"] = round(cocoEval.stats[0] * 100, 2) if round(cocoEval.stats[0] * 100, 2) > 0 else 0

            list_results = np.array([value for _, value in tmp_results.items()])        
            coco_results["mAP Hand + State"] = round(list_results.mean(), 2)
            
            ##### HAND + TARGET
            tmp_results = {}
            for idx_class, class_name in enumerate(self._class_names_objs):  
                cocoGT_filtred.dataset["annotations"] = [ann for _, ann in enumerate(self._coco_gt.dataset["annotations"]) if ann["category_id_obj"] == idx_class]
                cocoGT_filtred.createIndex()
                cocoPreds_filtred.dataset["annotations"] = [ann for _, ann in enumerate(cocoPreds.dataset["annotations"]) if ann["category_id_obj"] == idx_class]
                cocoPreds_filtred.createIndex()
                cocoEval = CustomHandTargetCOCOeval(cocoGT_filtred, cocoPreds_filtred, annType)
                cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
                cocoEval.params.iouThrs = np.array([0.5])
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                tmp_results[class_name] = round(cocoEval.stats[0] * 100, 2) if round(cocoEval.stats[0] * 100, 2) > 0 else 0
                if len(cocoGT_filtred.anns) == 0: tmp_results[class_name] = 1 if len(cocoPreds_filtred.anns) == 0 else 0
            coco_results["mAP Hand + Target Objects"] = round(np.array([tmp_results[class_name] for idx_class, class_name in enumerate(self._class_names_objs)]).mean(), 2)
            
            ##### HAND + ALL
            for idx_class, class_name in enumerate(self._class_names_objs):  
                cocoGT_filtred.dataset["annotations"] = [ann for _, ann in enumerate(self._coco_gt.dataset["annotations"]) if ann["category_id_obj"] == idx_class]
                cocoGT_filtred.createIndex()
                cocoPreds_filtred.dataset["annotations"] = [ann for _, ann in enumerate(cocoPreds.dataset["annotations"]) if ann["category_id_obj"] == idx_class]
                cocoPreds_filtred.createIndex()
                cocoEval = CustomHandAllCOCOeval(cocoGT_filtred, cocoPreds_filtred, annType)
                cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
                cocoEval.params.iouThrs = np.array([0.5])
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                coco_results[class_name] = round(cocoEval.stats[0] * 100, 2) if round(cocoEval.stats[0] * 100, 2) > 0 else 0
                if len(cocoGT_filtred.anns) == 0: coco_results[class_name] = 1 if len(cocoPreds_filtred.anns) == 0 else 0

            ### AP HAND + ALL
            cocoGT_filtred.dataset["annotations"] = [ann for _, ann in enumerate(self._coco_gt.dataset["annotations"]) ]
            cocoGT_filtred.createIndex()
            cocoPreds_filtred.dataset["annotations"] = [ann for _, ann in enumerate(cocoPreds.dataset["annotations"])]
            cocoPreds_filtred.createIndex()
            cocoEval = CustomHandAllCOCOeval(cocoGT_filtred, cocoPreds_filtred, annType, agn = True)
            cocoEval.params.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .1)) + 1, endpoint=True)
            cocoEval.params.iouThrs = np.array([0.5])
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_results["AP All"] = round(cocoEval.stats[0] * 100, 2)
        
        list_results = np.array([coco_results[class_name] for _, class_name in enumerate(self._class_names_objs)])        
        coco_results["mAP All"] = round(list_results.mean(), 2)

        return {"ehoi" : coco_results}