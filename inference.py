# import some common libraries
import numpy as np
import cv2
import random
import os
import torch
from typing import Dict, List
import argparse
import sys

# import some common detectron2 utilities
from detectron2.config import CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog,  SimpleMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.data.ehoi_dataset_mapper_v1 import *
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.custom_visualizer import *
from detectron2.utils.converters import *

##### ArgumentParser
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--dataset', dest='ref_dataset_json', help='reference json', default='./data/ref_enigma.json', type=str)
parser.add_argument('-w', '--weights_path', dest='weights', help='weights path', type=str, required = True)
parser.add_argument('--cfg_path', dest='cfg_path', help='cfg .yaml path', type=str)
parser.add_argument('--nms', dest='nms', help='nms', default = 0.3, type=float)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--images_path', type=str, help='directory/file to load images')
parser.add_argument('--video_path', type=str, help='video to process')
parser.add_argument('--save_dir', type=str, help='directory to save results', default = "./output_detection/")
parser.add_argument('--skip_the_fist_frames', type=int, help='skip the first n frames of the video.', default = 0)
parser.add_argument('--duration_of_record_sec', type=int, help='time (seconds) of the video to process', default = 10000000)
parser.add_argument('--hide_depth', action='store_true', default=False)
parser.add_argument('--hide_ehois', action='store_true', default=False)
parser.add_argument('--hide_bbs', action='store_true', default=False)
parser.add_argument('--hide_masks', action='store_true', default=False)
parser.add_argument('--save_masks', action='store_true', help='save masks of the image (only supported for images)', default=False)
parser.add_argument('--save_depth_map', action='store_true', help='save depth of the image (only supported for images)', default=False)
parser.add_argument('--thresh', help='thresh of the score', default=0.5, type = float)

args = parser.parse_args()

def format_times(times_dict):
    str_ = ""
    for k, v in times_dict.items():
        str_+= f"\t{k}: {v} ms\t\n"
    return str_

def clear_output(str_):
    for i in range(str_.count('\n') + 1):
        sys.stdout.write("\033[K\033[F")

def load_cfg():
    register_coco_instances("val_set", {}, args.ref_dataset_json, args.ref_dataset_json)
    _ = DatasetCatalog.get("val_set")
    metadata = MetadataCatalog.get("val_set")
    weights_path = args.weights
    cfg_path = os.path.join(weights_path.split("model_")[0], "cfg.yaml") if not args.cfg_path else args.cfg_path

    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_path))
    cfg.set_new_allowed(True)
    cfg.DATASETS.TEST = ("val_set",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.as_dict()["thing_dataset_id_to_contiguous_id"])
    cfg.MODEL.WEIGHTS = weights_path
    cfg.OUTPUT_DIR = "./output_dir/test"
    
    cfg.ADDITIONAL_MODULES.NMS_THRESH = args.nms
    cfg.UTILS.VISUALIZER.THRESH_OBJS = args.thresh
    cfg.UTILS.VISUALIZER.DRAW_EHOI = not args.hide_ehois
    cfg.UTILS.VISUALIZER.DRAW_MASK = not args.hide_masks
    cfg.UTILS.VISUALIZER.DRAW_OBJS = not args.hide_bbs
    cfg.UTILS.VISUALIZER.DRAW_DEPTH = not args.hide_depth

    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    cfg.freeze()

    return cfg, metadata

def main():
    kwargs = {}
    kwargs["cuda_device"] = args.cuda_device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg, metadata = load_cfg()

    ###INIT MODEL
    converter = MMEhoiNetConverterv1(cfg, metadata)
    model = MMEhoiNetv1(cfg, metadata)

    ####INIT MODEL
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = "cuda:" + str(args.cuda_device) if not args.no_cuda else "cpu"
    model.to(device)
    model.eval()
    print("Modello caricato:", model.device)

    #VISUALIZER AND MAPPER
    visualizer = EhoiVisualizerv1(cfg, metadata, converter, ** kwargs)
    mapper = SimpleMapper(cfg)

    os.makedirs(args.save_dir, exist_ok = True)
    
    with torch.no_grad():

        ###PROC IMAGES
        if args.images_path:
            
            #### kwargs init
            kwargs = {}
            kwargs["save_masks"] = args.save_masks
            kwargs["save_depth_map"] = args.save_depth_map
            if args.save_masks: 
                os.makedirs(os.path.join(args.save_dir, "masks_processed/"), exist_ok = True)
            if args.save_depth_map:
                os.makedirs(os.path.join(args.save_dir, "depth_maps_processed/"), exist_ok = True)
            save_dir_images = os.path.join(args.save_dir, "images_processed")
            os.makedirs(save_dir_images, exist_ok = True)
            
            ####IMAGE
            if args.images_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(args.images_path)
                r_ = model([mapper(image)])
                print(f"Inference times:\n {format_times(model._last_inference_times)}")

                ####kwargs
                if args.save_masks: 
                    kwargs["save_masks_path"] = os.path.join(args.save_dir, "masks_processed/" + args.images_path.split("/")[-1].split(".")[0] + "_masks.png")
                if args.save_depth_map: 
                    kwargs["save_depth_map_path"] = os.path.join(args.save_dir, "depth_maps_processed/" + args.images_path.split("/")[-1].split(".")[0] + "_depth_map.png")

                im_r = visualizer.draw_results(image, r_[0], **kwargs)
                filename = args.images_path.split("/")[-1]
                save_path = os.path.join(save_dir_images, filename)
                cv2.imwrite(save_path, im_r)

            elif os.path.isdir(args.images_path):
                n_files = len(os.listdir(args.images_path))
                for id_, file in enumerate(os.listdir(args.images_path)):
                    if args.save_masks: 
                        kwargs["save_masks_path"] = os.path.join(args.save_dir, "masks_processed/" + file.split(".")[0] + "_masks.png")
                    if args.save_depth_map: 
                        kwargs["save_depth_map_path"] = os.path.join(args.save_dir, "depth_maps_processed/" + file.split(".")[0] + "_depth_map.png")
                    
                    image = cv2.imread(os.path.join(args.images_path, file))
                    r_ = model([mapper(image)])
                    msg = f"\nfile checked: {id_} of {n_files} \t\nInference times:\n{format_times(model._last_inference_times)}"
                    print(msg)
                    clear_output(msg)
                    im_r = visualizer.draw_results(image, r_[0], **kwargs)
                    save_path = os.path.join(save_dir_images, file)
                    cv2.imwrite(save_path, im_r)
                print(msg)

        if args.video_path:
            save_dir_videos = os.path.join(args.save_dir, "videos_processed")
            os.makedirs(save_dir_videos, exist_ok = True)
            ####VIDEO
            if args.video_path.lower().endswith(('.mp4', '.avi')):
                filename = args.video_path.split("/")[-1]
                save_video_path = os.path.join(save_dir_videos, "p_" + filename)
                current_video_cap = cv2.VideoCapture(args.video_path)
                ret, frame = current_video_cap.read()
                r_ = model([mapper(frame)])
                im_r = visualizer.draw_results(frame, r_[0])
                current_video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), current_video_cap.get(5), (im_r.shape[1], im_r.shape[0]))
                duration_of_record_frames = args.duration_of_record_sec * current_video_cap.get(5) + args.skip_the_fist_frames
                while(ret):
                    frame_number = int(current_video_cap.get(1)) - 1
                    if frame_number > args.skip_the_fist_frames:
                        r_ = model([mapper(frame)])
                        msg = f"frame nr: {frame_number  - args.skip_the_fist_frames} of {int(min(current_video_cap.get(7) - args.skip_the_fist_frames, duration_of_record_frames  - args.skip_the_fist_frames))} \t\nInference times\n{format_times(model._last_inference_times)}"
                        print(msg)
                        clear_output(msg)
                        im_r = visualizer.draw_results(frame, r_[0])
                        current_video_writer.write(im_r.astype(np.uint8))
                    if frame_number >= duration_of_record_frames: break
                    ret, frame = current_video_cap.read()

                current_video_cap.release()
                current_video_writer.release()

    print("Done.")

if __name__ == "__main__":
    main()