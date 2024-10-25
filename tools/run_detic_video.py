# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import numpy as np
import os
import tempfile
import warnings
import cv2
from tqdm import tqdm
import sys
import pickle

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 export bbox for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_CXT21k_640b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", required=True, help="Path to video file.")
    parser.add_argument(
        "--output",
        required=True,
        help="A scenario id file to save output bounding boxes.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    export = VisualizationDemo(cfg, args)

    video = cv2.VideoCapture(args.video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # basename = os.path.basename(args.video_input)

    codec, file_ext = (
        ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )
    if codec == ".mp4v":
        warnings.warn("x264 codec not available, switching to mp4v")

    export_obj = []
    for frame in tqdm(frame_from_video(video), total=num_frames):
        predictions = export.predictor(frame)["instances"].to(export.cpu_device)
        boxes_list = predictions._fields["pred_boxes"].tensor.tolist()
        scores_list = predictions._fields["scores"].tolist()
        classes_list = predictions._fields["pred_classes"].tolist()

        fields_list = []
        assert len(boxes_list) == len(scores_list) == len(classes_list)
        for box, score, _cls in zip(boxes_list, scores_list, classes_list):
            fields = (box[0], box[1], box[2], box[3], score, _cls)  # x1, y1, x2, y2, score, class
            fields_list.append(fields)
        export_obj.append(np.array(fields_list))

    with open(args.output, mode="wb") as f:
        pickle.dump(export_obj, f)
    video.release()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

