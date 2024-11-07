import argparse
import multiprocessing as mp
import numpy as np
import os
import torch
import tempfile
import cv2
import sys
import h5py
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
import detectron2.data.detection_utils as utils
from detic.predictor import VisualizationDemo
from dataset_utils.annotation import ImageTextAnnotation

from torchvision.ops import MultiScaleRoIAlign

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
    # NOTE: https://pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = args.nms_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.TEST.DETECTIONS_PER_IMAGE = args.detections_per_image
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 extract region features for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--root-dir",
        required=True,
        type=str,
        help="path to input/output annotation dir",
    )
    parser.add_argument(
        "--dataset-name", required=True, type=str, choices=["jcre3", "f30k_ent_jp"]
    )
    parser.add_argument(
        "--output-file-name",
        required=True, type=str, default="default",
        help="A hdf file name to save output features.",
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
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.9,
        help="Maximum score for non-maximum suppression to limit object duplicates",
    )
    parser.add_argument(
        "--detections-per-image",
        type=int,
        default=256,
        help="Maximum number of objects to detect"
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

def get_inputs(predictor, original_image) -> dict:
    """Given a image return a list of dictionary with each dict corresponding to an image
    (refer to detectron2/engien/defaults.py)
    """
    # Apply pre-processing to image.
    if predictor.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = predictor.aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image.to(predictor.cfg.MODEL.DEVICE)
    return {"image": image, "height": height, "width": width}


def extract_region_feats(model, batched_inputs) -> dict:
    assert not model.training
    im_id = 0  # single image
    # model inference
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)
    proposals, _ = model.proposal_generator(images, features, None)

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=model.roi_heads.in_features,
        output_size=2,
        sampling_ratio=2,
    )
    proposal_boxes = [x.proposal_boxes.tensor for x in proposals]
    box_features = box_roi_pool(features, proposal_boxes, images.image_sizes)
    results, kept_indices = model.roi_heads(images, features, proposals)
    results = model._postprocess(results, batched_inputs, images.image_sizes)

    # save detection outputs into files
    boxes = (
        results[im_id]["instances"].get("pred_boxes").tensor.cpu()
    )  # boxes after per-class NMS, [#boxes, 4]
    scores = (
        results[im_id]["instances"].get("scores").cpu()
    )  # scores after per-class NMS, [#boxes]
    classes = (
        results[im_id]["instances"].get("pred_classes").cpu()
    )  # class predictions after per-class NMS, [#boxes], class value in [0, C]
    region_feats = box_features[
        kept_indices[0]
    ].cpu() # region features, [#boxes, feats(#proposal_boxes, out_size, out_size) ]

    # save features of detection regions (after per-class NMS)
    saved_dict = {}
    saved_dict["boxes"] = boxes
    saved_dict["scores"] = scores
    saved_dict["classes"] = classes
    saved_dict["feats"] = torch.flatten(
        region_feats, start_dim=1,
    )  # [#boxes, #proposal_boxes * out_size * out_size]

    return saved_dict


def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    export = VisualizationDemo(cfg, args)

    visual_dir = Path(args.root_dir) / "image_text_annotation"
    image_root = Path(args.root_dir) / "recording"
    output_root = Path(args.root_dir)

    visual_paths = visual_dir.glob("*.json")
    image_ext = "png"
    if args.dataset_name == "f30k_ent_jp":
        image_ext = "jpg"
    with h5py.File(output_root / f"{args.output_file_name}.h5", mode="w") as output_fp:
        for source in visual_paths:
            scenario_id = source.stem
            logger.info(
                f"[ScenarioID: {scenario_id}] Running object feature extraction"
            )
            image_dir = image_root / scenario_id / "images"
            image_text_annotation = ImageTextAnnotation.from_json(
                Path(source).read_text()
            )
            image_files = [
                (image_dir / f"{image.imageId}.{image_ext}")
                for image in image_text_annotation.images
            ]
            # extract features per images
            for image_idx, img_file_name in enumerate(image_files):
                image_id = img_file_name.stem
                # image_id = img_file_name.stem
                image = utils.read_image(img_file_name, format="BGR")
                #  predictions = export.predictor(image)
                with torch.no_grad():
                    batched_inputs = get_inputs(export.predictor, image)
                    output = extract_region_feats(export.predictor.model, [batched_inputs])

                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/boxes", data=output["boxes"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/scores", data=output["scores"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/classes", data=output["classes"]
                )
                output_fp.create_dataset(
                    f"{scenario_id}/{image_id}/feats", data=output["feats"]
                )
                # NOTE: 配列としてアクセス可能
                # e.g.) output_fp[f"{scenario_id}/{image_id}/boxes"][0]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

