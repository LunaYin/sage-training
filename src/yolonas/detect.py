import random

import cv2
import supervision as sv
import torch
from super_gradients.training import models

MODEL_ARCH = "yolo_nas_m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.get(MODEL_ARCH, num_classes=1, pretrained_weights="coco").to(DEVICE)

best_model = models.get(
    MODEL_ARCH,
    num_classes=1,
    checkpoint_path="/Users/i307132/Desktop/yolonas/src/checkpoints/my_yolo_run/ckpt_best.pth",
).to(DEVICE)
IMAGE_URL = "/Users/i307132/Desktop/yolonas/src/data/test/images/-_page_1_png.rf.33b0ca09dd81d5e5570396a93a4a6995.jpg"
best_model.predict(IMAGE_URL, conf=0.3).show()
# ds = sv.DetectionDataset.from_yolo(
#     images_directory_path="/Users/i307132/Desktop/yolonas/src/data/test/images",
#     annotations_directory_path="/Users/i307132/Desktop/yolonas/src/data/test/labels",
#     data_yaml_path="/Users/i307132/Desktop/yolonas/src/data/data.yaml",
#     force_masks=False,
# )
# predictions = {}
# for image_name, image in ds.images.items():
#     result = list(best_model.predict(image, conf=0.4))[
#     detections = sv.Detections(
#         xyxy=result.prediction.bboxes_xyxy,
#         confidence=result.prediction.confidence,
#         class_id=result.prediction.labels.astype(int),
#     )
#     predictions[image_name
