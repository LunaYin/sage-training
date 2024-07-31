import os

import supervision as sv
import torch
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

ROOT_DIR = os.getcwd()
MODEL_ARCH = "yolo_nas_m"
BATCH_SIZE = 8
MAX_EPOCHS = 160
CHECKPOINT_DIR = f"{ROOT_DIR}/checkpoints"
EXPERIMENT_NAME = "my_yolo_run_v2"
CLASSES = ["diagram-EKfo"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINER = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)
LOCATION = f"{ROOT_DIR}/data"

CONFIDENCE_TRESHOLD = 0.35

dataset_params = {
    "data_dir": LOCATION,
    "train_images_dir": "train/images",
    "train_labels_dir": "train/labels",
    "val_images_dir": "valid/images",
    "val_labels_dir": "valid/labels",
    "test_images_dir": "test/images",
    "test_labels_dir": "test/labels",
    "classes": CLASSES,
}
train_params = {
    "silent_mode": False,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": 300,
    "mixed_precision": False,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params["classes"]),
        reg_max=16,
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params["classes"]),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        )
    ],
    "metric_to_watch": "mAP@0.50",
}
train_data = coco_detection_yolo_format_train(
    dataset_params={
        "data_dir": dataset_params["data_dir"],
        "images_dir": dataset_params["train_images_dir"],
        "labels_dir": dataset_params["train_labels_dir"],
        "classes": dataset_params["classes"],
    },
    dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 0},
)
val_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir": dataset_params["data_dir"],
        "images_dir": dataset_params["val_images_dir"],
        "labels_dir": dataset_params["val_labels_dir"],
        "classes": dataset_params["classes"],
    },
    dataloader_params={"batch_size": BATCH_SIZE, "num_workers": 0},
)
model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params["classes"]),
    pretrained_weights="coco",
    checkpoint_path=f"{ROOT_DIR}/{EXPERIMENT_NAME}/average_model.pth",
).to(DEVICE)

TRAINER.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data,
)
