import argparse
import comet_ml
import torch
import time
import yaml
import os
from pathlib import Path

from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.common import MultiGPUMode
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients import init_trainer
from super_gradients.training import Trainer
from super_gradients.training import models


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", type=str, required=True, help="path to data.yaml")
    ap.add_argument("-n", "--name", type=str, help="Checkpoint dir name")
    ap.add_argument("-b", "--batch", type=int, default=8, help="Training batch size")
    ap.add_argument(
        "-e", "--epoch", type=int, default=100, help="Training number of epochs"
    )
    ap.add_argument(
        "-j", "--worker", type=int, default=8, help="Training number of workers"
    )
    ap.add_argument(
        "-w",
        "--weight",
        type=str,
        help="path to pre-trained model weight",
    )
    ap.add_argument("-s", "--size", type=int, default=1024, help="input image size")
    ap.add_argument("--num_gpus", type=int, default=1, help="Gpu count")
    ap.add_argument(
        "--no_cache_indexing",
        action="store_true",
        help="Do not perform dataset cache indexing",
    )
    ap.add_argument("--cpu", action="store_true", help="Run on CPU")

    args = vars(ap.parse_args())
    return args


def train(args, save_dir, name, yaml_params):
    init_trainer()
    base_data_dir = yaml_params["path"]
    if base_data_dir.endswith("yolo"):
        base_data_dir = base_data_dir[:-5]

    # Training on GPU or CPU
    if args["cpu"]:
        print("[INFO] Training on \033[1mCPU\033[0m")
        setup_device(device="cpu")
    elif args["num_gpus"] > 1:
        print(f"[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m")
        setup_device(
            multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, num_gpus=args["num_gpus"]
        )
    else:
        print(f"[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m")
        setup_device(device="cuda")
    trainer = Trainer(experiment_name=name, ckpt_root_dir=save_dir)

    if args["no_cache_indexing"]:
        cache_indexing = False
    else:
        cache_indexing = True
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": yaml_params["path"],
            "images_dir": yaml_params["train"],
            "labels_dir": yaml_params["train"],
            "classes": yaml_params["names"],
            "input_dim": (args["size"], args["size"]),
            "ignore_empty_annotations": False,
            "cache_annotations": cache_indexing,
        },
        dataloader_params={
            "batch_size": args["batch"],
            "num_workers": args["worker"],
        },
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": yaml_params["path"],
            "images_dir": yaml_params["val"],
            "labels_dir": yaml_params["val"],
            "classes": yaml_params["names"],
            "input_dim": (args["size"], args["size"]),
            "ignore_empty_annotations": False,
            "cache_annotations": cache_indexing,
        },
        dataloader_params={
            "batch_size": args["batch"],
            "num_workers": args["worker"],
        },
    )

    model = models.get(
        "yolo_nas_l",
        num_classes=len(yaml_params["names"]),
        pretrained_weights=args["weight"],
    )

    train_params = {
        # ENABLING SILENT MODE
        "save_ckpt_epoch_list": [35, 55, 65, 75, 85, 90],
        "silent_mode": False,
        "average_best_models": False,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "SGD",
        "optimizer_params": {"weight_decay": 0.0005},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": args["epoch"],
        "run_validation_freq": 10,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False, num_classes=len(yaml_params["names"]), reg_max=16
        ),
        "loss_logging_items_names": ["Loss"],
        "transforms": [
            {
                "DetectionRandomAffine": {
                    "degrees": 0.0,
                    "translate": 0.1,
                    "scales": [0, 0.5],
                    "shear": 0.0,
                    "target_size": [args["size"], args["size"]],
                    "filter_box_candidates": True,
                    "wh_thr": 2,
                    "area_thr": 0.1,
                    "ar_thr": 20,
                }
            },
            {"DetectionHSV": {"prob": 0.5, "hgain": 0.015, "sgain": 0.7, "vgain": 0.4}},
            {"DetectionHorizontalFlip": {"prob": 0.5}},
            {
                "DetectionTargetsFormatTransform": {
                    "input_dim": [args["size"], args["size"]],
                    "output_format": "LABEL_CXCYWH",
                }
            },
        ],
        # CometML
        "sg_logger": "cometml_sg_logger",
        "sg_logger_params": {
            "project_name": "yolo-nas",
            "experiment_name": name,
            "base_data_dir": base_data_dir,
            "checkpoints_dir_path": save_dir,
            "save_checkpoints_remote": True,
            "save_tensorboard_remote": False,
            "save_logs_remote": True,
        },
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.5,
                top_k_predictions=300,
                num_cls=len(yaml_params["names"]),
                normalize_targets=True,
                calc_best_score_thresholds=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
            DetectionMetrics_050_095(
                score_thres=0.5,
                top_k_predictions=300,
                num_cls=len(yaml_params["names"]),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
        ],
        "metric_to_watch": "mAP@0.50",
        "greater_metric_to_watch_is_better": True,
    }

    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data,
    )

    checkpoint_root_dir = trainer.ckpt_root_dir
    return checkpoint_root_dir


def validate(args, name, save_dir, best_model, yaml_params):
    init_trainer()

    # Training on GPU or CPU
    if args["cpu"]:
        print("[INFO] Training on \033[1mCPU\033[0m")
        setup_device(device="cpu")
    else:
        print(f"[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m")
        setup_device(device="cuda")
    trainer = Trainer(experiment_name=name, ckpt_root_dir=save_dir)

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": yaml_params["path"],
            "images_dir": yaml_params["val"],
            "labels_dir": yaml_params["val"],
            "classes": yaml_params["names"],
            "input_dim": (args["size"], args["size"]),
        },
        dataloader_params={"batch_size": args["batch"], "num_workers": args["worker"]},
    )

    # Evaluating on Val Dataset
    eval_model = trainer.test(
        model=best_model,
        test_loader=val_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(yaml_params["names"]),
            normalize_targets=True,
            calc_best_score_thresholds=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        ),
    )
    print("\033[1m [INFO] Validating Model:\033[0m")
    for i in eval_model:
        print(f"{i}: {float(eval_model[i])}")


def test(args, name, save_dir, best_model, yaml_params):
    init_trainer()

    # Training on GPU or CPU
    if args["cpu"]:
        print("[INFO] Training on \033[1mCPU\033[0m")
        setup_device(device="cpu")
    else:
        print(f"[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m")
        setup_device(device="cuda")
    trainer = Trainer(experiment_name=name, ckpt_root_dir=save_dir)

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": yaml_params["path"],
            "images_dir": yaml_params["test"],
            "labels_dir": yaml_params["test"],
            "classes": yaml_params["names"],
            "input_dim": (args["size"], args["size"]),
        },
        dataloader_params={"batch_size": args["batch"], "num_workers": args["worker"]},
    )

    # Evaluating on Test Dataset
    test_result = trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.3,
            top_k_predictions=300,
            num_cls=len(yaml_params["names"]),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        ),
    )
    print("\033[1m [INFO] Test Results:\033[0m")
    for i in test_result:
        print(f"{i}: {float(test_result[i])}")


def main():
    save_dir = "/home/model-output/yolonas/runs"
    args = get_args()
    s_time = time.time()

    if args["name"] is None:
        name = "train"
    else:
        name = args["name"]

    if not os.path.exists(os.path.join(save_dir, name)):
        Path(os.path.join(save_dir, name)).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Checkpoints saved in \033[1m{os.path.join(save_dir, name)}\033[0m")

    yaml_params = yaml.safe_load(open(args["data"], "r"))

    # Training
    checkpoint_root_dir = train(args, save_dir, name, yaml_params)

    # Load best model
    best_model = models.get(
        args["model"],
        num_classes=len(yaml_params["names"]),
        checkpoint_path=os.path.join(checkpoint_root_dir, "ckpt_best.pth"),
    )

    # Evaluating on Val Dataset
    validate(args, name, save_dir, best_model, yaml_params)

    # Evaluating on Test Dataset
    if "test" in (yaml_params.keys()):
        test(args, name, save_dir, best_model, yaml_params)

    print(
        f"[INFO] Training Completed in \033[1m{(time.time()-s_time)/3600} Hours\033[0m"
    )


if __name__ == "__main__":
    os.umask(0)
    main()
