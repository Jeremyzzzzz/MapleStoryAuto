# -*- coding: utf-8 -*-
"""训练人物(麻超圆)单类 YOLO 检测模型。

数据: training_data/player_capture_v2 (134 帧, 已验证标注有效) + 增强。
输出: training_runs/player_yolo_v1/weights/best.pt

用法(项目根): python tools/train_player_yolo.py --epochs 80 --imgsz 640
"""
import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Train player YOLO detector")
    p.add_argument("--source", default=str(REPO_ROOT / "training_data/player_capture_v2"))
    p.add_argument("--data-dir", default=str(REPO_ROOT / "training_data/player_yolo_v1"))
    p.add_argument("--out", default=str(REPO_ROOT / "training_runs/player_yolo_v1"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="yolo11n.pt",
                   help="基础模型(项目根 yolo11n.pt 或 ultralytics 自动下载)")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=12)
    return p.parse_args()


def build_dataset(source, data_dir, val_split):
    """复制源帧到 data_dir, 划分 train/val, 写 data.yaml(含增强注释)。"""
    src = Path(source)
    out = Path(data_dir)
    img_dst = out / "images"
    lbl_dst = out / "labels"
    (img_dst / "train").mkdir(parents=True, exist_ok=True)
    (img_dst / "val").mkdir(parents=True, exist_ok=True)
    (lbl_dst / "train").mkdir(parents=True, exist_ok=True)
    (lbl_dst / "val").mkdir(parents=True, exist_ok=True)

    items = []
    for lp in sorted((src / "labels").glob("*.txt")):
        ip = src / "images" / (lp.stem + ".jpg")
        if ip.exists():
            items.append((ip, lp))
    if len(items) < 10:
        raise RuntimeError(f"源帧太少: {len(items)}")
    random.seed(42)
    random.shuffle(items)
    n_val = max(2, int(len(items) * val_split))
    val_items = items[:n_val]
    train_items = items[n_val:]

    def copy_pair(items, split):
        for ip, lp in items:
            shutil.copy(ip, img_dst / split / ip.name)
            shutil.copy(lp, lbl_dst / split / lp.name)

    copy_pair(train_items, "train")
    copy_pair(val_items, "val")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f'path: "{out.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: player\n",
        encoding="utf-8",
    )
    return len(train_items), len(val_items)


def main():
    args = parse_args()
    n_train, n_val = build_dataset(args.source, args.data_dir, args.val_split)
    print(f"数据集: train={n_train} val={n_val} -> {args.data_dir}", flush=True)

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data_dir) / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=True,
        patience=args.patience,
        # 增强(单类人物, 角色固定): 平移/缩放/亮度轻微/翻转? 不翻转(角色朝向有意义)
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
        degrees=0.0, translate=0.10, scale=0.15, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.0,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
        close_mosaic=10,
        # 单类检测器目标: 宽容
        conf=0.001, iou=0.6,
        rect=False,
        verbose=True,
    )
    print(f"训练完成: {Path(args.out) / 'weights' / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
