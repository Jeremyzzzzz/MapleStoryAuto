"""Build YOLO dataset from a single labeled scene + heavy augmentation.

Output: training_data/scene1_yolo/{images/{train,val},labels/{train,val},data.yaml}
Augmentations: crop around each monster with random scale/rotation/HSV jitter,
flip, plus background-only crops as negatives? YOLO has no negatives, but we
make sure every augmentation still has the right label inside the crop.
"""

import random
from pathlib import Path
import cv2
import numpy as np
import yaml

REPO = Path(r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp")
SCENE = REPO / "monster" / "测试集" / "场景1.png"
OUT = REPO / "training_data" / "scene1_yolo"

# 类别: 0=slime, 1=red_snail, 2=green_mushroom, 3=stump, 4=flower_mushroom
# 注: 场景里红蜗牛/绿蘑菇/花蘑菇可能都未出现, 但保留类
CLASS_NAMES = ["slime", "red_snail", "green_mushroom", "stump", "flower_mushroom"]

# 我看到的怪 (中心, 半宽, 半高, 类)
# y=53 是上层水灵 (green water slime), y=305 是下层地面怪物
BOXES = [
    # (cx, cy, hw, hh, cls_name)
    (620, 52,  15, 12, "slime"),         # 上层左水灵
    (685, 53,  15, 12, "slime"),         # 上层右水灵
    (220, 305, 14, 13, "red_snail"),     # 玩家右侧红蜗牛
    (310, 298, 16, 16, "stump"),         # 戴橙色头巾的stump
    (365, 300, 17, 17, "stump"),         # 树妖 (黑眼睛)
    (425, 302, 17, 17, "stump"),         # 另一个树妖
    (480, 300, 22, 22, "green_mushroom"),# 大白蘑菇/绿蘑菇
]

N_TRAIN = 200
N_VAL = 40
SEED = 17


def render(img, boxes, rng):
    """Random augmentation of the scene + bbox list.
    boxes: [(cx, cy, w, h, cls_id)] in original image coords.
    Returns (out_img, out_boxes) where out_boxes use 0..1 normalized.
    """
    h0, w0 = img.shape[:2]
    # random scale (0.7-1.4)
    scale = rng.uniform(0.7, 1.4)
    # random rotation
    rot = rng.uniform(-12, 12)
    # random HSV jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.85, 1.15), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # scale around center
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), rot, scale)
    aug = cv2.warpAffine(aug, M, (w0, h0), borderValue=(0, 0, 0))
    # transform boxes
    out = []
    for cx, cy, bw, bh, cls_id in boxes:
        pt = np.array([cx, cy, 1.0])
        ncx, ncy = (M @ pt)[:2]
        nbw, nbh = bw * scale, bh * scale
        # ensure still in image
        if ncx - nbw / 2 < 5 or ncy - nbh / 2 < 5 or \
           ncx + nbw / 2 > w0 - 5 or ncy + nbh / 2 > h0 - 5:
            continue
        out.append((ncx, ncy, nbw, nbh, cls_id))
    if not out:
        return None, None
    # optional flip
    if rng.random() < 0.5:
        aug = cv2.flip(aug, 1)
        out = [(w0 - x, y, w, h, c) for (x, y, w, h, c) in out]
    # normalized
    norm = [(x / w0, y / h0, bw / w0, bh / h0, c) for (x, y, bw, bh, c) in out]
    return aug, norm


def save_split(imgs_path, lbs_path, img, boxes_norm, idx):
    fname = f"{idx:04d}.jpg"
    ip = imgs_path / fname
    ok, buf = cv2.imencode(".jpg", img)
    ip.write_bytes(buf.tobytes())
    label = "".join(f"{c} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}\n"
                    for (cx, cy, bw, bh, c) in boxes_norm)
    (lbs_path / fname.replace(".jpg", ".txt")).write_text(label, encoding="ascii")


def main():
    rng = random.Random(SEED)
    img = cv2.imdecode(np.fromfile(str(SCENE), dtype=np.uint8), cv2.IMREAD_COLOR)
    h0, w0 = img.shape[:2]
    print(f"场景图 {w0}x{h0}")

    boxes_with_id = [(cx, cy, bw, bh, CLASS_NAMES.index(cls)) for (cx, cy, bw, bh, cls) in BOXES]

    for split, n in (("train", N_TRAIN), ("val", N_VAL)):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            aug, norm = render(img, boxes_with_id, rng)
            if aug is None:
                continue
            save_split(OUT / "images" / split, OUT / "labels" / split, aug, norm, i)
    print(f"done: train={N_TRAIN} val={N_VAL}")

    (OUT / "data.yaml").write_text(
        yaml.safe_dump({
            "path": str(OUT),
            "train": "images/train",
            "val": "images/val",
            "names": {i: c for i, c in enumerate(CLASS_NAMES)},
        }, allow_unicode=True), encoding="utf-8")
    print("data.yaml ->", OUT / "data.yaml")


if __name__ == "__main__":
    main()