"""Build a YOLO dataset from the 5 class official art + real scene screenshots.

Strategy:
- Extract the monster sprite from each official art image (background removal
  via edge/color prior).
- Paste the sprite onto real game scene backgrounds (monster/测试集/*.png) at
  game-realistic sizes (30-90px), with slight rotation/scaling/flip and HSV
  jitter -> hundreds of synthetic training images.
- Also paste on plain synthetic backgrounds (dark/light) to teach shape.
- Background-only crops are added as negatives? (YOLO has no negatives in
  training; we just do not add them.)

Output: training_data/maple_art_v1/{images/{train,val},labels/{train,val},data.yaml}
Classes: red_snail, flower_mushroom, green_mushroom, green_slime, stump
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp")
ART_DIR = REPO / "monster"
SCENE_DIR = REPO / "monster" / "测试集"
OUT = REPO / "training_data" / "maple_art_v1"

CLASSES = ["red_snail", "flower_mushroom", "green_mushroom", "green_slime", "stump"]
ART_FILES = {
    "red_snail": "红蜗牛",
    "flower_mushroom": "花蘑菇",
    "green_mushroom": "绿蘑菇",
    "green_slime": "绿水灵",
    "stump": "树妖",
}

SEED = 17
N_PER_CLASS = 220          # synthetic images per class
IMG_W, IMG_H = 640, 480    # training image size
BG_CROP = 0.5              # fraction of scenes used for backgrounds


def imread_cn(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def remove_bg(img):
    """Return (sprite_rgb, alpha) with background estimated from image border.

    For tight sprites (e.g. black_axe_stump green-screen), this works well.
    For full-art 180x180 official illustrations, the border may share colors
    with the sprite and the result becomes too small. In that case we fall
    back to a central ellipse mask, then bump the alpha to the image edges
    when almost no pixels were marked (the whole picture IS the sprite).
    """
    edge = np.concatenate([
        img[0:3].reshape(-1, 3),
        img[-3:].reshape(-1, 3),
        img[:, 0:3].reshape(-1, 3),
        img[:, -3:].reshape(-1, 3),
    ])
    bg_color = np.median(edge, axis=0).astype(np.int16)
    diff = np.abs(img.astype(np.int16) - bg_color).sum(axis=2)
    alpha = (diff > 60).astype(np.uint8) * 255
    # green-screen override
    gs = np.all(img == [0, 255, 0], axis=2)
    alpha[gs] = 0
    # morphology cleanup
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # keep the largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if n > 1:
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        alpha = (labels == best).astype(np.uint8) * 255
    fg_frac = (alpha > 0).mean()
    # fallback: if too little was extracted, the picture is its own sprite
    if fg_frac < 0.10:
        alpha = np.full(img.shape[:2], 255, np.uint8)
    # also strip a 4px border (often aliasing residue) when the sprite is
    # near full image
    if fg_frac > 0.85:
        alpha[:4] = alpha[-4:] = alpha[:, :4] = alpha[:, -4:] = 0
    return img, alpha


def load_sprites():
    sprites = {}
    for cls, folder in ART_FILES.items():
        d = ART_DIR / folder
        if not d.exists():
            print(f"[build] missing {d}")
            continue
        items = []
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = imread_cn(p)
            if img is None:
                continue
            rgb, alpha = remove_bg(img)
            fg = alpha > 0
            if int(fg.sum()) < 60:
                continue
            # tight crop
            ys, xs = np.where(fg)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            spr = rgb[y0:y1 + 1, x0:x1 + 1]
            alp = alpha[y0:y1 + 1, x0:x1 + 1]
            items.append((spr, alp))
        if items:
            sprites[cls] = items
            print(f"[build] {cls}: {len(items)} sprite(s)")
        else:
            print(f"[build] WARNING {cls}: no sprite extracted")
    return sprites


def paste_sprite(bg, spr, alp, cls_rng, rng):
    """Paste sprite at a random position/size. Returns (image, box) or None."""
    sh, sw = spr.shape[:2]
    # random game size: 30-95 px tall
    target_h = rng.randint(30, 95)
    scale = target_h / max(sh, 1)
    nw, nh = max(8, int(sw * scale)), max(8, int(sh * scale))
    sp = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA)
    ap = cv2.resize(alp, (nw, nh), interpolation=cv2.INTER_AREA)
    _, ap = cv2.threshold(ap, 128, 255, cv2.THRESH_BINARY)
    # random rotation (slight)
    ang = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), ang, 1.0)
    sp = cv2.warpAffine(sp, M, (nw, nh), borderValue=(0, 255, 0))
    ap = cv2.warpAffine(ap, M, (nw, nh), borderValue=0)
    _, ap = cv2.threshold(ap, 128, 255, cv2.THRESH_BINARY)
    # optional flip
    if rng.random() < 0.5:
        sp = cv2.flip(sp, 1)
        ap = cv2.flip(ap, 1)
    # position: x anywhere, y on ground-ish lower half
    max_x = max(0, bg.shape[1] - nw)
    max_y = max(0, bg.shape[0] - nh)
    if max_x < 2 or max_y < 2:
        return None, None
    x = rng.randint(0, max_x)
    y = rng.randint(int(bg.shape[0] * 0.3), max_y)
    roi = bg[y:y + nh, x:x + nw]
    m = (ap > 0)
    roi[m] = sp[m]
    box = (x, y, nw, nh)
    return bg, box


def save_yolo(path, img, box, cls_id):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
    bw_n, bh_n = bw / w, bh / h
    label = f"{cls_id} {cx:.5f} {cy:.5f} {bw_n:.5f} {bh_n:.5f}\n"
    path.with_suffix(".txt").write_text(label, encoding="ascii")


def main():
    rng = random.Random(SEED)
    sprites = load_sprites()
    if not sprites:
        print("No sprites extracted; abort.")
        return

    # backgrounds: real scene crops + synthetic
    backgrounds = []
    scene_files = sorted(SCENE_DIR.glob("*.png")) if SCENE_DIR.exists() else []
    for sf in scene_files:
        img = imread_cn(sf)
        if img is None:
            continue
        h, w = img.shape[:2]
        # random crops with overlap -> many distinct backgrounds
        for _ in range(8):
            cw = rng.randint(320, w)
            ch = rng.randint(320, h)
            if cw < 320 or ch < 320:
                continue
            x = rng.randint(0, max(0, w - cw))
            y = rng.randint(0, max(0, h - ch))
            crop = cv2.resize(img[y:y + ch, x:x + cw], (IMG_W, IMG_H))
            backgrounds.append(crop)
    # synthetic backgrounds
    for _ in range(40):
        bg = np.full((IMG_H, IMG_W, 3), rng.randint(30, 200), np.uint8)
        # gradient
        for yy in range(IMG_H):
            bg[yy] = np.clip(bg[yy] + (yy - IMG_H / 2) * 0.2, 0, 255).astype(np.uint8)
        backgrounds.append(bg)
    print(f"[build] {len(backgrounds)} backgrounds ({len(scene_files)} scenes)")

    # split backgrounds: train / val
    rng.shuffle(backgrounds)
    n_val = max(1, int(len(backgrounds) * 0.2))
    val_bg = backgrounds[:n_val]
    train_bg = backgrounds[n_val:]

    for split, bg_list in (("train", train_bg), ("val", val_bg)):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    idx = 0
    for cls in CLASSES:
        if cls not in sprites:
            continue
        cls_id = CLASSES.index(cls)
        spr_pool = sprites[cls]
        for i in range(N_PER_CLASS):
            # pick a background (mostly train, some val)
            split = "train" if rng.random() < 0.85 else "val"
            bg_list = train_bg if split == "train" else val_bg
            bg = rng.choice(bg_list).copy()
            spr, alp = rng.choice(spr_pool)
            # HSV jitter on sprite
            hsv = cv2.cvtColor(spr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.85, 1.15), 0, 255)
            spr_j = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            out_img, box = paste_sprite(bg, spr_j, alp, cls, rng)
            if out_img is None:
                continue
            fname = f"{cls}_{i:03d}"
            img_path = OUT / "images" / split / f"{fname}.jpg"
            ok, buf = cv2.imencode(".jpg", out_img)
            img_path.write_bytes(buf.tobytes())
            save_yolo(img_path, out_img, box, cls_id)
            idx += 1
    print(f"[build] wrote {idx} labeled images")

    data_yaml = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(CLASSES)},
    }
    (OUT / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")
    print(f"[build] data.yaml -> {OUT / 'data.yaml'}")
    print("[build] done")


if __name__ == "__main__":
    main()