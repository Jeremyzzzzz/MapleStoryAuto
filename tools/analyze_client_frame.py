import argparse
import json

import cv2
import numpy as np

from src.utils.common import get_minimap_loc_size


def components(mask, min_width=1, max_height=None, min_area=2, y_offset=0):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    result = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if width < min_width or area < min_area:
            continue
        if max_height is not None and height > max_height:
            continue
        result.append(
            {
                "box": [x, y + y_offset, width, height],
                "area": area,
                "center": [
                    round(float(centroids[index][0]), 1),
                    round(float(centroids[index][1] + y_offset), 1),
                ],
            }
        )
    return sorted(result, key=lambda item: item["area"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    image_data = np.fromfile(args.image, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to load image: {args.image}")

    result = {
        "frame_size": [int(image.shape[1]), int(image.shape[0])],
        "minimap_box": None,
        "minimap_yellow_components": [],
        "status_color_components": {},
    }

    minimap_box = get_minimap_loc_size(image)
    if minimap_box is not None:
        x, y, width, height = map(int, minimap_box)
        result["minimap_box"] = [x, y, width, height]
        minimap = image[y : y + height, x : x + width]
        yellow_mask = cv2.inRange(
            minimap,
            np.array([0, 180, 180]),
            np.array([160, 255, 255]),
        )
        result["minimap_yellow_components"] = components(yellow_mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi_y = int(image.shape[0] * 0.85)
    color_masks = {
        "red": cv2.inRange(hsv, (0, 120, 120), (10, 255, 255)),
        "blue": cv2.inRange(hsv, (95, 100, 100), (135, 255, 255)),
        "yellow": cv2.inRange(hsv, (18, 100, 100), (40, 255, 255)),
    }
    for name, mask in color_masks.items():
        result["status_color_components"][name] = components(
            mask[roi_y:, :],
            min_width=20,
            max_height=30,
            min_area=20,
            y_offset=roi_y,
        )[:10]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
