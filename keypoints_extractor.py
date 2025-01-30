import cv2
import numpy as np
from pathlib import Path

kp_data = Path("kp_dataset")


def create_fence_mask_from_json(data):
    # Inicjalizacja list na punkty kluczowe
    kp_h = []  # Punkty dla Fence_KP_Horizontal
    kp_v = []  # Punkty dla Fence_KP_Vertical

    # Przetwarzanie anotacji
    for annotation in data["annotations"]:
        keypoints = annotation["keypoints"]
        category_id = annotation["category_id"]

        # Konwersja keypointów na listy krotek (x, y)

        if category_id == 1:  # Fence_KP_Horizontal
            kp_h.extend(keypoints)
        elif category_id == 2:  # Fence_KP_Vertical
            kp_v.extend(keypoints)

    kp_h = [(x, y) for x, y, _ in zip(kp_h[::3], kp_h[1::3], kp_h[2::3])]
    kp_v = [(x, y) for x, y, _ in zip(kp_v[::3], kp_v[1::3], kp_v[2::3])]

    kp_top = kp_h[len(kp_h) // 2 :]
    kp_bottom = kp_h[: len(kp_h) // 2][::-1]

    kp_left = kp_v[len(kp_v) // 2 :]
    kp_right = kp_v[: len(kp_v) // 2]

    height = 2160
    width = 3840

    line_thickness = 4

    array = np.zeros((height, width, 1), dtype=np.uint8)

    for (x_top, y_top), (x_bottom, y_bottom) in zip(kp_top, kp_bottom):
        cv2.line(
            array,
            (int(x_top), int(y_top)),
            (int(x_bottom), int(y_bottom)),
            255,
            line_thickness,
        )

    for (x_top, y_top), (x_bottom, y_bottom) in zip(kp_left, kp_right):
        cv2.line(
            array,
            (int(x_top), int(y_top)),
            (int(x_bottom), int(y_bottom)),
            255,
            line_thickness,
        )

    return array
