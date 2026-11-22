"""Garment outline for whole-garment photos, used to drop damage boxes that sit mostly on the floor.

The YOLOv8 damage detector was trained largely on close-up fabric datasets. On a phone photo of a whole garment
lying on a tiled floor it sometimes boxes the floor beside a sleeve or the hem as a "hole". GrabCut, started from
a rectangle just inside the photo border, separates the garment from that background; a box is kept only when
most of it lies inside the garment outline (gaps such as holes inside the outline count as garment).
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_SIZE = 480          # long side used for segmentation, in pixels
MIN_GARMENT_SHARE = 0.15  # below this the outline is unreliable (garment too small or not found)
MAX_GARMENT_SHARE = 0.92  # above this the garment fills the photo, so every box is on the garment anyway


def silhouette(image):
    """Filled garment outline at the image's size, or None when the photo cannot be segmented reliably."""
    if image is None or image.ndim != 3:
        return None
    height, width = image.shape[:2]
    scale = WORK_SIZE / max(height, width)
    small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    if min(sh, sw) < 32:
        return None
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(sw * 0.03), int(sh * 0.03), int(sw * 0.94), int(sh * 0.94))
    try:
        cv2.grabCut(small, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    garment = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(garment)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    garment = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(garment)
    cv2.drawContours(filled, contours, -1, 255, -1)
    share = float((filled > 0).mean())
    if not MIN_GARMENT_SHARE <= share <= MAX_GARMENT_SHARE:
        return None
    return cv2.resize(filled, (width, height), interpolation=cv2.INTER_NEAREST)


def share_on_garment(outline, bbox):
    """Fraction of a normalised [x, y, w, h] box that lies inside the garment outline."""
    height, width = outline.shape[:2]
    x, y, w, h = bbox
    x1, y1 = int(round(x * width)), int(round(y * height))
    x2, y2 = int(round((x + w) * width)), int(round((y + h) * height))
    region = outline[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    return float((region > 0).mean()) if region.size else 0.0
