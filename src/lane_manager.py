# -*- coding: utf-8 -*-
"""
车道管理模块：ROI判定、面积计算
"""
import cv2
import numpy as np

from config import LANE_ROIS, VIDEO_PATH


def point_in_roi(px, py, roi):
    """判断点是否在ROI多边形内"""
    roi_contour = roi.reshape((-1, 1, 2)).astype(np.int32)
    return cv2.pointPolygonTest(roi_contour, (px, py), False) >= 0


def get_lane_by_roi(cx, cy):
    """根据底部中心点获取所在车道名称"""
    for lane_name, roi in LANE_ROIS.items():
        if point_in_roi(cx, cy, roi):
            return lane_name
    return None


def calculate_pixel_area(roi):
    """计算ROI区域的像素面积"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    else:
        h, w = 1080, 1920
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi.astype(np.int32)], 255)
    return np.count_nonzero(mask)


# 预计算各车道面积
LANE_AREAS = {name: calculate_pixel_area(roi) for name, roi in LANE_ROIS.items()}