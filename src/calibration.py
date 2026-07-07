# -*- coding: utf-8 -*-
"""
透视变换模块：像素坐标 ↔ 实际坐标（米）
"""
import cv2
import numpy as np
import math

from config import PERSPECTIVE_MATRIX_PATH

# 加载透视变换矩阵
try:
    PERSPECTIVE_MATRIX = np.load(PERSPECTIVE_MATRIX_PATH)
    USE_PERSPECTIVE = True
    print("✅ 已加载透视变换矩阵")
except:
    PERSPECTIVE_MATRIX = None
    USE_PERSPECTIVE = False
    print("❌ 未找到透视变换矩阵，使用固定系数降级方案")
    PIXEL_TO_METER = 0.0797


def pixel_to_real(px, py):
    """将像素坐标转换为实际坐标（米）"""
    if USE_PERSPECTIVE and PERSPECTIVE_MATRIX is not None:
        src = np.array([[[px, py]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, PERSPECTIVE_MATRIX)
        return dst[0][0][0], dst[0][0][1]
    else:
        return px * PIXEL_TO_METER, py * PIXEL_TO_METER


def calc_real_distance(px1, py1, px2, py2):
    """计算两个像素点之间的实际距离（米）"""
    x1, y1 = pixel_to_real(px1, py1)
    x2, y2 = pixel_to_real(px2, py2)
    return math.hypot(x2 - x1, y2 - y1)


def calc_y_distance_at_x(cx, py1, py2):
    """计算同一X坐标下两个Y之间的实际距离"""
    _, y1 = pixel_to_real(cx, py1)
    _, y2 = pixel_to_real(cx, py2)
    return abs(y2 - y1)