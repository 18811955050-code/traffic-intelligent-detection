# -*- coding: utf-8 -*-
"""
排队长度计算模块
"""
import math
import cv2
import numpy as np

from config import LANE_ROIS
from calibration import calc_y_distance_at_x, pixel_to_real


# 全局平滑历史
_queue_history = {}
_QUEUE_SMOOTH_WINDOW = 5


def get_speed_threshold(cy):
    """根据Y坐标动态调整速度阈值（克服透视畸变）"""
    if cy > 1300:
        return 1.0
    elif cy > 1150:
        return 0.6
    elif cy > 1000:
        return 0.3
    elif cy > 850:
        return 0.12
    else:
        return 0.06


def smooth_queue(lane, value):
    """中值滤波平滑排队长度"""
    global _queue_history
    if lane not in _queue_history:
        _queue_history[lane] = []
    _queue_history[lane].append(value)
    if len(_queue_history[lane]) > _QUEUE_SMOOTH_WINDOW:
        _queue_history[lane].pop(0)
    return np.median(_queue_history[lane])


def update_queue_length(current_boxes, track_history, STOP_LINE_Y):
    """
    更新各车道排队长度
    """
    queue_length_m = {lane: 0.0 for lane in LANE_ROIS}
    stopped_vehicles = {lane: [] for lane in LANE_ROIS}
    stopped_vehicles_cx = {lane: [] for lane in LANE_ROIS}

    for (tid, cls_id, x1, y1, x2, y2, cx, cy_bottom) in current_boxes:
        # 确定车道
        lane_name = None
        for name, roi in LANE_ROIS.items():
            roi_contour = roi.reshape((-1, 1, 2)).astype(np.int32)
            if cv2.pointPolygonTest(roi_contour, (cx, cy_bottom), False) >= 0:
                lane_name = name
                break
        if lane_name is None:
            continue
        if tid not in track_history or len(track_history[tid]) < 8:
            continue

        # 计算最近5帧平均速度
        speeds = []
        start = max(0, len(track_history[tid]) - 5)
        for i in range(start + 1, len(track_history[tid])):
            prev_cx, prev_cy, prev_t = track_history[tid][i - 1]
            curr_cx, curr_cy, curr_t = track_history[tid][i]
            dist = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
            dt = curr_t - prev_t
            if dt > 0:
                speeds.append(dist / dt)
        avg_speed = np.mean(speeds) if speeds else 999
        threshold = get_speed_threshold(cy_bottom)

        if avg_speed < threshold:
            stopped_vehicles[lane_name].append(cy_bottom)
            stopped_vehicles_cx[lane_name].append(cx)

    # 计算各车道排队长度
    for lane_name in LANE_ROIS:
        stop_y = STOP_LINE_Y[lane_name]
        if stopped_vehicles[lane_name]:
            idx = np.argmin(stopped_vehicles[lane_name])
            min_y = stopped_vehicles[lane_name][idx]
            min_cx = stopped_vehicles_cx[lane_name][idx]
            raw_queue = calc_y_distance_at_x(min_cx, stop_y, min_y)
            smoothed = smooth_queue(lane_name, raw_queue)
            queue_length_m[lane_name] = round(smoothed, 2)
        else:
            queue_length_m[lane_name] = 0.0
            if lane_name in _queue_history:
                _queue_history[lane_name] = []

    return queue_length_m