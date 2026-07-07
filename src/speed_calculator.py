# -*- coding: utf-8 -*-
"""
速度计算模块：瞬时速度、过线速度
"""
from calibration import calc_real_distance


def calc_instant_speed(track_history, track_id, fps, smooth_window=3):
    """
    计算车辆瞬时速度（基于最近N帧位移）
    """
    if track_id not in track_history or len(track_history[track_id]) < 2:
        return 0.0
    n = min(smooth_window, len(track_history[track_id]))
    points = track_history[track_id][-n:]
    first_cx, first_cy, first_t = points[0]
    last_cx, last_cy, last_t = points[-1]
    dist_m = calc_real_distance(first_cx, first_cy, last_cx, last_cy)
    dt = last_t - first_t
    if dt > 0:
        return (dist_m / dt) * 3.6
    return 0.0