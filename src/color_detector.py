# -*- coding: utf-8 -*-
"""
车身颜色识别模块
"""
import cv2
import numpy as np

from config import COLOR_RANGES


def get_car_color(roi):
    """从车辆ROI区域识别车身颜色"""
    try:
        if roi.size == 0:
            return "未知"
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        count = {}

        # 红色需要双区间（跨0°/180°）
        l1, u1, l2, u2 = COLOR_RANGES['红色']
        count['红色'] = cv2.countNonZero(
            cv2.inRange(hsv, np.array(l1), np.array(u1)) +
            cv2.inRange(hsv, np.array(l2), np.array(u2))
        )

        # 其他颜色单区间
        for name, ran in COLOR_RANGES.items():
            if name == '红色':
                continue
            count[name] = cv2.countNonZero(
                cv2.inRange(hsv, np.array(ran[0]), np.array(ran[1]))
            )

        return max(count, key=count.get)
    except:
        return "未知"