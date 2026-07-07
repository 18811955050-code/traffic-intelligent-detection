# -*- coding: utf-8 -*-
"""
配置文件：所有路径、车道坐标、检测参数集中管理
"""
import os
import numpy as np

# ===================== 路径配置 =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "north.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_EXCEL_PATH = os.path.join(BASE_DIR, "outputs", "traffic_10_classes_output.xlsx")
PERSPECTIVE_MATRIX_PATH = os.path.join(BASE_DIR, "perspective_matrix.npy")

# ===================== 车道ROI坐标 =====================
LANE_ROIS = {
    "ev-line": np.array([[1338, 528], [1504, 524], [1440, 770], [1176, 788]]),
    "str1": np.array([[1504, 524], [1718, 506], [1794, 748], [1444, 768]]),
    "str2": np.array([[1718, 504], [1930, 494], [2140, 732], [1794, 748]]),
    "left": np.array([[1932, 498], [2140, 488], [2488, 724], [2142, 736]])
}

# ===================== 停止线和检测线 =====================
STOP_LINE = {
    "ev-line": ((1338, 528), (1504, 524)),
    "str1": ((1504, 524), (1718, 506)),
    "str2": ((1718, 504), (1930, 494)),
    "left": ((1932, 498), (2140, 488))
}

DETECT_LINE = {
    "ev-line": ((1176, 788), (1440, 770)),
    "str1": ((1444, 768), (1794, 748)),
    "str2": ((1794, 748), (2140, 732)),
    "left": ((2142, 736), (2488, 724))
}

# ===================== 检测参数 =====================
CONFIDENCE_THRESHOLD = 0.25
TARGET_CLASSES = [0, 1, 2, 3]
CLASS_NAMES = {0: "car", 1: "bus", 2: "truck", 3: "van"}
CLASS_LIST = ["car", "bus", "truck", "van"]

# ===================== 统计时间粒度 =====================
OCCUPANCY_INTERVAL = 5
TRAFFIC_FLOW_INTERVAL = 60

# ===================== 显示参数 =====================
DISPLAY_SCALE = 0.3
WAIT_MS = 1

# ===================== 颜色识别参数 =====================
COLOR_RANGES = {
    '红色': [(0, 70, 70), (10, 255, 255), (170, 70, 70), (180, 255, 255)],
    '黄色': [(15, 80, 80), (35, 255, 255)],
    '蓝色': [(85, 80, 80), (135, 255, 255)],
    '白色': [(0, 0, 150), (180, 30, 255)],
    '黑色': [(0, 0, 0), (180, 50, 60)],
    '灰色': [(0, 0, 60), (180, 30, 180)],
    '银色': [(0, 0, 120), (180, 20, 200)]
}


def get_line_y(line):
    """获取线段中心点的Y坐标"""
    return int((line[0][1] + line[1][1]) / 2)


STOP_LINE_Y = {name: get_line_y(line) for name, line in STOP_LINE.items()}
DETECT_LINE_Y = {name: get_line_y(line) for name, line in DETECT_LINE.items()}