# -*- coding: utf-8 -*-
"""
数据记录与导出模块
"""
import pandas as pd
import cv2
from collections import defaultdict

from config import LANE_ROIS, CLASS_LIST


class MetricsRecorder:
    """10类数据记录器"""

    def __init__(self):
        # 流量
        self.lane_flow_count = {lane: 0 for lane in LANE_ROIS}
        self.lane_flow_minute = defaultdict(lambda: defaultdict(int))

        # 占有率
        self.lane_time_occ = defaultdict(float)
        self.lane_time_occ_history = defaultdict(list)
        self.lane_space_occ_sum = defaultdict(float)
        self.lane_space_occ_frame_count = defaultdict(int)
        self.lane_space_occ_history = defaultdict(list)

        # 轨迹
        self.vehicle_trajectory = defaultdict(list)

        # 车辆类型
        self.vehicle_type_count = defaultdict(lambda: defaultdict(int))

        # 颜色
        self.vehicle_color = {}

        # 排队长度
        self.queue_records = []

        # 速度
        self.speed_records = []
        self.instant_speed_records = []
        self.cross_instant_speed_records = []

        # 车头时距/间距
        self.headway_records = []
        self.last_pass_time = {lane: None for lane in LANE_ROIS}
        self.last_pass_point = {lane: None for lane in LANE_ROIS}

        # 到达/离去
        self.arrival_time = {}
        self.departure_time = {}

        # 计数去重
        self.counted_ids = set()
        self.vehicle_states = {}
        self.track_history = defaultdict(list)

    def export_to_excel(self, output_path):
        """导出所有数据到Excel"""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: 流量
            flow_data = []
            for minute_idx in sorted(self.lane_flow_minute.keys()):
                row = {"分钟序号": minute_idx + 1}
                for lane in LANE_ROIS:
                    row[f"{lane}流量"] = self.lane_flow_minute[minute_idx].get(lane, 0)
                flow_data.append(row)
            if flow_data:
                pd.DataFrame(flow_data).to_excel(writer, sheet_name="1_流量", index=False)

            # Sheet 2: 时间占有率
            time_occ_data = []
            max_len = max(len(v) for v in self.lane_time_occ_history.values()) if self.lane_time_occ_history else 0
            for i in range(max_len):
                row = {"序号": i + 1}
                for lane in LANE_ROIS:
                    occ_list = self.lane_time_occ_history[lane]
                    row[f"{lane}_时间占有率(%)"] = round(occ_list[i], 2) if i < len(occ_list) else 0
                time_occ_data.append(row)
            if time_occ_data:
                pd.DataFrame(time_occ_data).to_excel(writer, sheet_name="2_时间占有率", index=False)

            # Sheet 3: 空间占有率
            space_occ_data = []
            max_len2 = max(len(v) for v in self.lane_space_occ_history.values()) if self.lane_space_occ_history else 0
            for i in range(max_len2):
                row = {"序号": i + 1}
                for lane in LANE_ROIS:
                    occ_list = self.lane_space_occ_history[lane]
                    row[f"{lane}_空间占有率(%)"] = round(occ_list[i], 2) if i < len(occ_list) else 0
                space_occ_data.append(row)
            if space_occ_data:
                pd.DataFrame(space_occ_data).to_excel(writer, sheet_name="3_空间占有率", index=False)

            # Sheet 4: 车头间距/时距
            if self.headway_records:
                pd.DataFrame(self.headway_records).to_excel(writer, sheet_name="4_车头间距_时距", index=False)

            # Sheet 5: 排队长度
            if self.queue_records:
                pd.DataFrame(self.queue_records).to_excel(writer, sheet_name="5_排队长度", index=False)

            # Sheet 6: 区间平均速度
            if self.speed_records:
                pd.DataFrame(self.speed_records).to_excel(writer, sheet_name="6_平均速度", index=False)

            # Sheet 7: 车辆类型
            vehicle_type_data = []
            for lane in LANE_ROIS:
                row = {"车道": lane}
                for cls_name in CLASS_LIST:
                    row[cls_name] = 0
                if lane in self.vehicle_type_count:
                    for cls_name, count in self.vehicle_type_count[lane].items():
                        row[cls_name] = count
                vehicle_type_data.append(row)
            if vehicle_type_data:
                pd.DataFrame(vehicle_type_data).to_excel(writer, sheet_name="7_车辆类型", index=False)

            # Sheet 8: 车身颜色
            color_data = [{"车辆ID": tid, "颜色": color} for tid, color in self.vehicle_color.items()]
            if color_data:
                pd.DataFrame(color_data).to_excel(writer, sheet_name="8_车身颜色", index=False)

            # Sheet 9: 到达/离去时间
            arrival_depart_data = []
            all_vehicles = set(self.arrival_time.keys()) | set(self.departure_time.keys())
            for tid in all_vehicles:
                arrival = self.arrival_time.get(tid, {})
                depart = self.departure_time.get(tid, {})
                arrival_depart_data.append({
                    "车辆ID": tid,
                    "车道": arrival.get("lane") or depart.get("lane"),
                    "到达时间(s)": arrival.get("time", "N/A"),
                    "离去时间(s)": depart.get("time", "N/A")
                })
            if arrival_depart_data:
                pd.DataFrame(arrival_depart_data).to_excel(writer, sheet_name="9_到达离去时间", index=False)

            # Sheet 10: 车辆轨迹
            trajectory_data = []
            for tid, points in self.vehicle_trajectory.items():
                for p in points:
                    trajectory_data.append({
                        "车辆ID": tid,
                        "时间(s)": p["time"],
                        "X坐标": p["x"],
                        "Y坐标": p["y"],
                        "所在车道": p["lane"]
                    })
            if trajectory_data:
                pd.DataFrame(trajectory_data).to_excel(writer, sheet_name="10_车辆轨迹", index=False)

            # Sheet 11: 瞬时速度
            if self.instant_speed_records:
                pd.DataFrame(self.instant_speed_records).to_excel(writer, sheet_name="11_瞬时速度", index=False)

            # Sheet 12: 过线瞬时速度
            if self.cross_instant_speed_records:
                pd.DataFrame(self.cross_instant_speed_records).to_excel(writer, sheet_name="12_过线瞬时速度",
                                                                        index=False)

    def print_summary(self, frame_count, video_duration):
        """打印统计摘要"""
        print("\n📊 统计摘要:")
        print(f"  - 总处理帧数: {frame_count}")
        print(f"  - 视频时长: {video_duration / 60:.2f} 分钟")
        print(f"  - 识别车辆数: {len(self.vehicle_color)}")
        print(f"  - 区间速度记录数: {len(self.speed_records)}")
        print(f"  - 瞬时速度记录数: {len(self.instant_speed_records)}")
        print(f"  - 过线瞬时速度记录数: {len(self.cross_instant_speed_records)}")
        print(f"  - 车头时距记录数: {len(self.headway_records)}")
        print(f"  - 排队长度记录数: {len(self.queue_records)}")