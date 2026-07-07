# -*- coding: utf-8 -*-
"""
智能交通信控优化 - 10类数据完整检测程序
整合模块：流量、时间占有率、空间占有率、车头间距/时距、排队长度、瞬时速度、
         车辆类型、车身颜色、到达/离去时间、车辆轨迹
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import time

# 导入各模块
from config import (
    VIDEO_PATH, MODEL_PATH, OUTPUT_EXCEL_PATH,
    LANE_ROIS, STOP_LINE, DETECT_LINE,
    STOP_LINE_Y, DETECT_LINE_Y,
    CONFIDENCE_THRESHOLD, TARGET_CLASSES, CLASS_NAMES,
    OCCUPANCY_INTERVAL, TRAFFIC_FLOW_INTERVAL,
    DISPLAY_SCALE, WAIT_MS
)
from calibration import calc_real_distance
from color_detector import get_car_color
from lane_manager import get_lane_by_roi, point_in_roi, LANE_AREAS
from queue_length import update_queue_length
from speed_calculator import calc_instant_speed
from metrics_recorder import MetricsRecorder


def main():
    print("=" * 60)
    print("智能交通信控优化 - 10类数据完整检测程序")
    print("=" * 60)
    print(f"视频路径: {VIDEO_PATH}")
    print(f"输出路径: {OUTPUT_EXCEL_PATH}")
    print(f"车道: {list(LANE_ROIS.keys())}")
    print("=" * 60)

    start_time = time.time()

    if not os.path.exists(config.MODEL_PATH):
        print("=" * 60)
        print("❌ 模型文件不存在！")
        print(f"   期望路径: {config.MODEL_PATH}")
        print("   请从网盘下载 best.pt 并放入 models/ 目录")
        print("=" * 60)
        return

    # 加载模型
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    print(f"视频FPS: {fps:.2f}, 总帧数: {total_frames}, 时长: {video_duration / 60:.2f}分钟")

    # 初始化记录器
    recorder = MetricsRecorder()

    frame_count = 0
    flow_window_start = 0
    last_occupancy_time = time.time()
    last_queue_print_time = 0

    print("\n🚗 开始检测...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time_sec = frame_count / fps

        # YOLO 跟踪
        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            half=True,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES
        )

        current_ids = set()
        current_boxes = []
        h, w = frame.shape[:2]
        vehicle_mask = np.zeros((h, w), dtype=np.uint8)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy_bottom = y2

                current_ids.add(track_id)
                current_boxes.append((track_id, cls_id, x1, y1, x2, y2, cx, cy_bottom))

                cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

                # 轨迹记录
                recorder.vehicle_trajectory[track_id].append({
                    "time": round(current_time_sec, 2),
                    "x": cx,
                    "y": cy_bottom,
                    "lane": get_lane_by_roi(cx, cy_bottom)
                })

                # 颜色识别
                if track_id not in recorder.vehicle_color:
                    roi = frame[y1:y2, x1:x2]
                    recorder.vehicle_color[track_id] = get_car_color(roi)

                # 轨迹历史（用于速度/排队计算）
                recorder.track_history[track_id].append((cx, cy_bottom, current_time_sec))
                if len(recorder.track_history[track_id]) > 10:
                    recorder.track_history[track_id].pop(0)

                # 瞬时速度（实时）
                instant_speed = calc_instant_speed(recorder.track_history, track_id, fps, smooth_window=3)
                if 0.5 < instant_speed < 200:
                    recorder.instant_speed_records.append({
                        "车辆ID": track_id,
                        "时间(s)": round(current_time_sec, 2),
                        "车道": get_lane_by_roi(cx, cy_bottom),
                        "车型": CLASS_NAMES[cls_id],
                        "瞬时速度(km/h)": round(instant_speed, 1)
                    })

                lane_name = get_lane_by_roi(cx, cy_bottom)
                if lane_name:
                    recorder.vehicle_type_count[lane_name][CLASS_NAMES[cls_id]] += 1

                if lane_name:
                    detect_y = DETECT_LINE_Y[lane_name]
                    stop_y = STOP_LINE_Y[lane_name]

                    if track_id not in recorder.vehicle_states:
                        recorder.vehicle_states[track_id] = {"detect_crossed": False, "stop_crossed": False}
                    state = recorder.vehicle_states[track_id]

                    # 到达检测线
                    if not state["detect_crossed"] and cy_bottom <= detect_y:
                        state["detect_crossed"] = True
                        state["detect_time"] = current_time_sec
                        recorder.arrival_time[track_id] = {"lane": lane_name, "time": current_time_sec}

                        if track_id not in recorder.counted_ids:
                            recorder.lane_flow_count[lane_name] += 1
                            recorder.counted_ids.add(track_id)

                        # 过线瞬时速度
                        cross_speed = calc_instant_speed(recorder.track_history, track_id, fps, smooth_window=3)
                        if 0.5 < cross_speed < 200:
                            recorder.cross_instant_speed_records.append({
                                "车辆ID": track_id,
                                "过线时间(s)": round(current_time_sec, 2),
                                "车道": lane_name,
                                "车型": CLASS_NAMES[cls_id],
                                "颜色": recorder.vehicle_color.get(track_id, "未知"),
                                "过线瞬时速度(km/h)": round(cross_speed, 1)
                            })

                        # 车头间距/时距
                        if recorder.last_pass_time[lane_name] is not None:
                            headway = current_time_sec - recorder.last_pass_time[lane_name]
                            if recorder.last_pass_point[lane_name] is not None:
                                spacing_m = calc_real_distance(
                                    recorder.last_pass_point[lane_name][0],
                                    recorder.last_pass_point[lane_name][1],
                                    cx, cy_bottom
                                )
                                recorder.headway_records.append({
                                    "车道": lane_name,
                                    "前车ID": recorder.last_pass_point[lane_name][2] if len(
                                        recorder.last_pass_point[lane_name]) > 2 else "N/A",
                                    "后车ID": track_id,
                                    "时间": round(current_time_sec, 2),
                                    "车头时距(秒)": round(headway, 2),
                                    "车头间距(米)": round(spacing_m, 2)
                                })

                        recorder.last_pass_time[lane_name] = current_time_sec
                        recorder.last_pass_point[lane_name] = (cx, cy_bottom, track_id)

                    # 越过停止线
                    if not state["stop_crossed"] and cy_bottom <= stop_y:
                        state["stop_crossed"] = True
                        state["stop_time"] = current_time_sec
                        recorder.departure_time[track_id] = {"lane": lane_name, "time": current_time_sec}

                        if "detect_time" in state:
                            dt = current_time_sec - state["detect_time"]
                            if dt > 0:
                                dist_m = calc_real_distance(cx, detect_y, cx, stop_y)
                                speed_kmh = (dist_m / dt) * 3.6
                                recorder.speed_records.append({
                                    "车辆ID": track_id,
                                    "车道": lane_name,
                                    "车型": CLASS_NAMES[cls_id],
                                    "颜色": recorder.vehicle_color[track_id],
                                    "到达时间(s)": round(state["detect_time"], 2),
                                    "离开时间(s)": round(current_time_sec, 2),
                                    "时间差(s)": round(dt, 2),
                                    "速度(km/h)": round(speed_kmh, 1)
                                })

        # 清理无效轨迹
        for tid in list(recorder.track_history.keys()):
            if tid not in current_ids:
                del recorder.track_history[tid]

        # 空间占有率
        for lane_name, roi in LANE_ROIS.items():
            lane_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(lane_mask, [roi.astype(np.int32)], 255)
            intersect = cv2.bitwise_and(vehicle_mask, lane_mask)
            veh_area = np.count_nonzero(intersect)
            if LANE_AREAS[lane_name] > 0:
                single_occ = (veh_area / LANE_AREAS[lane_name]) * 100
                single_occ = min(single_occ, 100.0)
                recorder.lane_space_occ_sum[lane_name] += single_occ
                recorder.lane_space_occ_frame_count[lane_name] += 1

        # 时间占有率
        for lane_name, roi in LANE_ROIS.items():
            has_vehicle = False
            for (tid, cls_id, x1, y1, x2, y2, cx, cy_bottom) in current_boxes:
                if point_in_roi(cx, cy_bottom, roi):
                    has_vehicle = True
                    break
            if has_vehicle:
                recorder.lane_time_occ[lane_name] += (1.0 / fps)

        # 排队长度
        queue_length_m = update_queue_length(
            current_boxes,
            recorder.track_history,
            STOP_LINE_Y
        )

        if current_time_sec - last_queue_print_time >= 1.0:
            recorder.queue_records.append({
                "时间(秒)": round(current_time_sec, 1),
                "ev-line排队长度(米)": queue_length_m["ev-line"],
                "str1排队长度(米)": queue_length_m["str1"],
                "str2排队长度(米)": queue_length_m["str2"],
                "left排队长度(米)": queue_length_m["left"]
            })
            last_queue_print_time = current_time_sec

        # 定时统计输出
        now = time.time()
        if now - last_occupancy_time >= OCCUPANCY_INTERVAL:
            print(f"\n===== {round(current_time_sec, 1)}秒 - 5秒统计 =====")
            for lane_name in LANE_ROIS:
                time_occ = (recorder.lane_time_occ[lane_name] / OCCUPANCY_INTERVAL) * 100
                time_occ = min(time_occ, 100.0)
                recorder.lane_time_occ_history[lane_name].append(time_occ)

                frame_cnt = recorder.lane_space_occ_frame_count[lane_name]
                space_occ = recorder.lane_space_occ_sum[lane_name] / frame_cnt if frame_cnt > 0 else 0
                space_occ = min(space_occ, 100.0)
                recorder.lane_space_occ_history[lane_name].append(space_occ)

                print(f"  {lane_name}: 时间占有率={time_occ:.1f}%, 空间占有率={space_occ:.1f}%")

                recorder.lane_time_occ[lane_name] = 0
                recorder.lane_space_occ_sum[lane_name] = 0
                recorder.lane_space_occ_frame_count[lane_name] = 0
            last_occupancy_time = now

        # 分钟流量统计
        if current_time_sec - flow_window_start >= TRAFFIC_FLOW_INTERVAL:
            minute_idx = int(flow_window_start / 60)
            print(f"\n===== 第{minute_idx + 1}分钟 - 流量统计 =====")
            for lane_name in LANE_ROIS:
                flow = recorder.lane_flow_count[lane_name]
                recorder.lane_flow_minute[minute_idx][lane_name] = flow
                print(f"  {lane_name}: {flow} 辆/分钟")
            recorder.lane_flow_count = {lane: 0 for lane in LANE_ROIS}
            recorder.counted_ids.clear()
            flow_window_start = current_time_sec

        # ===== 可视化 =====
        display_frame = frame.copy()

        colors = {
            "ev-line": (0, 0, 255),
            "str1": (0, 255, 0),
            "str2": (255, 0, 0),
            "left": (255, 255, 0)
        }

        for lane_name, roi in LANE_ROIS.items():
            roi_pts = roi.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(display_frame, [roi_pts], True, colors[lane_name], 2)

            if lane_name in STOP_LINE:
                pt1 = STOP_LINE[lane_name][0]
                pt2 = STOP_LINE[lane_name][1]
                cv2.line(display_frame, pt1, pt2, (0, 255, 255), 3)

            if lane_name in DETECT_LINE:
                pt1 = DETECT_LINE[lane_name][0]
                pt2 = DETECT_LINE[lane_name][1]
                cv2.line(display_frame, pt1, pt2, (255, 0, 255), 2)

        for (tid, cls_id, x1, y1, x2, y2, cx, cy_bottom) in current_boxes:
            cv2.circle(display_frame, (cx, cy_bottom), 6, (0, 0, 255), -1)
            cv2.putText(display_frame, f"ID:{tid}", (cx - 20, cy_bottom - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        y_offset = 30
        cv2.putText(display_frame, "=== Traffic Detection ===", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 30
        for lane_name in LANE_ROIS:
            text = f"{lane_name}: Q={queue_length_m[lane_name]:.2f}m"
            cv2.putText(display_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[lane_name], 2)
            y_offset += 25

        show_frame = cv2.resize(display_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("10 Classes Traffic Detection", show_frame)
        if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
            break

    # 帧率评估
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n实际处理帧率: {actual_fps:.2f} fps")
    if actual_fps >= 15:
        print("✅ 帧率达标（≥15fps），实时性评分：15分")
    else:
        print(f"❌ 帧率未达标（{actual_fps:.1f}fps<15fps），扣{int(15 - actual_fps)}分")

    cap.release()
    cv2.destroyAllWindows()

    # 导出Excel
    print("\n" + "=" * 60)
    print("📊 正在汇总数据并导出Excel...")
    print("=" * 60)
    recorder.export_to_excel(OUTPUT_EXCEL_PATH)

    print("\n" + "=" * 60)
    print("✅ 检测完成！")
    print(f"📁 输出文件: {OUTPUT_EXCEL_PATH}")
    print("=" * 60)
    recorder.print_summary(frame_count, video_duration)
    print("=" * 60)


if __name__ == "__main__":
    main()