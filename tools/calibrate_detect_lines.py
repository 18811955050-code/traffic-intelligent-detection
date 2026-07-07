# 新增：解决 OpenMP dll 重复加载报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import json

lanes = {}
current_lane = 0
current_line = "line1"  # 'line1' or 'line2'
points = []

def mouse_callback(event, x, y, flags, param):
    global points, current_line, current_lane, lanes
    frame = param
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0,255,0), -1)
        cv2.imshow("Calibration", frame)
        if len(points) == 2:
            if current_lane not in lanes:
                lanes[current_lane] = {}
            lanes[current_lane][current_line] = points.copy()
            color = (255,0,0) if current_line == "line1" else (0,255,255)
            cv2.line(frame, points[0], points[1], color, 3)
            cv2.imshow("Calibration", frame)
            print(f"Lane {current_lane} {current_line} saved: {points}")
            points = []
            if current_line == "line1":
                current_line = "line2"
                print(f"Lane {current_lane} - click two points for LINE2")
            else:
                current_lane += 1
                current_line = "line1"
                print(f"\nLane {current_lane} - click two points for LINE1")

VIDEO_PATH = r"C:\YOLO\原博主yolo\video test001\南进口_20260420075959至20260420081500.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", 1200, 700)
cv2.setMouseCallback("Calibration", mouse_callback, frame)

print("=== 车道线标定（每条车道两条线）===")
print(f"Lane {current_lane} - click two points for LINE1 (blue)")

while True:
    display = frame.copy()
    for lid, lines in lanes.items():
        if "line1" in lines:
            cv2.line(display, lines["line1"][0], lines["line1"][1], (255,0,0), 3)
        if "line2" in lines:
            cv2.line(display, lines["line2"][0], lines["line2"][1], (0,255,255), 3)
    for p in points:
        cv2.circle(display, p, 5, (0,255,0), -1)
    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        with open("south_lines_config.json", "w") as f:
            json.dump(lanes, f, indent=2)
        print("配置已保存")
cv2.destroyAllWindows()