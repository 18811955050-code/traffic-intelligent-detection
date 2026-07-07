import cv2
import numpy as np

# ===================== 【配置区】 =====================
VIDEO_PATH = r"C:\YOLO\原博主yolo\video test001\north.mp4"
SCALE_FACTOR = 0.5  # 必须和你的主程序完全一样！
TARGET_FRAME = 1000  # 【关键参数】你想取第几帧，改这个数字就行
# =====================================================================

# 全局变量
points = []       # 存储当前区域的点
regions = []      # 存储所有闭合区域
frame = None
display_frame = None


def mouse_click(event, x, y, flags, param):
    global points, regions, display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # 还原为视频原始真实坐标
        real_x = int(x / SCALE_FACTOR)
        real_y = int(y / SCALE_FACTOR)

        points.append((real_x, real_y))
        print(f"✅ 拾取点：({real_x}, {real_y})")

        # 4个点 → 自动闭合为一个车道区域
        if len(points) == 4:
            regions.append(points.copy())
            print(f"✅ 车道区域保存完成：{points}")
            points = []  # 清空，准备画下一个区域

        # 刷新画面
        redraw()


def redraw():
    global display_frame, frame
    display_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    # 绘制已保存的所有车道区域
    for region in regions:
        pts = np.array(region, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

    # 绘制当前未闭合的点
    for (rx, ry) in points:
        px = int(rx * SCALE_FACTOR)
        py = int(ry * SCALE_FACTOR)
        cv2.circle(display_frame, (px, py), 5, (0, 0, 255), -1)

    cv2.imshow("车道区域取点工具 - 按q退出", display_frame)


# ===================== 打开视频并跳转到指定帧 =====================
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
ret, frame = cap.read()

if not ret:
    print(f"❌ 无法读取第 {TARGET_FRAME} 帧，请检查帧数是否正确！")
    cap.release()
    exit()

# ===================== 鼠标交互 =====================
cv2.namedWindow("车道区域取点工具 - 按q退出")
cv2.setMouseCallback("车道区域取点工具 - 按q退出", mouse_click)
redraw()

print("=" * 60)
print("🎯 使用说明：")
print(f"1. 当前正在显示：第 {TARGET_FRAME} 帧画面")
print("2. 左键依次点击 4 个点 → 自动生成一个车道区域")
print("3. 可连续画多个车道")
print("4. 按 q 退出并输出最终坐标")
print("=" * 60)

# 等待退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ===================== 输出最终结果 =====================
print("\n" + "=" * 60)
print("📌 直接复制到占有率程序的车道坐标：")
for i, region in enumerate(regions):
    print(f'"车道{i+1}": np.array({region}),')
print("=" * 60)