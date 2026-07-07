import cv2
import numpy as np
import os

# ===================== 配置 =====================
VIDEO_PATH = r"C:\YOLO\原博主yolo\video test001\north.mp4"
TARGET_FRAME = 200
DISPLAY_SCALE = 0.5
SAVE_PATH = r"C:\YOLO\原博主yolo\perspective_matrix.npy"

# 实际坐标（按用户定义的坐标系）
# 点1: 左上(远端)   (0, 25)
# 点2: 右上(远端)   (10.5, 25)
# 点3: 左下(近端)   (0, 0)
# 点4: 右下(近端)   (10.5, 0)
REAL_POINTS = np.array([
    [0.0, 25.0],
    [10.5, 25.0],
    [0.0, 0.0],
    [10.5, 0.0]
], dtype=np.float32)

# =============================================

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 无法读取视频")
    exit()

h, w = frame.shape[:2]
print(f"📐 视频原始尺寸: {w} × {h}")
print("\n" + "=" * 60)
print("🎯 一键标定工具（取点 + 生成透视矩阵）")
print("=" * 60)
print("   点1: 最左侧车道【左边缘】远端 → 实际 (0, 25)")
print("   点2: 最右侧车道【右边缘】远端 → 实际 (10.5, 25)")
print("   点3: 最左侧车道【左边缘】近端 → 实际 (0, 0)")
print("   点4: 最右侧车道【右边缘】近端 → 实际 (10.5, 0)")
print("")
print("💡 确保点1和点3在相同X位置（最左边缘）")
print("💡 确保点2和点4在相同X位置（最右边缘）")
print("   取完4个点后自动计算矩阵并保存")
print("   按 q 退出（不会保存）")
print("=" * 60)

points = []  # 存储像素坐标
display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

# 绘制参考网格
for i in range(0, w, 100):
    cv2.line(display_frame, (int(i*DISPLAY_SCALE), 0), (int(i*DISPLAY_SCALE), int(h*DISPLAY_SCALE)), (255,255,255,50), 1)
for i in range(0, h, 100):
    cv2.line(display_frame, (0, int(i*DISPLAY_SCALE)), (int(w*DISPLAY_SCALE), int(i*DISPLAY_SCALE)), (255,255,255,50), 1)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x / DISPLAY_SCALE)
        real_y = int(y / DISPLAY_SCALE)
        points.append((real_x, real_y))
        label = ["左上(0,25)", "右上(10.5,25)", "左下(0,0)", "右下(10.5,0)"]
        print(f"✅ 点{len(points)} {label[len(points)-1]}: ({real_x}, {real_y})")
        cv2.circle(display_frame, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(display_frame, f"P{len(points)}", (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("取点工具 - 三条车道总宽标定", display_frame)
        if len(points) >= 4:
            print("\n✅ 4个点已取完！正在计算透视矩阵...")
            # 自动计算并保存矩阵
            compute_and_save()

def compute_and_save():
    if len(points) < 4:
        print("❌ 点不足4个，无法计算")
        return
    src_pts = np.array(points[:4], dtype=np.float32)
    dst_pts = REAL_POINTS

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 保存
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.save(SAVE_PATH, M)
    print(f"\n💾 透视矩阵已保存到: {SAVE_PATH}")

    # 验证
    print("\n🔍 验证标定结果（像素→实际坐标）：")
    for i, (px, py) in enumerate(src_pts):
        rx, ry = pixel_to_real(px, py, M)
        print(f"   点{i+1}: 像素({px},{py}) → 实际({rx:.2f}, {ry:.2f})米")

    # 额外验证：计算Y和X方向距离
    _, y1 = pixel_to_real(src_pts[0][0], src_pts[0][1], M)
    _, y3 = pixel_to_real(src_pts[2][0], src_pts[2][1], M)
    x3, _ = pixel_to_real(src_pts[2][0], src_pts[2][1], M)
    x4, _ = pixel_to_real(src_pts[3][0], src_pts[3][1], M)
    print(f"\n📏 Y方向距离（点1→点3）: {abs(y3-y1):.2f} 米 (期望: 25.0米)")
    print(f"📏 X方向距离（点3→点4）: {abs(x4-x3):.2f} 米 (期望: 10.5米)")

    if abs(abs(y3-y1) - 25.0) < 0.5 and abs(abs(x4-x3) - 10.5) < 0.5:
        print("\n✅ 标定成功！矩阵可直接用于十类数据检测程序。")
    else:
        print("\n⚠️ 标定偏差较大，建议重新取点（确保垂直对齐）。")

    print("\n按 q 退出程序")

def pixel_to_real(px, py, M):
    src = np.array([[[px, py]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, M)
    return dst[0][0][0], dst[0][0][1]

cv2.namedWindow("取点工具 - 三条车道总宽标定")
cv2.setMouseCallback("取点工具 - 三条车道总宽标定", mouse_callback)
cv2.imshow("取点工具 - 三条车道总宽标定", display_frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) < 4:
    print("❌ 未取完4个点，未生成矩阵")
else:
    print("\n程序结束")