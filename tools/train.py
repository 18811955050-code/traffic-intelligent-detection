# 新增：解决 OpenMP dll 重复加载报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'C:\YOLO\原博主yolo\yolo26n.pt')

    model.train(
        data=r'C:\YOLO\原博主yolo\datasets\cars\dataset.yaml',
        imgsz=416,
        epochs=100,
        batch=8,
        name=r'slim001',
        device=0,
        workers=4,
        lr0=0.01,
        mosaic=1.0,  # 新增：启用 Mosaic 增强，概率为100%
        close_mosaic=10,  # 保留：最后10轮关闭 Mosaic
        half = True #混合精度训练
    )