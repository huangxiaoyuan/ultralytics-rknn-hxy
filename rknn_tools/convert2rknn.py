#安装了rknn toolkit使用
from rknn.api import RKNN
import os

# 配置路径
ONNX_PATH = "yolo11s.onnx"  # 输入ONNX模型
RKNN_PATH = "yolo11s.rknn"  # 输出RKNN模型
CALIB_PATH = "calib_dataset.txt"  # 校准图路径文件


def main():
    # 1. 创建RKNN实例，开启日志（方便调试）
    rknn = RKNN(verbose=True)

    # 2. 加载ONNX模型（关键：指定输入尺寸和归一化参数）
    print("=== 加载ONNX模型 ===")
    ret = rknn.load_onnx(
        model=ONNX_PATH,
        # 声明输入节点信息：名称（从Netron查看）、尺寸、数据类型
        inputs=["images"],  # YOLO11的输入节点名固定为images
        input_size_list=[[3, 640, 640]],  # NCHW格式
        mean_values=[[0, 0, 0]],  # 归一化均值（YOLO11用0）
        std_values=[[255, 255, 255]]  # 归一化标准差（YOLO11用255，对应输入除以255）
    )
    if ret != 0:
        print("加载ONNX模型失败！")
        exit(ret)

    # 3. 配置NPU运行参数（针对RK3588优化）
    print("=== 配置模型参数 ===")
    rknn.config(
        target_platform="rk3588",  # 目标硬件
        optimization_level=3,  # 优化级别（3为最高）
        quantize_input_node=True,  # 对输入节点量化（提升速度）
        # 若需混合精度量化（部分层用FP16），添加以下行：
        # precision_mode="hybrid",
        # float_dtype="fp16"
    )

    # 4. 构建模型（量化核心步骤，耗时约10分钟）
    print("=== 构建模型 ===")
    ret = rknn.build(
        do_quantization=True,  # 开启INT8量化
        dataset=CALIB_PATH,  # 校准数据集
        quantized_dtype="int8"  # 量化类型
    )
    if ret != 0:
        print("构建模型失败！")
        exit(ret)

    # 5. 导出RKNN模型
    print("=== 导出RKNN模型 ===")
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print("导出模型失败！")
        exit(ret)

    # 6. 可选：测试模型在NPU上的推理效果（验证精度）
    print("=== 初始化运行时 ===")
    ret = rknn.init_runtime()
    if ret != 0:
        print("初始化运行时失败！")
        exit(ret)

    # 用一张测试图验证（可选）
    import cv2
    import numpy as np
    img = cv2.imread("calib_images/00000000100.jpg")  # 随便选一张校准图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC→CHW
    img = img.astype(np.float32) / 255.0  # 归一化
    img = np.expand_dims(img, axis=0)  # 加batch维度

    print("=== 测试推理 ===")
    outputs = rknn.inference(inputs=[img])
    print(f"推理输出形状：{outputs[0].shape}")  # 应输出(1, 8400, 85)

    # 7. 释放资源
    rknn.release()
    print("=== 转换完成 ===")


if __name__ == "__main__":
    main()