import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def process_image(img_path):
    """
    读取并重新保存图片，去除多余的 ICC Profile 和 EXIF 等元数据
    """
    try:
        # 转换为字符串路径
        img_str = str(img_path)

        # 包含透明通道读取（如果有的话）
        image = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)

        if image is not None:
            # 判断后缀以决定保存参数
            ext = img_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                # JPG格式：设置保存质量为100，防止二次压缩导致画质受损
                cv2.imwrite(img_str, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                # PNG格式：直接覆盖保存，默认去除了错误的 iCCP chunk
                cv2.imwrite(img_str, image)
            return True
        else:
            # OpenCV 返回 None 说明图片已严重损坏
            return False

    except Exception as e:
        print(f"\n无法处理图片 {img_path}: {e}")
        return False


def fix_dataset_images(dataset_dir, num_workers=8):
    """
    遍历目录并使用多线程修复所有 JPG 和 PNG 图片
    """
    dataset_path = Path(dataset_dir)

    # 支持的图片后缀名
    valid_extensions = {'.png', '.jpg', '.jpeg'}

    print("正在扫描图片文件，请稍候...")
    # 遍历获取所有符合后缀的文件 (统一转换为小写判断，兼容大小写后缀)
    img_files = [
        p for p in dataset_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    total_files = len(img_files)

    if total_files == 0:
        print(f"在目录 {dataset_dir} 中没有找到 JPG 或 PNG 图片。")
        return

    print(f"共找到 {total_files} 张图片，开始清洗元数据...")

    success_count = 0
    # 使用线程池加速 I/O 操作
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 结合 tqdm 显示进度条
        results = list(tqdm(executor.map(process_image, img_files), total=total_files, desc="Processing images"))

    success_count = sum(1 for r in results if r)

    print("\n" + "=" * 40)
    print("✨ 数据清洗与修复完成！")
    print(f"✅ 成功处理: {success_count} / {total_files} 张")

    failed_count = total_files - success_count
    if failed_count > 0:
        print(f"❌ 警告: 有 {failed_count} 张图片已损坏且无法被 OpenCV 读取。")
        print("建议在文件夹中搜索大小为 0KB 的文件或直接删除损坏文件以免报错。")


if __name__ == "__main__":
    # ==========================================
    # 在这里修改为你的数据集所在路径
    # 例如：r"D:\datasets\my_yolo_dataset"
    # ==========================================
    DATASET_DIRECTORY = r"E:\bird\dataset\Calibration_data_1200"

    # 开始修复（num_workers 建议设置为电脑 CPU 核心数，默认8）
    fix_dataset_images(DATASET_DIRECTORY, num_workers=8)