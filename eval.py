import nibabel as nib
import numpy as np

def calculate_recall(a_path, b_path):
    # 读取 .nii.gz 文件 [[10]]
    a_img = nib.load(a_path)
    b_img = nib.load(b_path)

    # 获取数据数组
    a_data = a_img.get_fdata()
    b_data = b_img.get_fdata()

    # 检查形状是否一致
    if a_data.shape != b_data.shape:
        raise ValueError("输入的两个标签文件形状不一致")

    # 转换为布尔数组，非零表示标签存在
    a_bool = a_data != 0
    b_bool = b_data != 0

    # 计算 TP 和 FN
    tp = np.logical_and(a_bool, b_bool).sum()
    fn = np.logical_and(a_bool, ~b_bool).sum()

    # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return recall

# 示例调用
a_path = '/home/zjc/nnDetection/det_models/Task101_quanbiao/RetinaUNetV003CBAM_D3V001_3d/fold0/test_predictions_nii/case_16_boxes.nii.gz'
b_path = '/home/zjc/nnDetection/det_data/Task101_quanbiao/raw_splitted/labelsTs/case_16.nii.gz'

recall = calculate_recall(a_path, b_path)
print(f"Recall: {recall:.4f}")