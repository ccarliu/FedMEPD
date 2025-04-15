import os
import numpy as np
import nibabel as nib

def convert_npy_to_niigz(path):
    # 遍历给定路径下的所有文件和子文件夹
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                
                # 读取 .npy 文件中的数组
                array = np.load(file_path)
                
                # 将数组的数值类型转换为 np.uint8
                array_uint8 = array.astype(np.uint8)
                
                # 创建一个 NIfTI 图像对象
                nii_image = nib.Nifti1Image(array_uint8, affine=np.eye(4))
                
                # 构建输出文件路径，替换扩展名为 .nii.gz
                output_file_path = os.path.splitext(file_path)[0] + '.nii.gz'
                
                # 保存 NIfTI 图像为 .nii.gz 文件
                nib.save(nii_image, output_file_path)
                print(f"Converted {file_path} to {output_file_path}")
                
                # 删除原始的 .npy 文件
                os.remove(file_path)
                print(f"Deleted original file {file_path}")

# 示例用法
convert_npy_to_niigz('/apdcephfs_cq10/share_1290796/lh/FedMEMA/test_results_2/brats20_rf_c8_1m_pid_1')