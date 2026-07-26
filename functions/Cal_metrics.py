import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from pytorch_fid import fid_score
from pathlib import Path

# 初始化设备和模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_images_to_tensor(image_paths, transform):
    """
    批量加载图像并转换为张量

    Args:
    image_paths (list): 图像路径列表
    transform (callable): 图像变换

    Returns:
    torch.Tensor: 图像张量
    """
    images_tensor = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        images_tensor.append(img_tensor)

    return torch.cat(images_tensor, dim=0).to(device)


def calculate_batch_metrics(image_pairs, loss_fn):
    """
    批量计算图像对的指标

    Args:
    image_pairs (list): 图像对路径的列表

    Returns:
    dict: 各个指标的列表
    """
    # 准备图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 提取路径
    paths_a = [pair[0] for pair in image_pairs]
    paths_b = [pair[1] for pair in image_pairs]

    # 批量加载图像
    images_a = [np.array(Image.open(path)) for path in paths_a]
    images_b = [np.array(Image.open(path).convert('RGB')) for path in paths_b]


    # PSNR
    psnr_values = [psnr(img_a, img_b) for img_a, img_b in zip(images_a, images_b)]

    # SSIM
    ssim_values = [ssim(img_a, img_b, win_size=7, channel_axis=2, multichannel=True)
                   for img_a, img_b in zip(images_a, images_b)]

    # LPIPS - 批量计算
    images_a_tensor = load_images_to_tensor(paths_a, transform)
    images_b_tensor = load_images_to_tensor(paths_b, transform)



    # 批量计算 LPIPS
    with torch.no_grad():
        lpips_values = loss_fn(images_a_tensor, images_b_tensor).cpu().numpy().flatten()

    return {
        'PSNR': psnr_values,
        'SSIM': ssim_values,
        'LPIPS': lpips_values
    }


def process_image_folders(folder_a, folder_b, save_file, batch_size=50):
    """
    处理文件夹中的图像并计算指标

    Args:
    folder_a (str): 第一个图像文件夹路径
    folder_b (str): 第二个图像文件夹路径
    batch_size (int): 批处理大小

    Returns:
    dict: 平均指标
    """
    # 获取排序后的文件列表
    files_a = sorted(os.listdir(folder_a))
    files_b = sorted(os.listdir(folder_b))

    # 验证文件数量
    assert len(files_a) == len(files_b), "文件夹中的文件数量必须相同"

    # 构建完整路径
    image_pairs = [
        (os.path.join(folder_a, file_a), os.path.join(folder_b, file_b))
        for file_a, file_b in zip(files_a, files_b)
    ]

    # 批量处理指标
    all_metrics = {
        'PSNR': [],
        'SSIM': [],
        'LPIPS': []
    }
    # 初始化一次 LPIPS
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 分批处理
    for i in range(0, len(image_pairs), batch_size):
        batch_pairs = image_pairs[i:i + batch_size]
        batch_metrics = calculate_batch_metrics(batch_pairs,loss_fn_vgg)

        for metric_name, metric_values in batch_metrics.items():
            all_metrics[metric_name].extend(metric_values)

    # 计算FID
    fid_value = fid_score.calculate_fid_given_paths(
        [folder_a, folder_b],
        batch_size=batch_size,
        device=device,
        dims=2048,  # InceptionV3特征
        num_workers=4
    )

    save_file = Path(save_file)
    # 保存结果
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w") as f:
        f.write("Metrics Log\n")
        f.write(f"图像数量: {len(files_a)}\n")
        f.write(f"Delta PSNR: {np.mean(all_metrics['PSNR']):.4f}\n")
        f.write(f"Delta SSIM: {np.mean(all_metrics['SSIM']):.4f}\n")
        f.write(f"Delta LPIPS: {np.mean(all_metrics['LPIPS']):.6f}\n")
        f.write(f"FID: {fid_value:.4f}\n")

    # 返回平均指标
    return {
        'Average PSNR': np.mean(all_metrics['PSNR']),
        'Average SSIM': np.mean(all_metrics['SSIM']),
        'Average LPIPS': np.mean(all_metrics['LPIPS']),
        'FID': fid_value
    }


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='计算两个文件夹间的图像质量指标')

    # 添加源图像和生成图像文件夹参数
    parser.add_argument('--source_folder', type=str, required=True,
                        help='源图像文件夹路径')
    parser.add_argument('--generated_folder', type=str, required=True,
                        help='生成图像文件夹路径')
    parser.add_argument('--save_file', type=str, required=True,
                        help='保存结果的文件路径')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='批处理大小（默认10）')

    # 解析参数
    args = parser.parse_args()

    # 处理图像并计算指标
    results = process_image_folders(
        args.source_folder,
        args.generated_folder,
        args.save_file,
        batch_size=args.batch_size
    )

    # 打印结果
    print("指标结果保存在：", args.save_file)
    print("图像质量指标:")
    for metric, value in results.items():
        print(f"{metric}: {value}")


if __name__ == '__main__':

    main()