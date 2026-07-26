from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
import argparse
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy import linalg

# =========================================================
# 计算FID
# =========================================================
def calculate_fid(real_features, fake_features, device='cuda:0'):
    """使用预先计算的特征计算FID"""
    mu_r = real_features.mean(dim=0)
    sigma_r = torch.cov(real_features.T)

    mu_f = fake_features.mean(dim=0)
    sigma_f = torch.cov(fake_features.T)

    diff = mu_r - mu_f

    # 计算协方差矩阵的平方根
    sigma_r_np = sigma_r.cpu().numpy()
    sigma_f_np = sigma_f.cpu().numpy()
    covmean_np = linalg.sqrtm(sigma_r_np @ sigma_f_np)
    covmean = torch.from_numpy(np.real(covmean_np)).to(sigma_r.device)

    if torch.is_complex(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + torch.trace(sigma_r + sigma_f - 2 * covmean)
    return fid.item()


def get_inception_model(device='cuda:0'):
    """初始化InceptionV3模型（仅保留特征提取部分）"""
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = torch.nn.Identity()  # 移除分类层，保留特征提取
    return inception


def main():
    # 设备配置
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 初始化LPIPS损失函数
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_dir', type=str, required=True, help='预测图像文件夹路径')
    parser.add_argument('--label_dir', type=str, required=True, help='真实标签图像文件夹路径')
    parser.add_argument('--save_file', type=str, required=True, help='结果保存文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='FID计算的批次大小（根据显存调整）')
    args = parser.parse_args()

    # 路径处理
    delta_recon_root = Path(args.prediction_dir)
    label_root = Path(args.label_dir)
    save_file = Path(args.save_file)

    # 检查路径有效性
    if not delta_recon_root.exists():
        raise ValueError(f"预测图像文件夹不存在: {delta_recon_root}")
    if not label_root.exists():
        raise ValueError(f"标签文件夹不存在: {label_root}")

    # 初始化指标列表
    psnr_delta_list = []
    lpips_delta_list = []

    # 获取所有图像文件
    delta_files = sorted(delta_recon_root.glob('*.png'))
    num_files = len(delta_files)
    if num_files == 0:
        raise ValueError(f"预测图像文件夹中没有PNG文件: {delta_recon_root}")
    print(f"找到 {num_files} 个图像文件，开始计算指标...")

    # 初始化Inception模型
    inception_model = get_inception_model(device)

    # 存储FID特征（CPU存储，节省显存）
    real_features_list = []
    fake_features_list = []

    # 初始化批次变量（提前设为None，避免未定义错误）
    current_real_batch = None
    current_fake_batch = None

    # 修改批次处理部分
    for idx in tqdm(range(num_files)):
        fname = str(idx).zfill(4)

        try:
            label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
            delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
        except Exception as e:
            print(f"读取图像 {fname} 失败: {e}")
            continue

        # 计算PSNR
        psnr_delta = peak_signal_noise_ratio(label, delta_recon)
        psnr_delta_list.append(psnr_delta)

        # 转换为tensor并归一化到[-1,1]
        delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).float().to(device) * 2. - 1.
        label = torch.from_numpy(label).permute(2, 0, 1).float().to(device) * 2. - 1.

        delta_recon = delta_recon.view(1, 3, 768, 768)
        label = label.view(1, 3, 768, 768)

        # 计算LPIPS
        with torch.no_grad():
            delta_d = loss_fn_vgg(delta_recon, label)
        lpips_delta_list.append(delta_d.item())  # 直接存储数值

        # 累积批次（当前值已经是[-1,1]）
        real_img = label
        fake_img = delta_recon

        if current_real_batch is None:
            current_real_batch = real_img
            current_fake_batch = fake_img
        else:
            current_real_batch = torch.cat([current_real_batch, real_img], dim=0)
            current_fake_batch = torch.cat([current_fake_batch, fake_img], dim=0)

        # 批次满或最后一个样本时，计算特征
        if current_real_batch.shape[0] >= args.batch_size or idx == num_files - 1:
            # 调整大小到Inception需要的299x299
            real_resized = F.interpolate(
                current_real_batch, size=(299, 299), mode='bilinear', align_corners=False
            )
            fake_resized = F.interpolate(
                current_fake_batch, size=(299, 299), mode='bilinear', align_corners=False
            )

            real_resized = (real_resized + 1) / 2  # [-1,1] -> [0,1]
            fake_resized = (fake_resized + 1) / 2  # [-1,1] -> [0,1]

            # 计算特征并转移到CPU
            with torch.no_grad():
                real_feat = inception_model(real_resized).detach().cpu()
                fake_feat = inception_model(fake_resized).detach().cpu()

            real_features_list.append(real_feat)
            fake_features_list.append(fake_feat)

            # 清理内存
            del current_real_batch, current_fake_batch, real_resized, fake_resized, real_feat, fake_feat
            current_real_batch = None
            current_fake_batch = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 计算PSNR和LPIPS平均值
    if not psnr_delta_list:
        raise ValueError("没有有效图像计算PSNR")
    psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)

    if not lpips_delta_list:
        raise ValueError("没有有效图像计算LPIPS")
    lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)  # 直接用Python计算

    # 计算FID
    if not real_features_list or not fake_features_list:
        raise ValueError("没有有效特征计算FID")
    all_real_features = torch.cat(real_features_list, dim=0).to(device)
    all_fake_features = torch.cat(fake_features_list, dim=0).to(device)
    fid_score = calculate_fid(all_real_features, all_fake_features, device=device)

    # 保存结果
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "w") as f:
        f.write("Metrics Log\n")
        f.write(f"图像数量: {num_files}\n")
        f.write(f"Delta PSNR: {psnr_delta_avg:.4f}\n")
        f.write(f"Delta LPIPS: {lpips_delta_avg:.6f}\n")
        f.write(f"FID: {fid_score:.4f}\n")

    # 打印结果
    print(f"\n计算完成！结果保存至: {save_file}")
    print(f"平均PSNR: {psnr_delta_avg:.4f}")
    print(f"平均LPIPS: {lpips_delta_avg:.6f}")
    print(f"FID分数: {fid_score:.4f}")


if __name__ == "__main__":
    main()