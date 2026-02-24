# 输入模型路径，从test loader中随机采样一点图片，输入，输出label uncertainty和predicted uncertainty的对比图
import timm
import os
import torch
from torch.utils.data import DataLoader
from regress_model import ViTRegressor
from regress_dataset import RegressionDataset, RegressionDataset_singleclass, RegressionDataset_final
import sys
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from torchvision.utils import save_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../../"))                 # PUN
shapenet_path = os.path.join(BASE_DIR, "../../../../shapenet/ShapeNetCore.v2") # absolute path
NUM_path = os.path.join(BASE_DIR, "../../data/shapenet/NUM_example")

from fep_nbv.utils.generate_viewpoints import *
from fep_nbv.utils.transform_viewpoints import *

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np


def plot_uncertainty_scatter(outputs_all, labels_all, save_path="uncertainty_scatter.png"):
    """
    绘制 predicted uncertainty vs ground-truth uncertainty 的散点图。

    参数:
        outputs_all: List[np.ndarray], 每个元素是模型预测的 48 维向量（或 1 维）
        labels_all:  List[np.ndarray], 每个元素是 GT 的 48 维向量（或 1 维）
    """
    outputs_all = np.stack(outputs_all, axis=0)     # [N, D]
    labels_all = np.stack(labels_all, axis=0)       # [N, D]

    D = outputs_all.shape[1]

    if D == 1:
        # ---------- 单维情况 ----------
        pred = outputs_all[:, 0]
        gt = labels_all[:, 0]

        plt.figure(figsize=(6, 6))
        plt.scatter(gt, pred, alpha=0.5, s=10)
        plt.xlabel("GT Uncertainty")
        plt.ylabel("Predicted Uncertainty")
        plt.title("Predicted vs GT Uncertainty (1D)")

        # 回归线
        coef = np.polyfit(gt, pred, deg=1)
        x_line = np.linspace(min(gt), max(gt), 200)
        plt.plot(x_line, coef[0] * x_line + coef[1], color="red", linewidth=2)

        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved scatter plot to {save_path}")

    else:
        # ---------- 多维情况：可视化前 48 维之一 ----------
        for dim in range(D):
            pred = outputs_all[:, dim]
            gt = labels_all[:, dim]

            plt.figure(figsize=(6, 6))
            plt.scatter(gt, pred, alpha=0.5, s=8)
            plt.xlabel("GT Uncertainty")
            plt.ylabel("Predicted Uncertainty")
            plt.title(f"Pred vs GT Uncertainty (Dim {dim})")

            # 回归线
            coef = np.polyfit(gt, pred, deg=1)
            x_line = np.linspace(min(gt), max(gt), 200)
            plt.plot(x_line, coef[0] * x_line + coef[1], color="red", linewidth=2)

            plt.tight_layout()
            save_name = save_path.replace(".png", f"_dim{dim}.png")
            plt.savefig(save_name, dpi=300)
            plt.close()
            print(f"Saved: {save_name}")

def compute_correlation(outputs_list, labels_list):
    """
    输入：
        outputs_list: List[np.ndarray], 每个元素是模型输出的一个样本（通常为 shape=[48]）
        labels_list: List[np.ndarray], 每个元素是对应的 ground truth label（shape=[48]）

    返回：
        overall_corr: float, 将所有样本的所有维度拼接后的一维整体相关性
        dim_corrs: np.ndarray of shape [48], 每个维度的 Pearson 相关系数
    """
    outputs_arr = np.array(outputs_list)  # shape: [N, 48]
    labels_arr = np.array(labels_list)    # shape: [N, 48]

    # 计算整体的相关性（flatten 后）
    overall_corr, _ = pearsonr(outputs_arr.flatten(), labels_arr.flatten())

    # 每个维度分别计算相关性
    dim_corrs = np.zeros(outputs_arr.shape[1])
    for i in range(outputs_arr.shape[1]):
        dim_corrs[i], _ = pearsonr(outputs_arr[:, i], labels_arr[:, i])

    return overall_corr, dim_corrs

def visualize_HEALPix_distribution_polar2(prediction, groundtruth, n_side=2, save_path=None, original_viewpoint=np.array([0,0,1]), mode='PSNR'):
    poses = generate_HEALPix_viewpoints(n_side=n_side, original_viewpoint=original_viewpoint)
    phi, theta = pose2polar(poses)  # (N,), (N,)
    if mode=="PSNR" or mode=="SSIM":
        prediction_norm = (np.max(prediction) - prediction) / (np.max(prediction) - np.min(prediction) + 1e-8)
        groundtruth_norm = (np.max(groundtruth) - groundtruth) / (np.max(groundtruth) - np.min(groundtruth) + 1e-8)
    else:
        prediction_norm = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction) + 1e-8)
        groundtruth_norm = (groundtruth - np.min(groundtruth)) / (np.max(groundtruth) - np.min(groundtruth) + 1e-8)

    # 归一化 prediction 和 groundtruth
    norm = Normalize(vmin=0.0, vmax=1.0)
    levels = np.linspace(0.0, 1.0, 9) 

    # prediction_norm = Normalize(vmin=np.min(prediction), vmax=np.max(prediction))(prediction)
    # groundtruth_norm = Normalize(vmin=np.min(groundtruth), vmax=np.max(groundtruth))(groundtruth)
    # prediction_norm = prediction_norm
    # groundtruth_norm = groundtruth_norm

    # 创建网格
    phi_grid, theta_grid = np.meshgrid(
        np.linspace(0, 2 * np.pi, 200),
        np.linspace(0, np.pi, 100)
    )

    # 插值 prediction
    prediction_grid = griddata(
        points=np.stack([phi, theta], axis=1),
        values=prediction_norm,
        xi=np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(phi_grid.shape)

    # 插值 groundtruth
    groundtruth_grid = griddata(
        points=np.stack([phi, theta], axis=1),
        values=groundtruth_norm,
        xi=np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(phi_grid.shape)

    fig_pred = plt.figure(figsize=(6, 6))
    ax_pred = fig_pred.add_subplot(111, polar=True)
    ax_pred.tick_params(labelsize=14)
    im = ax_pred.contourf(phi_grid, theta_grid, prediction_grid, cmap='viridis', norm=norm, levels=levels)
    r_tick_labels_deg = [0, 45, 90, 135, 180]
    r_tick_locations_rad = np.radians(r_tick_labels_deg)
    ax_pred.set_yticks(r_tick_locations_rad)
    ax_pred.set_yticklabels(r_tick_labels_deg)
    ax_pred.axis('off')  # 去掉坐标轴
    # fig_pred.colorbar(im, ax=ax_pred, shrink=0.7)
    fig_pred.savefig(save_path.replace('gt','pred'), bbox_inches='tight', pad_inches=0)
    plt.close(fig_pred)

    fig_gt = plt.figure(figsize=(6, 6))
    ax_gt = fig_gt.add_subplot(111, polar=True)
    ax_gt.tick_params(labelsize=14)
    im = ax_gt.contourf(phi_grid, theta_grid, groundtruth_grid, cmap='viridis', norm=norm,levels=levels)
    r_tick_labels_deg = [0, 45, 90, 135, 180]
    r_tick_locations_rad = np.radians(r_tick_labels_deg)
    ax_gt.set_yticks(r_tick_locations_rad)
    ax_gt.set_yticklabels(r_tick_labels_deg)
    ax_gt.axis('off')  # 去掉坐标轴
    # fig_gt.colorbar(im, ax=ax_gt, shrink=0.7)
    fig_gt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig_gt)

    
def visualize_HEALPix_distribution_polar(prediction, groundtruth, n_side=2, save_path=None, original_viewpoint=np.array([0,0,1])):
    poses = generate_HEALPix_viewpoints(n_side=n_side, original_viewpoint=original_viewpoint)
    phi, theta = pose2polar(poses)  # (N,), (N,)

    # 归一化 prediction 和 groundtruth
    prediction_norm = Normalize(vmin=np.min(prediction), vmax=np.max(prediction))(prediction)
    groundtruth_norm = Normalize(vmin=np.min(groundtruth), vmax=np.max(groundtruth))(groundtruth)
    prediction_norm = prediction_norm
    groundtruth_norm = groundtruth_norm

    # 创建网格
    phi_grid, theta_grid = np.meshgrid(
        np.linspace(0, 2 * np.pi, 200),
        np.linspace(0, np.pi, 100)
    )

    # 插值 prediction
    prediction_grid = griddata(
        points=np.stack([phi, theta], axis=1),
        values=prediction_norm,
        xi=np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(phi_grid.shape)

    # 插值 groundtruth
    groundtruth_grid = griddata(
        points=np.stack([phi, theta], axis=1),
        values=groundtruth_norm,
        xi=np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=1),
        method='nearest',
        fill_value=0
    ).reshape(phi_grid.shape)

    # 开始画图
    fig = plt.figure(figsize=(16, 6))
    
    # Prediction
    ax1 = fig.add_subplot(121, polar=True)
    im1 = ax1.contourf(phi_grid, theta_grid, prediction_grid, cmap='viridis')
    fig.colorbar(im1, ax=ax1, shrink=0.7)
    ax1.set_title("Prediction Polar Map")

    # Groundtruth
    ax2 = fig.add_subplot(122, polar=True)
    im2 = ax2.contourf(phi_grid, theta_grid, groundtruth_grid, cmap='viridis')
    fig.colorbar(im2, ax=ax2, shrink=0.7)
    ax2.set_title("Groundtruth Polar Map")

    # Save or show
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

if __name__=='__main__':
    model_path = 'data/UPNet/vit_small_patch16_224_PSNR_250425172703'
    model_ckpt = os.path.join(model_path, 'best_vit_regressor.pth')
    dataset_path = NUM_path
    os.makedirs(os.path.join(model_path, "images"), exist_ok=True)

    for mode in ['_SSIM', '_PSNR', '_MSE',"_LPIPS"]:
        if len(model_path.split(mode)) > 1:
            uncertainty_mode = mode
            break
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vit_used = model_path.split('/')[-1].split(uncertainty_mode)[0]
    model = ViTRegressor(model_name=vit_used,output_dim=48).to(device)
    checkpoint = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(checkpoint)  # 注意 key 可能要对齐
    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    test_dataset = RegressionDataset_final(
        root_dir=dataset_path,  # 替换为你的路径
        transform=transform,
        mode=uncertainty_mode[1:],
        split='test_all'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    i=0
    i_max=50
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)           
            if i<(i_max+10) and i>i_max:
                save_image(images[0], os.path.join(model_path, "images", f"{i}_input_img_{i}.png"))
                visualize_HEALPix_distribution_polar2(
                    prediction=outputs.cpu().numpy()[0],
                    groundtruth=labels.cpu().numpy()[0],
                    n_side=2,
                    save_path=os.path.join(model_path, "images", f"{i}_polar_map_gt_{i}.png")
                )
                i+=1
            elif i<(i_max+30):
                i+=1
                continue
            else:
                break

    outputs_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            outputs_all.append(outputs.cpu().numpy()[0])
            labels_all.append(labels.cpu().numpy()[0])
    overall_corr, dim_corrs = compute_correlation(outputs_all, labels_all)
    print("Overall correlation:", overall_corr)
    print("Per-dimension correlations:", dim_corrs)

    plot_uncertainty_scatter(outputs_all, labels_all,save_path="uncertainty_scatter.png")