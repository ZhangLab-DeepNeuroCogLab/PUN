# 加载模型，输入一个视角，输出这个视角对应的不确定性
import argparse
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys,os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "../../"))                 # PUN
sys.path.append(os.path.join(BASE_DIR, "../../", "08-vit-train")) # PUN/08-vit-train

NMR_dataset_path = '' # NMR dataset path
gt_images_path = os.path.join(BASE_DIR, "../../data/test/gt_temp")
Shapenet_path = os.path.join(BASE_DIR, "../../data/shapenet/instance_example")
root_path = os.path.join(BASE_DIR, "../../")

from fep_nbv.uncertainty_map_generation.scene.gaussian_predictor import GaussianSplatPredictor
from data.splatter_image_datasets.dataset_factory import get_dataset
import numpy as np
import os
import glob 
import json
import time
from PIL import Image
from mathutils import Matrix, Vector


from fep_nbv.utils.generate_viewpoints import generate_HEALPix_viewpoints, pose_from_xyz
from fep_nbv.utils.transform_viewpoints import to_transform
from blender_utils import blender_interface
from fep_nbv.uncertainty_map_generation.uncertainty_utils import generate_gt_images,make_poses_relative_to_first,Metricator,save_img,get_source_cw2wT
from fep_nbv.uncertainty_map_generation.utils.general_utils import matrix_to_quaternion
from data.splatter_image_datasets.dataset_readers import readCamerasFromTxt
from fep_nbv.uncertainty_map_generation.utils.graphics_utils import getWorld2View2,getProjectionMatrix,getView2World,getWorld2View
from gaussian_renderer import render_predicted
from utils.gif_saver import create_gif_from_pngs

shapenet_path = os.path.join(BASE_DIR, "../../../../shapenet/ShapeNetCore.v2") # absolute path
output_path = os.path.join(BASE_DIR, "../../data/shapenet/NUM_example")
NMR_dataset_path = ''

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--dataset_name', type=str,default='nmr', help='Dataset to evaluate on',
                        choices=['objaverse', 'gso', 'cars', 'chairs', 'hydrants', 'teddybears', 'nmr'])
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the parent folder of the model. \
                        If set to None, a pretrained model will be downloaded')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'vis', 'train'],
                        help='Split to evaluate on (default: test). \
                        Using vis renders loops and does not return scores - to be used for visualisation. \
                        You can also use this to evaluate on the training or validation splits.')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save renders (default: out)')
    parser.add_argument('--save_vis', type=int, default=0, help='Number of examples for which to save renders (default: 0)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to evaluate.')
    parser.add_argument('--viewpoint_index', type=int, default=0, help='Viewpoint index to evaluate on.')
    parser.add_argument('--offset_phi_index', type=int, default=0, help='Offset phi index to evaluate on.')
    parser.add_argument('--CUDA_id', type=int, default=0, help='CUDA id to use')
    return parser.parse_args()


def load_camera_info(training_cfg, gt_images_path):
    trans = np.array([0.0, 0.0, 0.0])
    # training_cfg.data.fov = 51.98948897809546
    projection_matrix = getProjectionMatrix(
                    znear=training_cfg.data.znear, zfar=training_cfg.data.zfar,
                    fovX=training_cfg.data.fov * 2 * np.pi / 360, 
                    fovY=training_cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
    rgb_paths = sorted(glob.glob(os.path.join(gt_images_path, "rgb", "*")))
    pose_paths = sorted(glob.glob(os.path.join(gt_images_path, "pose", "*")))
    all_world_view_transforms = []
    all_full_proj_transforms = []
    all_camera_centers = []
    all_view_to_world_transforms = []

    cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))],dataset_name=training_cfg.data.category)

    for i,cam_info in enumerate(cam_infos):
        R = cam_info.R
        T = cam_info.T

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, 1.0)).transpose(0, 1)
        view_world_transform = torch.tensor(getView2World(R, T, trans, 1.0)).transpose(0, 1)

        camera_center = world_view_transform.inverse()[3, :3]
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                
        all_world_view_transforms.append(world_view_transform)
        all_view_to_world_transforms.append(view_world_transform)
        all_full_proj_transforms.append(full_proj_transform)
        all_camera_centers.append(camera_center)
            
    all_world_view_transforms = torch.stack(all_world_view_transforms)
    all_view_to_world_transforms = torch.stack(all_view_to_world_transforms)
    all_full_proj_transforms = torch.stack(all_full_proj_transforms)
    all_camera_centers = torch.stack(all_camera_centers)
    images_and_camera_poses = {
                    "world_view_transforms": all_world_view_transforms,
                    "view_to_world_transforms": all_view_to_world_transforms,
                    "full_proj_transforms": all_full_proj_transforms,
                    "camera_centers": all_camera_centers
                }
    images_and_camera_poses = make_poses_relative_to_first(images_and_camera_poses)
    images_and_camera_poses["source_cv2wT_quat"] = get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])
    return images_and_camera_poses

def calculate_raidus_scale(NMR_dataset_path, model_path):
    class_offset = model_path.split('/')[-2]
    model_index = model_path.split('/')[-1]
    if os.path.exists(os.path.join(NMR_dataset_path,class_offset,model_index)):
        _coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        _coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        _pixelnerf_to_colmap = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        print('load radius from npz')
        all_cam = np.load(os.path.join(NMR_dataset_path,class_offset,model_index,'cameras.npz'))
        radius = np.sqrt((all_cam['world_mat_inv_0']**2).sum()).item()
        c2w_cmo = all_cam['world_mat_inv_0']
        c2w_cmo = (
                _coord_trans_world
                @ torch.tensor(c2w_cmo, dtype=torch.float32)
                @ _coord_trans_cam # to pixelnerf coordinate system
                @ _pixelnerf_to_colmap # to colmap coordinate system
            ) 
        radius = np.sqrt((c2w_cmo[:3, 3] ** 2).sum()).item()
        scale= 2.0
        print(f'radius of {class_offset}/{model_index} is {radius} loaded from npz')
    else:
        radius = 2.73
        scale= 2.0
        print(f'radius of {class_offset}/{model_index} is {radius}')
    return radius,scale

def uncertainty_map_generation(device, training_cfg, background, input_viewpoint_index, metricator, image_path, example_image_path, uncertainty_path, gif_path, gt_images, images_and_camera_poses, reconstruction):
    focals_pixels_render = None
    Uncertainty = {'PSNR':[],'SSIM':[],"MSE":[], "LPIPS":[]}
    images = []
    for r_idx in range(gt_images.shape[0]):
        image = render_predicted({k: v[0].contiguous() for k, v in reconstruction.items()},
                                            images_and_camera_poses['world_view_transforms'][r_idx].to(device),
                                            images_and_camera_poses['full_proj_transforms'][r_idx].to(device), 
                                            images_and_camera_poses['camera_centers'][r_idx].to(device),
                                            background.to(device),
                                            training_cfg,
                                            focals_pixels=focals_pixels_render)["render"]
        gt_rgb = gt_images[r_idx]
        predicted_rgb = image
        images.append(Image.fromarray((predicted_rgb.permute(1, 2, 0)*255).cpu().detach().numpy().astype(np.uint8)))
        psnr, ssim, lpips, mse = metricator.compute_metrics(predicted_rgb, gt_rgb.permute(2, 0, 1)/255)
        Uncertainty['PSNR'].append(psnr)
        Uncertainty['SSIM'].append(ssim)
        Uncertainty['MSE'].append(mse)
        Uncertainty['LPIPS'].append(lpips)
        if r_idx==0:
            image1 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
            image2 = Image.fromarray((predicted_rgb.permute(1, 2, 0)*255).cpu().detach().numpy().astype(np.uint8))
        if r_idx==round(gt_images.shape[0]-1-input_viewpoint_index):
            image3 = Image.fromarray(gt_rgb.cpu().detach().numpy().astype(np.uint8))
            image4 = Image.fromarray((predicted_rgb.permute(1, 2, 0)*255).cpu().detach().numpy().astype(np.uint8))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    width, height = image1.size
    canvas = Image.new("RGB", (2 * width, 2 * height))
    canvas.paste(image1, (0, 0))  # 左上角
    canvas.paste(image2, (width, 0))  # 右上角
    canvas.paste(image3, (0, height))  # 左下角
    canvas.paste(image4, (width, height))  # 右下角

    canvas.save(example_image_path)
    save_img(gt_images[0].detach().cpu(),image_path)
    with open(uncertainty_path, 'w') as f:
        json.dump(Uncertainty, f)

if __name__ == "__main__":
    # parameter settings and print
    args = parse_arguments()
    dataset_name = args.dataset_name
    print("Evaluating on dataset {}".format(dataset_name))
    experiment_path = args.experiment_path
    if args.experiment_path is None:
        print("Will load a model released with the paper.")
    else:
        print("Loading a local model according to the experiment path")

    # set device and random seed
    device = torch.device("cuda:{}".format(args.CUDA_id))
    # device = torch.device("cuda")
    torch.cuda.set_device(device)

    # download model if no model downloaded
    if args.experiment_path is None:
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format(dataset_name))
        if dataset_name in ["gso", "objaverse"]:
            model_name = "latest"
        else:
            model_name = dataset_name
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(model_name))
        
    else:
        cfg_path = os.path.join(experiment_path, ".hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    
    # load cfg
    training_cfg = OmegaConf.load(cfg_path)

    # check that training and testing datasets match if not using official models 
    if args.experiment_path is not None:
        if dataset_name == "gso":
            # GSO model must have been trained on objaverse
            assert training_cfg.data.category == "objaverse", "Model-dataset mismatch"
        else:
            assert training_cfg.data.category == dataset_name, "Model-dataset mismatch"
            
    # load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()
    print('Loaded model!')

    # set background color for 3DGS rendering
    bg_color = [1, 1, 1] if training_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # INPUT
    input_viewpoint_index = args.viewpoint_index
    offset_phi_index = args.offset_phi_index
    model_path = args.model_path
    print(model_path)

    # 计算模型对应的半径
    if dataset_name=='cars':
        radius = 1.3
        scale = 1.0
    else:
        radius, scale = calculate_raidus_scale(NMR_dataset_path, model_path)


    time1 = time.time()
            
    print(f'dealing with viewpoint {input_viewpoint_index} and rotate {offset_phi_index}')
    # function
    offset_phi = offset_phi_index * 0.25 * np.pi
    candidate_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2,radius=radius)
    candidate_viewpoint_pose = candidate_viewpoint_poses[input_viewpoint_index]
    # camera_location = np.array([[1.2185771465301514,0.4454287588596344,0.08162768185138702 ]]) # test
    # candidate_viewpoint_pose = pose_from_xyz(camera_location,camera_location[0]) # tes
    mesh_fpath = os.path.join(model_path,'models/model_normalized.obj')
    output_path = model_path.replace(shapenet_path,output_path)
    metricator = Metricator(device)

    # 创建 image 和 uncertainty 文件夹
    image_dir = os.path.join(output_path, "images")
    uncertainty_dir = os.path.join(output_path, "uncertainties")
    nerf_dir = os.path.join(output_path, "nerf")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    os.makedirs(nerf_dir, exist_ok=True)

    # 如果东西都存在了，那就说明和这个input_viewpoint_index有关的东西都已经处理过了
    image_path = os.path.join(image_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.png")
    example_image_path = os.path.join(image_dir, f"viewpoint_example_{input_viewpoint_index}_offset_phi_{offset_phi_index}.png")
    uncertainty_path = os.path.join(uncertainty_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.json")
    gt_images_path = os.path.join(image_dir, f"gt_images/{input_viewpoint_index}_offset_phi_{offset_phi_index}")
    gif_path = os.path.join(image_dir, f"viewpoint_{input_viewpoint_index}_offset_phi_{offset_phi_index}.gif")
    if os.path.exists(image_path) and os.path.exists(example_image_path) and os.path.exists(uncertainty_path):
        print(f'viewpoint {input_viewpoint_index} and rotate {offset_phi} already finished so skip')
        sys.exit()
        

    # gt enviroment
    obj_location = np.zeros((1,3))
    rot_mat = np.eye(3)
    hom_coords = np.array([[0., 0., 0., 1.]]).reshape(1, 4)
    obj_pose = np.concatenate((rot_mat, obj_location.reshape(3,1)), axis=-1)
    obj_pose = np.concatenate((obj_pose, hom_coords), axis=0)
    renderer = blender_interface.BlenderInterface(resolution=training_cfg['data']['training_resolution'])
    renderer.import_mesh(mesh_fpath, scale=scale, object_world_matrix=obj_pose)

    # relative viewpoint poses
    absolute_viewpoint_poses = generate_HEALPix_viewpoints(n_side=2,radius=radius,original_viewpoint=np.array(candidate_viewpoint_pose[4:]),offset_phi=offset_phi)
    gt_images = generate_gt_images(renderer, absolute_viewpoint_poses, gt_images_path).to(device)
    focals_pixels_pred = None

    # load camera infos   
    images_and_camera_poses = load_camera_info(training_cfg, gt_images_path)
    rot_transform_quats = images_and_camera_poses["source_cv2wT_quat"][0]

    reconstruction = model((gt_images[0]/255).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
                            images_and_camera_poses["view_to_world_transforms"][0, ...].unsqueeze(0).unsqueeze(0).to(device),
                            rot_transform_quats.unsqueeze(0).unsqueeze(0).to(device),
                            focals_pixels_pred)

    uncertainty_map_generation(device, training_cfg, background, input_viewpoint_index, metricator, image_path, example_image_path, uncertainty_path, gif_path, gt_images, images_and_camera_poses, reconstruction)
    
    time2 = time.time()
    elapsed_time=time2-time1
    print(f"代码运行时间: {elapsed_time:.6f} 秒")