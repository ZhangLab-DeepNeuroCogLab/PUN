import os
from PIL import Image
import torch
import time
import numpy as np
import lpips as lpips_lib
import sys
from nltk.corpus import wordnet as wn
sys.path.append('/home/zhengquan/06-splatter-image')
from utils.loss_utils import ssim as ssim_fn
from utils.general_utils import matrix_to_quaternion

def generate_gt_images(renderer, absolute_viewpoint_poses, gt_images_path):
    if os.path.exists(gt_images_path):
        existing_images = [f for f in os.listdir(gt_images_path) if f.endswith('_gt.png')]
        if len(existing_images) == len(absolute_viewpoint_poses):
            print(f"Found {len(existing_images)} GT images, loading directly from {gt_images_path}")
            # 读取已存在的图片
            gt_images = []
            for index in range(len(absolute_viewpoint_poses)):
                image_path = os.path.join(gt_images_path, f'{index}_gt.png')
                img = Image.open(image_path)
                img = torch.FloatTensor(np.array(img)[...,:3])  # 保持与生成流程一致
                gt_images.append(img)
            # 将读取的图片转换为 Tensor
            gt_images = torch.stack(gt_images)
            print(f"Loaded {len(gt_images)} GT images successfully!")
            return gt_images

    t1 = time.time()
    gt_images = renderer.render(gt_images_path, absolute_viewpoint_poses, write_cam_params=True)
    t2 = time.time()
    print(f'generating gt images, time used: {t2-t1:2f} seconds')
    gt_images = torch.stack([torch.FloatTensor(iii)[...,:3] for iii in gt_images])
    print('saving groundtruth images')
    if not os.path.exists(gt_images_path):
        os.makedirs(gt_images_path,exist_ok=True)
    print(gt_images.shape)
    for index,gt_image in enumerate(gt_images):
        image = Image.fromarray((gt_image*255).cpu().detach().numpy().astype(np.uint8))
        image.save(os.path.join(gt_images_path,f'{index}_gt.png'))
    
    return gt_images*255

def make_poses_relative_to_first(images_and_camera_poses):
    inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
    for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
        images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                            inverse_first_camera.unsqueeze(0),
                                            images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
        images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                            images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                            inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
        images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                            inverse_first_camera.unsqueeze(0),
                                            images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
        images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
    return images_and_camera_poses


class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        mse = torch.mean((image - target) ** 2, dim=[0, 1, 2]).item()
        return psnr, ssim, lpips,mse

def save_img(img, filepath):
    '''
    保存图片
    - input
        img:    numpy.array
        filepat: str
    - output
        no
    '''
    path = os.path.dirname(filepath)
    if not os.path.exists(path):
        os.makedirs(path)
    if img.max()>1:
        img = Image.fromarray(np.uint8(img))
    else:
        img = Image.fromarray(np.uint8(img*255))
    
    # print(filepath)
    img.save(filepath)
    img.close()

def get_source_cw2wT(source_cameras_view_to_world):
    # Compute view to world transforms in quaternion representation.
    # Used for transforming predicted rotations
    qs = []
    for c_idx in range(source_cameras_view_to_world.shape[0]):
        qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
    return torch.stack(qs, dim=0)

def offset2word(offset):
    if isinstance(offset,str):
        offset = int(offset)

    pos = 'n' 
    synset = wn.synset_from_pos_and_offset(pos, offset)
    words = synset.lemma_names()

    return words[0]

def word2offset(word):
    synsets = wn.synsets(word)
    if not synsets:
        return f"No synset found for {word}"
    offset = str(synsets[0].offset()).zfill(8)
    return offset