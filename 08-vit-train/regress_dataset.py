import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader, random_split
from regress_model import ViTRegressor
import timm
import os
from glob import glob
import json

test_class = ['02958343','03001627']#['car','chair']

class RegressionDataset_final(Dataset):
    def __init__(self, root_dir, transform=None, mode='PSNR', split='train',skip=1,train_num=90,test_class=test_class):
        self.samples = []
        self.transform = transform
        self.uncertainty_mode = mode
        self.skip = skip

        # 遍历每个类别
        print(test_class)
        for class_folder in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            model_folders = sorted(os.listdir(class_path))
            if split == 'train':
                if class_folder in test_class:
                    continue
                else:
                    selected_models = model_folders[:train_num]
            elif split == 'test_all':
                if class_folder in test_class:
                    selected_models = model_folders[:100]
                else:
                    selected_models = model_folders[90:100]
            elif split == 'test_in_class':
                if class_folder in test_class:
                    continue
                else:
                    selected_models = model_folders[90:100]
            elif split == 'test_out_class':
                if class_folder in test_class:
                    selected_models = model_folders[:100]
                else:
                    continue
            else:
                raise ValueError("split must be 'train' or 'test' test_in_class test_out_class")

            for model_folder in selected_models:
                model_path = os.path.join(class_path, model_folder)
                img_dir = os.path.join(model_path, 'images')
                label_dir = os.path.join(model_path, 'uncertainties')

                if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                    continue

                for img_path in glob(os.path.join(img_dir, "*.png")):
                    if img_path.split('offset_phi_')[-1].split('.png')[0]!='0':
                        continue
                    base_name = os.path.basename(img_path).replace(".png", "_complexity.json")
                    label_path = os.path.join(label_dir, base_name)
                    
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            label = json.load(f)
                        if "complexity" not in label.keys():
                            continue
                        self.samples.append((img_path, label_path))
        self.samples = self.samples[::skip]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.uncertainty_mode == 'complexity':
            with open(label_path, 'r') as f:
                label = json.load(f)
            if self.transform:
                image = self.transform(image)
            return image, label
        else:
            with open(label_path, 'r') as f:
                label = json.load(f)[self.uncertainty_mode]
            label = torch.tensor(label, dtype=torch.float32)
            if self.uncertainty_mode == 'MSE':
                label = label * 1000
            if self.transform:
                image = self.transform(image)
            return image, label


class RegressionDataset_singleclass(Dataset):
    def __init__(self, root_dir, transform=None, mode='PSNR', split='train'):
        self.samples = []
        self.transform = transform
        self.uncertainty_mode = mode

        # 遍历每个类别
        for class_folder in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            model_folders = sorted(os.listdir(class_path))
            if split == 'train':
                selected_models = model_folders[:90]
            elif split == 'test':
                selected_models = model_folders[90:100]
            else:
                raise ValueError("split must be 'train' or 'test'")

            for model_folder in selected_models:
                model_path = os.path.join(class_path, model_folder)
                img_dir = os.path.join(model_path, 'images')
                label_dir = os.path.join(model_path, 'uncertainties')

                if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                    continue

                for img_path in glob(os.path.join(img_dir, "*.png")):
                    if img_path.split('offset_phi_')[-1].split('.png')[0]!='0':
                        continue
                    base_name = os.path.basename(img_path).replace(".png", ".json")
                    label_path = os.path.join(label_dir, base_name)
                    if os.path.exists(label_path):
                        self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            label = json.load(f)[self.uncertainty_mode]
        label = torch.tensor(label, dtype=torch.float32)
        if self.uncertainty_mode == 'MSE':
            label = label * 1000
        if self.transform:
            image = self.transform(image)
        return image, label

class RegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='PSNR'):
        self.samples = []
        self.transform = transform
        self.uncertainty_mode = mode

        # 遍历 class/model_index/ 目录结构
        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path): continue
            for model_folder in os.listdir(class_path):
                model_path = os.path.join(class_path, model_folder)
                img_dir = os.path.join(model_path, 'images')
                label_dir = os.path.join(model_path, 'uncertainties')

                if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
                    continue

                for img_path in glob(os.path.join(img_dir, "*.png")):
                    base_name = os.path.basename(img_path).replace(".png", ".json")
                    label_path = os.path.join(label_dir, base_name)
                    if os.path.exists(label_path):
                        self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            label = json.load(f)
            label = label[self.uncertainty_mode]
        label = torch.tensor(label, dtype=torch.float32)
        if self.uncertainty_mode == 'MSE':
            label = label*255

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__=='__main__':
    # load transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRegressor(output_dim=48).to(device)
    data_cfg = timm.data.resolve_data_config(model.backbone.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)

    
    # dataset = RegressionDataset(
    #     root_dir="/attached/remote-home2/zzq/data/splat-image-distribution-dataset",  # 替换为你的路径
    #     transform=transform
    # )
    train_dataset = RegressionDataset_singleclass(
        root_dir='/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-car',  # 替换为你的路径
        transform=transform,
        mode='PSNR',
        split='train'
    )
    test_dataset = RegressionDataset_singleclass(
        root_dir='/mnt/hdd/zhengquan/single_datasets/splat-image-distribution-dataset-car',  # 替换为你的路径
        transform=transform,
        mode='PSNR',
        split='test'
    )
    # **拆分训练/验证集**
    train_size = int(0.8 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    train_dataset, val_dataset= random_split(train_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f'train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}')
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试输出
    for images, labels in train_dataloader:
        print("Images shape:", images.shape)     # [B, 3, 224, 224]
        print("Labels shape:", labels.shape)     # [B, 48]
        print("Label sample:", labels[0][:5])    # 打印部分标签内容
        # plt.imshow(images[0].permute(1, 2, 0))   # 可视化第一张图
        # plt.title("Sample Image")
        # plt.axis("off")
        
        break