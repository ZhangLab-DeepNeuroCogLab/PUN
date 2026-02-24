import os
import json
import random
from pathlib import Path
from typing import List, Dict
root_path = '/home/zhengquan/06-splatter-image'
shapenet_path = "/mnt/hdd/zhengquan/Shapenet/ShapeNetCore.v2"
distribution_dataset_path = '/mnt/hdd/zhengquan/Splat-image-distribution-dataset'

import sys
sys.path.append(root_path)

from fep_nbv.uncertainty_generation.uncertainty_utils import offset2word,word2offset

def update_status_json(distribution_dataset_path, output_file,num_viewpoints=48,num_rotations=8):
    all_update_rotation = 0 
    # 读取现有的 model_status.json，如果不存在，则初始化为空
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            model_status = json.load(f)
    else:
        generate_model_status_json(shapenet_path, percentage=percentage/100,output_file=output_file)
        with open(output_file, "r") as f:
            model_status = json.load(f)
        print('no corresponding json file ')
        return
    # 遍历 distribution_dataset_path 目录
    for category in os.listdir(distribution_dataset_path):
        offset = category
        category_path = os.path.join(distribution_dataset_path, category)
        if not os.path.isdir(category_path):
            continue  # 跳过非目录项

        for model in os.listdir(category_path):
            model_path = os.path.join(category_path, model)
            if not os.path.isdir(model_path):
                continue  # 只处理文件夹

            model_key = os.path.join(offset, model)

            # 确保 model_key 在 model_status 中
            if model_key not in model_status:
                model_status[model_key] = {
                    "finished": [[False for _ in range(num_rotations)] for _ in range(num_viewpoints)]
                }

            # 遍历 viewpoint 和 rotation
            image_dir = os.path.join(model_path, "images")
            if not os.path.exists(image_dir):
                continue  # 如果 images 文件夹不存在，跳过

            for filename in os.listdir(image_dir):
                if filename.startswith("viewpoint_") and filename.endswith(".png"):
                    parts = filename.split("_")
                    if len(parts) < 5:
                        continue  # 确保文件命名格式正确

                    try:
                        viewpoint = int(parts[1])
                        rotation = int(parts[4].split(".")[0])
                        if 0 <= viewpoint < num_viewpoints and 0 <= rotation < num_rotations:
                            if model_status[model_key]["finished"][viewpoint][rotation]==False:
                                print(f'{model_key} {viewpoint} {rotation} already generate final image, set to True')
                                all_update_rotation+=1
                            model_status[model_key]["finished"][viewpoint][rotation] = True
                    except ValueError:
                        continue  # 如果解析失败，则跳过该文件

    # 保存更新后的 JSON 状态文件
    with open(output_file, "w") as f:
        json.dump(model_status, f, indent=4)
    
    print(f'{all_update_rotation} rotations updated to True')

    print(f"Model status successfully updated and saved to {output_file}")

def generate_model_status_json(shapenet_dir=root_path, percentage=0.3, output_file='data/test/model_status.json',num_viewpoints=48,num_rotations=8):
    """
    生成 ShapeNet 模型的状态文件，同时打印每个模型的统计信息。

    :param shapenet_dir: ShapeNet 数据集根目录
    :param percentage: 选择每个类别的模型的比例
    :param output_file: 输出 JSON 文件路径
    """
    model_status = {}
    total_model_number = 0

    for category in os.listdir(shapenet_dir):
        category_path = os.path.join(shapenet_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 获取所有模型列表
        models = [model for model in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, model))]
        models = sorted(models)

        # 按比例选择模型
        num_selected_models = max(1, int(len(models) * percentage))
        selected_models = models[:num_selected_models]

        # 初始化每个模型的 viewpoint 和 rotation 状态
        for model in selected_models:
            model_key = os.path.join(category, model)
            model_status[model_key] = {
                "finished": [[False for _ in range(num_rotations)] for _ in range(num_viewpoints)]
            }

            # 打印模型统计信息
        print(f"类别: {category}, 模型数量: {len(selected_models)}, 总模型数量: {len(models)}, 比例: {len(selected_models)/len(models):2f}")
        total_model_number += len(selected_models)
    
    print(f'\n总模型数量: {total_model_number}')
    # 将结果保存到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(model_status, f, indent=4)

    print(f"\nJSON 文件已保存到: {output_file}")


def read_model_status(json_file):
    """
    读取 ShapeNet 模型状态 JSON 文件，并打印模型的统计信息。

    :param json_file: JSON 文件路径
    """
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: JSON 文件 {json_file} 不存在！")
        return

    # 读取 JSON 文件
    with open(json_file, "r") as f:
        model_status = json.load(f)
    num_finished = 0

    # 遍历并输出每个模型的状态信息
    for model_key, status in model_status.items():
        num_finished += sum(
            sum(rotation_finished for rotation_finished in viewpoint_finished)
            for viewpoint_finished in status["finished"]
        )

        # 输出每个模型的统计信息
        # print(f"模型: {model_key}, 已完成任务: {num_finished}/{total_tasks}")
    print(f'model num: {len(model_status.keys())},tasks:{num_finished}/{48*8*len(model_status.keys())}')

    print("\n读取完成")
    return model_status

def all_tasks_finished(model_status, model):
    """检查某个模型是否所有任务都已完成"""
    return all(all(view) for view in model_status[model]["finished"])

def check_and_update_status(model, viewpoint, rotation, model_status, distribution_dataset_path):
    """检查图片是否已存在，存在则更新状态为 True"""
    category = model.split('/')[0]
    temp_model = os.path.join(category,model.split('/')[1])
    image_path = os.path.join(distribution_dataset_path, temp_model, f'images/viewpoint_{viewpoint}_offset_phi_{rotation}.png')
    if os.path.exists(image_path):
        print(f'{model} final image exist, so set True')
        model_status[model]["finished"][viewpoint][rotation] = True
        return True
    return False

def save_status_to_file(model_status, output_file):
    current_model_status = read_model_status(output_file)
    current_model_status.update(model_status)
    model_status.update(current_model_status)
    """保存状态到 JSON 文件"""
    with open(output_file, 'w') as f:
        json.dump(current_model_status, f, indent=4)

# 示例调用
if __name__ == "__main__":
    shapenet_root = shapenet_path  # 修改为你的 ShapeNet 路径
    percentage = 20
    output_file = f'data/test/model_status_{percentage}.json'  
    output_file = '/attached/data/remote-home2/zzq/05-offline-NeRF-based-NBV/json_files/model_status_20.json'   # 输出文件路径
    output_file = os.path.join(root_path,f'json_files/model_status_{percentage}.json')
    # generate_model_status_json(shapenet_root, percentage=percentage/100,output_file=output_file)

    # model_status = read_model_status(output_file)
    # print(len(model_status.keys()))
    # print(model_status.keys())

    update_status_json(distribution_dataset_path,output_file)