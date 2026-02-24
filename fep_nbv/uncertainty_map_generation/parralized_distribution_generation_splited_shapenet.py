import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import time
from multiprocessing import Manager, Lock
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(BASE_DIR, "../../")
shapenet_path = os.path.join(BASE_DIR, "../../../../shapenet/ShapeNetCore.v2") # absolute path
output_path = os.path.join(BASE_DIR, "../../data/shapenet/instance_example")

# if not os.path.exists(root_path):
#     root_path=root_path.replace('/attached/data','/attached')
#     shapenet_path=shapenet_path.replace('/attached/data','/attached')
#     output_path=output_path.replace('/attached/data','/attached')
# 读取服务器配置信息
config_path = os.path.join(root_path, 'server_config.json')
with open(config_path, 'r') as f:
    server_config = json.load(f)
# 获取服务器 ID 和总服务器数
SERVER_ID = server_config["server_id"]  # 当前服务器编号
TOTAL_SERVERS = server_config["total_servers"]  # 服务器总数
MAX_WORKERS = server_config["max_workers"]  # 最大并行任务数
included_categories = ['bench']

import sys
sys.path.append(root_path)

from fep_nbv.uncertainty_map_generation.shapenet_split_test import *
from fep_nbv.uncertainty_map_generation.uncertainty_utils import offset2word


# def get_free_gpu_memory():
#     """获取每张 GPU 的可用显存（单位：MB）"""
#     result = subprocess.run(
#         ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
#         stdout=subprocess.PIPE
#     )
#     free_memory = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
#     return free_memory


def get_free_gpu_memory(duration=10):
    """获取每张 GPU 在过去 5 分钟内的最大显存占用（单位：MB）"""
    # duration = 10  # 5 分钟
    interval = 1       # 每 5 秒记录一次
    recorded_memory = []

    for _ in range(duration // interval):
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True
        )
        memory_usage = [int(x) for x in result.stdout.strip().split("\n")]
        recorded_memory.append(memory_usage)
        time.sleep(interval)

    # 计算每张 GPU 在过去 5 分钟内的最大显存占用
    max_usage = [max(gpu) for gpu in zip(*recorded_memory)]

    # 获取总显存
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True
    )
    total_memory = [int(x) for x in result.stdout.strip().split("\n")]

    # 计算过去 5 分钟内的最小可用显存
    free_memory = [t - u for t, u in zip(total_memory, max_usage)]
    
    # print(f"过去 5 分钟内每张 GPU 的最小可用显存（MiB）: {free_memory}")
    return free_memory

def run_python_script(task_id, script_path, model_path, viewpoint, rotation, gpu_id, running_tasks, task_lock):
    """
    运行一个 Python 脚本，传递参数。
    
    :param task_id: 任务 ID
    :param script_path: 要运行的 Python 脚本路径
    :param model_path: 模型路径参数
    :param viewpoint: 视角参数
    :param rotation: 旋转参数
    :return: 任务结果
    """
    with task_lock:
        running_tasks.value += 1
        print(f"任务 {task_id} 开始，使用 GPU {gpu_id}，当前运行任务数: {running_tasks.value}")

    # 构造命令
    command = [
        "python", script_path,
        '--dataset_name',
        'nmr',
        '--model_path',
        model_path,
        '--viewpoint_index',
        str(viewpoint),
        '--offset_phi_index',
        str(rotation),
        '--CUDA_id',
        str(gpu_id)
    ]
    # command =['echo aaaaa'] 

    # 运行子进程
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 使用 Popen 进行非阻塞子进程调用
    # result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # stdout, stderr = result.communicate()

    with task_lock:
        running_tasks.value -= 1

    if result.returncode == 0:
        print(f"任务 {task_id} 完成，当前运行任务数: {running_tasks.value}")
        return result.stdout.strip()
    else:
        print(f"任务 {task_id} 失败: {result.stderr.strip()}，当前运行任务数: {running_tasks.value}")
        return None

def log_failed_task(task_id, model_path, viewpoint, rotation):
    """将失败的任务参数记录到文件"""
    with open(root_path+"/nohup_log/failed_tasks.log", "a") as f:
        f.write(f"任务 {task_id}: model_path={model_path}, viewpoint={viewpoint}, rotation={rotation}\n")


if __name__ == "__main__":
    percentage = 100
    output_file = os.path.join(root_path,f'json_files/model_status_{percentage}_bench.json')     # 输出文件路径
    if not os.path.exists(output_file):
        generate_model_status_json(shapenet_path, percentage=percentage/100,output_file=output_file)
        update_status_json(output_path,output_file)
    model_status = read_model_status(output_file)
    manager = Manager()
    running_tasks = manager.Value('i', 0)
    task_lock = manager.Lock()

    # 获取每张 GPU 的可用显存
    free_memory = get_free_gpu_memory()
    # free_memory = [24000]
    task_memory_usage = 4000

    script_path = os.path.join(root_path,'fep_nbv/uncertainty_map_generation/single_rotation_generation.py')  # 子任务 Python 脚本路径
    viewpoints = [i for i in range(48)]                          # 总任务数
    rotations = [i for i in range(1)]
    max_workers = 6                            # 并行任务数
    # model_path = os.path.join(shapenet_path,'02691156/1a32f10b20170883663e90eaf6b4ca52')

    # 计算每张 GPU 可并行任务数
    max_tasks_per_gpu = [free // task_memory_usage for free in free_memory]
    total_max_workers = sum(max_tasks_per_gpu)
    total_max_workers = min(total_max_workers, MAX_WORKERS)  # 限制最大并行任务数)
    # total_max_workers = 5

    print(f"每张 GPU 可运行任务数: {max_tasks_per_gpu}")
    print(f"总并行任务数: {total_max_workers}")

    gpu_task_count = manager.dict({i: 0 for i in range(len(get_free_gpu_memory()))})

    # **对模型列表进行排序并切片**
    all_models = sorted([model for model in model_status.keys() if offset2word(model.split('/')[0]) in included_categories])  # 排序模型
    models_per_server = len(all_models) // TOTAL_SERVERS
    start_idx = (SERVER_ID - 1) * models_per_server
    end_idx = start_idx + models_per_server if SERVER_ID < TOTAL_SERVERS else len(all_models)
    models_to_process = all_models[start_idx:end_idx]  # 取当前服务器的部分
    

    print(f"服务器 {SERVER_ID}: 处理 {start_idx} 到 {end_idx} 之间的模型（共 {len(models_to_process)} 个）")

    category_count = 0
    with ProcessPoolExecutor(max_workers=total_max_workers) as executor:
        futures = []

        for model in models_to_process:
            if all_tasks_finished(model_status, model):
                category_count += 1
                print(f"模型 {model} 的所有任务已完成，跳过...")
                continue
            category = offset2word(model.split('/')[0])
            if category_count>=100:
                print(f'{category} dataset more than 100')
                continue
            if category not in included_categories:
                continue
            category_count += 1
            model_path = os.path.join(shapenet_path,model)

            time1 = time.time()
            for viewpoint in viewpoints:
                for rotation in rotations:
                    task_id = rotation+viewpoint*8
                    if check_and_update_status(model, viewpoint, rotation, model_status, output_path):
                        print(f'{model} {viewpoint} {rotation} png existed so skip')
                        continue
                    # **循环等待有可用的 GPU**
                    while True:
                        with task_lock:
                            if running_tasks.value < total_max_workers:
                                break
                        print("任务过多，等待 1 分钟...")
                        time.sleep(60)
                    while True:
                        free_memory = get_free_gpu_memory()
                        available_gpus = [
                            i for i, mem in enumerate(free_memory)
                            if mem > task_memory_usage and gpu_task_count[i] < max_tasks_per_gpu[i]
                        ]

                        if available_gpus:
                            break  # 找到符合条件的 GPU，退出循环
                        else:
                            print("所有 GPU 都没有可用资源，等待 1 分钟后重试...")
                            time.sleep(60)  # 等待 5 分钟后重试
                    # **随机选择一个符合条件的 GPU**
                    available_gpus_tasks = [max_tasks_per_gpu[i]-gpu_task_count[i] for i in available_gpus]
                    max_index = max(range(len(available_gpus_tasks)), key=lambda i: available_gpus_tasks[i])
                    gpu_id = 0
                    gpu_task_count[gpu_id] += 1  # 增加任务计数

                    # 提交任务
                    future = executor.submit(run_python_script, task_id, script_path, model_path, viewpoint, rotation, gpu_id, running_tasks, task_lock)

                    def task_done_callback(fut, model=model, viewpoint=viewpoint, rotation=rotation, gpu_id=gpu_id):
                        gpu_task_count[gpu_id] -= 1
                        if fut.result():
                            model_status[model]["finished"][viewpoint][rotation] = True
                            print(f'{model} {viewpoint} {rotation} finished, set status to True')
                            # save_status_to_file(model_status, output_file)

                    future.add_done_callback(task_done_callback)
                    futures.append(future)
            save_status_to_file(model_status, output_file)
            time2 = time.time()
            elapsed_time=time2-time1
            print(f"{model_path} 代码运行时间: {elapsed_time:.6f} 秒")

            # 处理任务结果
            for future in as_completed(futures):
                try:
                    output = future.result()
                    if output:
                        print(f"任务输出：\n{output}")
                except Exception as e:
                    print(f"任务执行异常: {e}")
                    log_failed_task(task_id, model_path, viewpoint, rotation)
