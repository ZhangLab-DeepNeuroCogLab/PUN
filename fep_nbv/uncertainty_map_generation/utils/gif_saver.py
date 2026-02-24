import os
from PIL import Image

def create_gif_from_pngs(input_folder, output_gif, duration=100, loop=0):
    """
    从文件夹中按顺序读取 PNG 图片并生成 GIF

    :param input_folder: 包含 PNG 文件的文件夹路径
    :param output_gif: 生成的 GIF 文件路径
    :param duration: 每帧持续时间（毫秒）
    :param loop: GIF 循环次数，0 表示无限循环
    """
    # 获取所有 PNG 文件，并按文件名排序
    png_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    
    if not png_files:
        print("❌ 没有找到 PNG 文件！")
        return
    
    # 读取所有图片
    images = [Image.open(os.path.join(input_folder, f)) for f in png_files]
    
    # 保存为 GIF
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=loop)
    print(f"✅ GIF 生成成功: {output_gif}")

# 使用示例
# input_folder = "/home/zhengquan/06-splatter-image/out/1857_02691156_1a04e3eab45ca15dd86060f189eb133"  # 替换为你的 PNG 文件夹路径
# output_gif = os.path.join(input_folder,"output.gif")  # 生成的 GIF 文件路径
# create_gif_from_pngs(input_folder, output_gif, duration=100, loop=0)