import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2

# 配置参数 - 必须与训练参数一致
IMG_WIDTH = 256
IMG_HEIGHT = 256
MODEL_PATH = 'pix2pix_generator_final.h5'  # 训练后保存的模型路径
DEFAULT_OUTPUT_DIR = './output_images'    # 默认输出目录
OVERLAP = 64  # 增加重叠区域大小（像素）以减少重影
MIN_SIZE = 1024  # 最小尺寸阈值
BLEND_MODE = 'gaussian'  # 融合模式：'linear' 或 'gaussian'

def load_image_no_resize(image_path):
    """加载图像但不调整大小"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def preprocess_image(image):
    """预处理图像（调整大小和标准化）"""
    # 调整大小
    image_resized = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # 标准化到 [-1, 1] 范围
    image_normalized = (image_resized - 0.5) * 2
    return image_normalized

def create_weight_map(height, width, overlap=64, mode='gaussian'):
    """创建权重图用于融合重叠区域"""
    # 创建水平方向权重
    x = np.ones(width)
    if mode == 'linear':
        x[:overlap] = np.linspace(0, 1, overlap)
        x[-overlap:] = np.linspace(1, 0, overlap)
    elif mode == 'gaussian':
        # 高斯权重提供更平滑的过渡
        gaussian = np.exp(-np.linspace(-3, 3, overlap*2)**2)
        x[:overlap] = gaussian[:overlap]
        x[-overlap:] = gaussian[overlap:]
    
    # 创建垂直方向权重
    y = np.ones(height)
    if mode == 'linear':
        y[:overlap] = np.linspace(0, 1, overlap)
        y[-overlap:] = np.linspace(1, 0, overlap)
    elif mode == 'gaussian':
        y[:overlap] = gaussian[:overlap]
        y[-overlap:] = gaussian[overlap:]
    
    # 创建二维权重图
    weight_map = np.outer(y, x)
    
    # 添加通道维度
    weight_map = np.expand_dims(weight_map, axis=-1)
    
    return weight_map

def pad_image(image, target_height, target_width, mode='reflect'):
    """填充图像到指定尺寸（使用反射填充）"""
    # 计算需要填充的高度和宽度
    pad_h = max(target_height - tf.shape(image)[0], 0)
    pad_w = max(target_width - tf.shape(image)[1], 0)
    
    # 填充图像（使用反射填充）
    padded = tf.pad(image, 
                    [[0, pad_h], [0, pad_w], [0, 0]], 
                    mode=mode)  # 反射填充
    
    return padded, pad_h, pad_w

def split_image(image, tile_size=256, overlap=64):
    """将大图像分割成多个重叠块"""
    # 获取图像尺寸
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    
    # 计算有效步长
    stride = tile_size - overlap
    
    # 计算需要的行列数
    num_rows = int(np.ceil((height - overlap) / stride))
    num_cols = int(np.ceil((width - overlap) / stride))
    
    tiles = []
    positions = []
    
    # 分割图像
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前块的起始位置（考虑重叠）
            start_h = max(0, min(row * stride, height - tile_size))
            start_w = max(0, min(col * stride, width - tile_size))
            end_h = start_h + tile_size
            end_w = start_w + tile_size
            
            # 确保不超过边界
            end_h = min(end_h, height)
            end_w = min(end_w, width)
            
            # 提取当前块
            tile = image[start_h:end_h, start_w:end_w, :]
            
            # 如果需要，填充当前块（使用反射填充）
            tile_height, tile_width = tf.shape(tile)[0], tf.shape(tile)[1]
            if tile_height < tile_size or tile_width < tile_size:
                tile, pad_h, pad_w = pad_image(tile, tile_size, tile_size, mode='REFLECT')
            else:
                pad_h, pad_w = 0, 0
                
            tiles.append(tile)
            positions.append((start_h, start_w, end_h - start_h, end_w - start_w, pad_h, pad_w))
    
    return tiles, positions, height, width

def resize_to_min_size(image, min_size=1024):
    """将图像调整到至少一边为min_size，保持长宽比"""
    # 获取当前尺寸（转换为浮点数）
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    
    # 如果两边都大于min_size，直接返回
    if height >= min_size and width >= min_size:
        return image
    
    # 计算缩放比例
    scale = min_size / min(height, width)
    
    # 计算新尺寸（转换为整数）
    new_height = tf.cast(tf.math.round(height * scale), tf.int32)
    new_width = tf.cast(tf.math.round(width * scale), tf.int32)
    
    # 调整大小（使用双线性插值）
    resized_image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    
    return resized_image

def postprocess_image(image):
    """后处理图像以减少重影"""
    # 应用高斯模糊减少锐利边缘
    image_np = image.numpy() if isinstance(image, tf.Tensor) else image
    blurred = cv2.GaussianBlur(image_np, (3, 3), 0)
    
    # 混合原始图像和模糊图像
    alpha = 0.7  # 原始图像权重
    processed = alpha * image_np + (1 - alpha) * blurred
    
    return np.clip(processed, 0, 1)

def generate_image(model, input_image_path, output_dir=None):
    """
    使用模型生成图像（支持大图像，带重叠融合）
    """
    # 创建输出目录（如果不存在）
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像（不调整大小）
    raw_image = load_image_no_resize(input_image_path)
    
    # 获取原始尺寸
    orig_height, orig_width = tf.shape(raw_image)[0], tf.shape(raw_image)[1]
    
    # 调整到最小尺寸
    resized_image = resize_to_min_size(raw_image, MIN_SIZE)
    image_height, image_width = tf.shape(resized_image)[0], tf.shape(resized_image)[1]
    
    # 创建输出文件名
    timestamp = int(time.time())
    input_filename = os.path.basename(input_image_path)
    output_filename = f"generated_{timestamp}_{input_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 打印尺寸信息
    print(f"原始尺寸: {orig_height}x{orig_width}, 调整后尺寸: {image_height}x{image_width}")
    
    # 如果图像小于或等于目标尺寸，直接处理
    if image_height <= IMG_HEIGHT and image_width <= IMG_WIDTH:
        # 预处理图像
        input_image = preprocess_image(resized_image)
        # 添加batch维度
        input_image_batch = tf.expand_dims(input_image, 0)
        # 生成预测图像
        prediction = model(input_image_batch, training=False)
        # 将输出从[-1, 1]转换回[0, 1]
        prediction_image = prediction[0] * 0.5 + 0.5
        # 应用后处理减少重影
        processed_image = postprocess_image(prediction_image)
        # 保存图像
        plt.imsave(output_path, processed_image)
        print(f"输出图像已保存至: {output_path}")
    else:
        # 打印图像尺寸信息
        print(f"图像尺寸 ({image_height}x{image_width}>256x256)，将分割成小块处理（重叠{OVERLAP}像素）")
        
        # 分割图像（带重叠）
        tiles, positions, _, _ = split_image(resized_image, IMG_HEIGHT, OVERLAP)
        num_tiles = len(tiles)
        
        # 创建用于最终拼接的画布（使用黑色背景）和权重累加器
        full_output = np.zeros((image_height, image_width, 3), dtype=np.float32)
        weight_accumulator = np.zeros((image_height, image_width, 1), dtype=np.float32)
        
        # 创建权重图（用于融合重叠区域）
        weight_map = create_weight_map(IMG_HEIGHT, IMG_WIDTH, OVERLAP, BLEND_MODE)
        
        print(f"共分割成 {num_tiles} 个重叠块，正在处理中...")
        
        # 处理每个图像块
        for i, (tile, pos) in enumerate(zip(tiles, positions)):
            start_h, start_w, real_height, real_width, pad_h, pad_w = pos
            
            print(f"处理小块 #{i+1}/{num_tiles} (位置: {start_h},{start_w} 尺寸: {real_height}x{real_width})...")
            
            # 预处理当前块
            preprocessed_tile = preprocess_image(tile)
            
            # 添加batch维度
            tile_batch = tf.expand_dims(preprocessed_tile, 0)
            
            # 生成预测图像
            prediction = model(tile_batch, training=False)
            
            # 将输出从[-1, 1]转换回[0, 1]
            prediction_image = prediction[0] * 0.5 + 0.5
            
            # 移除填充部分（如果有）
            unpadded_tile = prediction_image[:real_height, :real_width, :]
            
            # 应用后处理减少重影
            processed_tile = postprocess_image(unpadded_tile)
            
            # 计算实际结束位置
            end_h = start_h + real_height
            end_w = start_w + real_width
            
            # 裁剪权重图以匹配实际尺寸
            current_weight_map = weight_map[:real_height, :real_width, :]
            
            # 将当前块的权重应用到权重累加器
            weight_accumulator[start_h:end_h, start_w:end_w, :] += current_weight_map
            
            # 将处理后的块（带权重）添加到最终图像
            full_output[start_h:end_h, start_w:end_w, :] += processed_tile * current_weight_map
        
        print("所有块处理完成，正在进行融合...")
        
        # 避免除以零（将权重累加器中的零替换为1）
        zero_mask = weight_accumulator == 0
        weight_accumulator[zero_mask] = 1.0
        
        # 加权融合
        full_output = full_output / weight_accumulator
        
        # 应用全局后处理
        final_output = postprocess_image(full_output)
        
        # 保存完整图像
        plt.imsave(output_path, np.clip(final_output, 0, 1))
        print(f"拼接后的输出图像已保存至: {output_path}")
    
    return output_path

def main():
    # 尝试加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        print("请确保模型文件存在或运行训练脚本")
        return
    
    try:
        print(f"加载模型: {MODEL_PATH}")
        generator = load_model(MODEL_PATH, compile=False)
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return
    
    # 获取输入路径
    input_path = input("\n请输入要转换的图像路径 (或输入'q'退出):\n> ")
    
    while input_path.lower() != 'q':
        if not os.path.exists(input_path):
            print(f"错误: 文件 '{input_path}' 不存在")
        else:
            print("\n正在处理图像...")
            try:
                output_path = generate_image(generator, input_path)
                print(f"\n 转换完成! 输出路径: {output_path}\n")
                
                # 可选项: 显示生成的图像
                display_image = input("是否要显示生成的图像? (y/n): ").lower()
                if display_image == 'y':
                    output_image = plt.imread(output_path)
                    plt.figure(figsize=(10, 10))
                    plt.title(f"生成的图像: {os.path.basename(output_path)}")
                    plt.imshow(output_image)
                    plt.axis('off')
                    plt.show()
            except Exception as e:
                print(f"处理图像时发生错误: {str(e)}")
        
        # 获取下一个输入路径
        input_path = input("\n请输入下一个图像路径 (或输入'q'退出):\n> ")
    
    print("程序结束")

if __name__ == '__main__':
    print("=" * 50)
    print("生成式对抗网络图像转换器（带重叠拼接）")
    print(f"模型分辨率: {IMG_WIDTH}x{IMG_HEIGHT}, 重叠大小: {OVERLAP}像素")
    print(f"最小尺寸: {MIN_SIZE}像素, 融合模式: {BLEND_MODE}")
    print("=" * 50)
    
    # 设置GPU内存自动增长
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"检测到GPU: {physical_devices}")
    
    main()