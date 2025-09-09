import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 配置参数
BUFFER_SIZE = 400
BATCH_SIZE = 4
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3  # 假设输出是RGB图像
EPOCHS = 50
LAMBDA = 100  # L1损失的权重

# 数据集准备函数保持不变
def load_and_preprocess_image(image_path):
    """加载和预处理图像"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # 调整大小
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # 标准化到 [-1, 1] 范围
    image = (image - 0.5) * 2
    return image

# def create_dataset(input_paths, target_paths):
#     """创建TensorFlow数据集"""
#     dataset = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
#     dataset = dataset.map(lambda input_path, target_path: (
#         load_and_preprocess_image(input_path),
#         load_and_preprocess_image(target_path)
#     ))
#     dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#     return dataset

# 修改 create_dataset 函数来实现8种翻转
def create_dataset(input_paths, target_paths):
    """创建TensorFlow数据集，每张图像生成8种变换版本"""
    # 定义8种变换操作（包括原图）
    def apply_transformations(input_img, target_img):
        # 原始图像
        images = [(input_img, target_img)]
        # 水平翻转
        images.append((tf.image.flip_left_right(input_img), tf.image.flip_left_right(target_img)))
        # 垂直翻转
        images.append((tf.image.flip_up_down(input_img), tf.image.flip_up_down(target_img)))
        # 水平+垂直翻转
        flip_both_img = tf.image.flip_left_right(tf.image.flip_up_down(input_img))
        flip_both_target = tf.image.flip_left_right(tf.image.flip_up_down(target_img))
        images.append((flip_both_img, flip_both_target))
        # 旋转90度
        images.append((tf.image.rot90(input_img, k=1), tf.image.rot90(target_img, k=1)))
        # 旋转180度
        images.append((tf.image.rot90(input_img, k=2), tf.image.rot90(target_img, k=2)))
        # 旋转270度
        images.append((tf.image.rot90(input_img, k=3), tf.image.rot90(target_img, k=3)))
        # 水平翻转+旋转90度
        flip_rot90_img = tf.image.rot90(tf.image.flip_left_right(input_img), k=1)
        flip_rot90_target = tf.image.rot90(tf.image.flip_left_right(target_img), k=1)
        images.append((flip_rot90_img, flip_rot90_target))
        
        return images
    
    # 创建基础数据集（每对路径）
    dataset = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
    
    # 加载和预处理图像
    dataset = dataset.map(lambda input_path, target_path: (
        load_and_preprocess_image(input_path),
        load_and_preprocess_image(target_path)
    ))
    
    # 对每对图像应用8种变换并展开
    dataset = dataset.flat_map(lambda input_img, target_img: 
        tf.data.Dataset.from_tensor_slices(apply_transformations(input_img, target_img))
    )
    
    # 混洗和批处理
    # dataset = dataset.shuffle(BUFFER_SIZE * 8).batch(BATCH_SIZE)
    return dataset

# 生成器构建函数保持不变
# 生成器构建函数 - 参数量翻倍版本
def build_generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS])
    
    # 1. 定义下采样块函数（通道数翻倍）
    def downsample_block(filters, size, apply_batchnorm=True):
        filters *= 2  # 将通道数翻倍
        result = tf.keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                 use_bias=False))
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result
    
    # 2. 下采样块定义（通道数翻倍）
    down1 = downsample_block(64, 4, apply_batchnorm=False)  # 64 → 128
    down2 = downsample_block(128, 4)                        # 128 → 256
    down3 = downsample_block(256, 4)                        # 256 → 512
    down4 = downsample_block(512, 4)                        # 512 → 1024
    down5 = downsample_block(512, 4)                         # 512 → 1024
    
    # 3. 定义上采样块函数（通道数翻倍）
    def upsample_block(filters, size, apply_dropout=False):
        filters *= 2  # 将通道数翻倍
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                          kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                          use_bias=False))
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result
    
    # 4. 上采样块定义（通道数翻倍）
    up1 = upsample_block(512, 4, apply_dropout=True)  # 512 → 1024
    up2 = upsample_block(256, 4)                       # 256 → 512
    up3 = upsample_block(128, 4)                       # 128 → 256
    up4 = upsample_block(64, 4)                         # 64 → 128
    
    # 5. 最后一个上采样层（保持不变，因为输出通道固定）
    last = layers.Conv2DTranspose(
        OUTPUT_CHANNELS, 4, strides=2,
        padding='same', activation='tanh'
    )  # (128,128) -> (256,256)
    
    # 6. 下采样路径
    # 初始输入 (256, 256, 3)
    x = inputs
    
    # 下采样过程
    d1 = down1(x)    # -> (128,128,128)
    d2 = down2(d1)   # -> (64,64,256)
    d3 = down3(d2)   # -> (32,32,512)
    d4 = down4(d3)   # -> (16,16,1024)
    d5 = down5(d4)   # -> (8,8,1024) - 瓶颈层
    
    # 7. 上采样过程
    u1 = up1(d5)     # (8,8)->(16,16) [1024通道]
    u1 = layers.Concatenate()([u1, d4])  # 连接相同的16x16尺寸 -> 2048通道
    
    u2 = up2(u1)      # (16,16)->(32,32) [512通道]
    u2 = layers.Concatenate()([u2, d3])  # 连接相同的32x32尺寸 -> 1024通道
    
    u3 = up3(u2)      # (32,32)->(64,64) [256通道]
    u3 = layers.Concatenate()([u3, d2])  # 连接相同的64x64尺寸 -> 512通道
    
    u4 = up4(u3)      # (64,64)->(128,128) [128通道]
    u4 = layers.Concatenate()([u4, d1])  # 连接相同的128x128尺寸 -> 256通道
    
    # 8. 最终输出
    output = last(u4)  # (128,128)->(256,256) [3通道]
    
    return tf.keras.Model(inputs=inputs, outputs=output)


# 构建判别器 (PatchGAN)
def build_discriminator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS], name='input_image')
    targets = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS], name='target_image')
    
    x = layers.concatenate([inputs, targets])  # 连接输入和目标图像
    
    down1 = downsample(64, 4, False)(x)        # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)           # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)           # (bs, 32, 32, 256)
    
    zero_pad1 = layers.ZeroPadding2D()(down3)   # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02), 
                         use_bias=False)(zero_pad1)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02))(zero_pad2)  # (bs, 30, 30, 1)
    
    return Model(inputs=[inputs, targets], outputs=last)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

# 定义损失函数
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # L1损失
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + LAMBDA * l1_loss

# 创建优化器
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# 创建模型
generator = build_generator()
discriminator = build_discriminator()

# 修复1: 使用CheckpointManager来管理检查点
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# 修复2: 创建CheckpointManager并设置max_to_keep=1
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, 
    directory=checkpoint_dir, 
    max_to_keep=1
)

# 训练步骤
@tf.function
def train_step(input_image, target):
    
    if len(input_image.shape) == 3:
        input_image = tf.expand_dims(input_image, axis=0)
    if len(target.shape) == 3:
        target = tf.expand_dims(target, axis=0)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 生成图像
        gen_output = generator(input_image, training=True)
        
        # 判别器输出
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        # 计算损失
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
    # 计算梯度
    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # 应用梯度
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# 训练循环
def train(dataset, epochs):
    start_time = time.time()
    
    # 检查是否有保存的模型
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"从检查点恢复模型: {checkpoint_manager.latest_checkpoint}")
    else:
        print("没有找到检查点，从零开始训练")
    
    for epoch in range(epochs):
        print(f"\n开始训练周期 {epoch+1}/{epochs}")
        epoch_start = time.time()
        batch_losses = []
        
        # 训练每个batch
        for batch_idx, (input_batch, target_batch) in enumerate(dataset):
            gen_loss, disc_loss = train_step(input_batch, target_batch)
            batch_losses.append((gen_loss, disc_loss))
            
            # 每10个batch打印一次损失
            if batch_idx % 10 == 0:
                avg_gen_loss = tf.reduce_mean([l[0] for l in batch_losses]).numpy()
                avg_disc_loss = tf.reduce_mean([l[1] for l in batch_losses]).numpy()
                print(f'批次 {batch_idx}: G Loss: {gen_loss:.4f}, D Loss: {disc_loss:.4f}, 平均: G: {avg_gen_loss:.4f}, D: {avg_disc_loss:.4f}')
        
        # 计算并打印epoch平均损失
        avg_gen_loss = tf.reduce_mean([l[0] for l in batch_losses]).numpy()
        avg_disc_loss = tf.reduce_mean([l[1] for l in batch_losses]).numpy()
        print(f'周期 {epoch+1} 完成 - 平均G损失: {avg_gen_loss:.4f}, D损失: {avg_disc_loss:.4f}')
        
        with open('loss.txt', 'a') as f:
            f.write(f'{epoch+1},{avg_gen_loss:.6f},{avg_disc_loss:.6f}\n')
        
        if epoch % 10 == 0:
            generator.save(f'./save/{epochs}_pix2pix_generator_final.h5')
            print("备份模型已保存")
        
        generator.save('pix2pix_generator_final.h5')
        print("模型已保存为 pix2pix_generator_final.h5")
        
        # 保存检查点（保留最新）
        checkpoint_manager.save()
        epoch_time = time.time() - epoch_start
        print(f'周期 {epoch+1} 耗时: {epoch_time:.2f}秒')
    
    # 打印总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")
    

# 生成图像函数
def generate_images(model, input, target):
    prediction = model(input, training=True)
    plt.figure(figsize=(15, 15))
    
    display_list = [input[0], target[0], prediction[0]]
    title = ['输入图像', '目标图像', '生成图像']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # 将图像从[-1,1]转换回[0,1]
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(f'generated_image_{int(time.time())}.png')
    plt.show()

# 主函数
def main():
    # 这里替换成你自己的数据集路径
    # 假设你有input_images和target_images两个文件夹
    # 包含成对的输入和目标图像
    
    # 示例文件路径 (需要替换)
    input_image_paths = [f'./input_images/{i:06d}.png' for i in range(0, 4000)]
    target_image_paths = [f'./target_images/{i:06d}.png' for i in range(0, 4000)]
    
    # 验证文件是否存在
    for path in input_image_paths[:10]:
        if not os.path.exists(path):
            print(f"警告: 输入文件 {path} 不存在")
    for path in target_image_paths[:10]:
        if not os.path.exists(path):
            print(f"警告: 目标文件 {path} 不存在")
    
    # 创建数据集
    dataset = create_dataset(input_image_paths, target_image_paths)
    print(f"已创建数据集, 包含 {len(input_image_paths)} 对图像")
    
    # 开始训练
    train(dataset, EPOCHS)

if __name__ == '__main__':
    # 启用内存增长（避免一次性分配所有显存）
    physical_devices = tf.config.list_physical_devices('GPU')
    # if physical_devices:
    #     for device in physical_devices:
    #         tf.config.experimental.set_memory_growth(device, True)
    #     print(f"GPU加速可用: {physical_devices}")
    # else:
    #     print("警告: 未检测到GPU，将在CPU上训练")
    
    main()