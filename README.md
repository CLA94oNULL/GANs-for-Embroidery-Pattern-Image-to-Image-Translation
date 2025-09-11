# GANs-for-Embroidery-Pattern-Image-to-Image-Translation
GANs for Embroidery Pattern Image-to-Image Translation

该项目需要python3.8，tensorflow2.13.0，matplotlib，numpy，cv2

GAN.py用于网络训练。运行即可自动开始训练，支持保存checkpoint用于回溯训练过程。每次训练开始时会自动加载最新的checkpoint，在此基础上开始训练

image_converter.py用于图生图

pix2pix_generator_final.h5是权重文件

在input_image和target_image文件夹中放入对应的图片，并从000000.png开始依次递增，即可完成数据集的构建

output_image文件夹用于存储生成的图像