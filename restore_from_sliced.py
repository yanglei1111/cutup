import os
import cv2
from PIL import Image
slice_dir='sliced_images'
#将sliced的图像整合为一张图像
image_files = os.listdir(slice_dir)

# 解析图片名称以获取坐标
def parse_coordinates(filename):
    parts = filename.split('.')
    coords = parts[0].split('_')
    x_start, y_start, x_end, y_end = map(int, coords[1:])
    return x_start, y_start, x_end, y_end

# 创建一个足够大的画布来容纳所有切片
max_x = max(parse_coordinates(file)[2] for file in image_files)
max_y = max(parse_coordinates(file)[3] for file in image_files)
canvas = Image.new('RGB', (max_x, max_y))

# 粘贴每张图片到画布上
for image_file in image_files:
    x_start, y_start, x_end, y_end = parse_coordinates(image_file)
    img = Image.open(slice_dir +'/'+ image_file)
    canvas.paste(img, (x_start, y_start))

canvas.show()