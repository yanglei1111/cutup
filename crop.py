from affine_transform import apply_affine_transform
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
def make_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        import shutil
        shutil.rmtree(path)
        os.makedirs(path)
    return
def crop(img, crop_width, crop_height, width_stride, height_stride, is_affine, points=None):
    '''
    :param img: 要裁剪的图片
    :param crop_width: 裁剪后的图片的宽度
    :param crop_height: 裁剪后的图片的高度
    :param width_stride: x轴方向的步长
    :param height_stride: y轴方向的步长
    :param is_affine: 如果为True，则对图片进行仿射变换
    :param points: 如果is_affine为True，则必须提供四个角点，否则为None
    :return:
    包含裁剪后的图片的列表  [height_num, width_num, crop_height, crop_width, channel]
    '''
    if is_affine:
        assert points is not None and len(points) == 4, "提供四个点"
        #assert all(isinstance(point, list) and len(point) == 2 for point in points), "坐标点包括x、y"
        img = apply_affine_transform(img, points)
    # 依据给定的尺寸成批裁剪图片
    cropped_images = []
    img_height, img_width = img.shape[:2]
    for i in range(0, img_height - crop_height + 1, height_stride):
        row_images = []
        for j in range(0, img_width - crop_width + 1, width_stride):
            row_images.append(img[i:i + crop_height, j:j + crop_width])
        cropped_images.append(row_images)
        # 保存裁剪后的图片,名称为坐标点，
    make_dir("sliced_img")
    for i in range(len(cropped_images)):
        for j in range(len(cropped_images[i])):
            cv2.imwrite("sliced_img/crop_{}_{}_{}_{}.png".format(j * width_stride, i * height_stride,
                                                                 j * width_stride + crop_width,
                                                                 i * height_stride + crop_height),
                        cropped_images[i][j])
    return cropped_images
if __name__ == "__main__":
    img = cv2.imread("img/demo_rotate.png")
    points = np.float32([[352, 82], [2702, 465], [112, 1557], [2462, 1938]])
    crop_width = 800
    crop_height = 800
    is_affine = True
    width_stride = 400
    height_stride = 400
    cropped_images = crop(img, crop_width, crop_height, width_stride, height_stride, is_affine, points)
    print(np.array(cropped_images).shape)
    x_num= len(cropped_images[0])
    y_num= len(cropped_images)
    print(x_num, y_num)
    f, axarr = plt.subplots(y_num-1,x_num-1, figsize=(13, 13))
    for ind2 in range(0,x_num-1):
        for ind1 in range(0,y_num-1):
            img = Image.open("sliced_img/crop_{}_{}_{}_{}.png".format(ind2 * width_stride, ind1 * height_stride,
                                                                     ind2 * width_stride + crop_width,
                                                                     ind1 * height_stride + crop_height))
            axarr[ind1, ind2].imshow(img)
    plt.show()
    # w11= cropped_images[12][18]
    # cv2.namedWindow("w11", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("w11", 256, 256)
    # cv2.imshow("w11", w11)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(len(cropped_images))
