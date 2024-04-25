import cv2
import numpy as np
from matplotlib import pyplot as plt


def apply_affine_transform(image, points):
    '''

    :param image:
    :param points:  np.float32,一个包含四个点的列表，每个点是一个包含两个元素的列表，分别表示点的x和y坐标  np.float32([[352, 82], [2702, 465], [112, 1557], [2462, 1938]])
    :return:  返回一个经过仿射变换的图像
    '''
    # 将点转换为适合OpenCV的格式
    points = np.array(points, dtype=np.float32)
    #计算目标矩形的宽度和高度
    #distance=[np.sqrt((points[0][0]-points[i][0])**2+(points[0][1]-points[i][1])**2) for i in range(1,3)]
    # 计算目标矩形的宽度和高度
    target_width = int(np.max(points[:, 0]) - np.min(points[:, 0]))
    target_height = int(np.max(points[:, 1]) - np.min(points[:, 1]))

    # 定义目标矩形的四个角点，使其成为一个水平矩形
    target_points = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]],
                             dtype=np.float32)

    # 计算仿射变换矩阵
    M = cv2.getPerspectiveTransform(points, target_points)

    # 应用仿射变换
    transformed_image = cv2.warpPerspective(image, M, (target_width, target_height))

    return transformed_image

def restore(transformed_image, points, original_shape):
    '''
    :param transformed_image: 经过仿射变换的图像
    :param points:  np.float32, 一个包含四个点的列表，每个点是一个包含两个元素的列表，分别表示点的x和y坐标
    :param original_shape: 原始图像的尺寸 (width, height)
    :return: 返回一个恢复为原始图像的图像
    '''
    # 将点转换为适合OpenCV的格式
    points = np.array(points, dtype=np.float32)

    # 定义目标矩形的四个角点，使其成为一个水平矩形
    target_points = np.array([[0, 0], [original_shape[0], 0], [0, original_shape[1]], [original_shape[0], original_shape[1]]],
                             dtype=np.float32)

    # 计算仿射变换矩阵的逆矩阵
    M_inverse = cv2.getPerspectiveTransform(target_points, points)

    # 应用仿射逆变换
    restored_image = cv2.warpPerspective(transformed_image, M_inverse, original_shape)

    return restored_image

if __name__ == '__main__':
    # 读取图像
    image = cv2.imread('img/demo_rotate.png')
    original_shape = (image.shape[1], image.shape[0])
    # 定义原始矩形的四个角点
    original_points = np.float32([[352, 82], [2702, 465], [112, 1557], [2462, 1938]])

    transformed_image = apply_affine_transform(image, original_points)
    print(image.shape)
    print(transformed_image.shape)

    restored_image = restore(transformed_image, original_points, original_shape)
    print(restored_image.shape)

    # 一幅窗口显示原始图像和变换后的图像
    # 显示结果
    figura = plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
    plt.title('Restored Image')
    plt.axis('off')
    plt.show()

    # cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Original Image', 600, 600)
    # cv2.imshow('Original Image', image)
    # cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Transformed Image', 600, 600)
    # cv2.imshow('Transformed Image', transformed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
