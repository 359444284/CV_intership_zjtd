import numpy as np
import cv2 as cv

# Copy From https://www.jianshu.com/p/9194f43fd68a
# 效果是以中心为原点旋转
def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv.warpAffine(image, M, (nW, nH))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # default
    img_colour = cv.imread('./images/test/test_img_1.png', cv.IMREAD_COLOR)
    img_gray = cv.imread('./images/test/test_img_1.png', cv.IMREAD_GRAYSCALE)
    img_withAlpha = cv.imread('./images/test/test_img_1.png', cv.IMREAD_UNCHANGED)

    # 图像显示
    # cv.imshow('img_gray', img_gray)
    # cv.imshow('img_withAlpha', img_withAlpha)

    # 像素修改
    # img_colour[0:50, 0:50] = [0]
    # img_colour[-50:, -50:] = img_colour[0:50, 0:50]

    # 保存图像
    # cv.imwrite('./test_write.jpg', img_colour)

    # 图像属性
    # print(img_colour.shape)  # 获取图像的形状，返回值是一个包含行数、列数、通道数的元组。
    # print(img_colour.size)  # 获得图像的像素数目。
    # print(img_colour.dtype)  # 获得图像的数据类型

    # 图像缩放
    # image_resize_1 = cv.resize(img_colour, (200, 200), interpolation=cv.INTER_LINEAR)
    # image_resize_2 = cv.resize(img_colour, (500, 500), interpolation=cv.INTER_LINEAR)
    # image_resize_3 = cv.resize(img_colour, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    # image_resize_4 = cv.resize(img_colour, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    # image_resize_5 = cv.resize(img_colour, (200, 200), fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    # cv.imshow('200*200', image_resize_1)
    # cv.imshow('500*500', image_resize_2)
    # cv.imshow('half', image_resize_3)
    # cv.imshow('double', image_resize_4)
    # cv.imshow('double_fix', image_resize_5)

    # 金字塔
    # level = 3
    # new_img = cv.resize(img_colour, (256, 256))
    # tmp = new_img.copy()
    # imgs = []
    # for i in range(level):
    #     lower = cv.pyrDown(tmp)
    #     imgs.append(lower)
    #     # cv.imshow('lower: ' + str(i),lower)
    #     tmp = lower.copy()
    #
    # cv.waitKey()
    #
    # # 从下到上循环
    # lpls = []
    # for i in range(len(imgs)-1, -1, -1):
    #     if (i-1) < 0:
    #         up = cv.pyrUp(imgs[i])
    #         lpl = cv.subtract(new_img, up)
    #         lpls.append(lpl)
    #     else:
    #         up = cv.pyrUp(imgs[i])
    #         lpl = cv.subtract(imgs[i-1], up)
    #         lpls.append(lpl)
    #     # cv.imshow('lpl: ' + str(i), lpl)
    #
    # cv.waitKey()
    #
    # lpl_rep = imgs[-1]
    # normal_rep = imgs[-1]
    # for i in range(len(imgs)):
    #     lpl_rep = lpls[i] + cv.pyrUp(lpl_rep)
    #     normal_rep = cv.pyrUp(normal_rep)
    #
    # print(lpl_rep.shape)
    # cv.imshow('lpl_rep', lpl_rep)
    # cv.imshow('normal_rep: ', normal_rep)

    # 图像旋转
    # rotated_1 = rotate_bound(img_colour, 50)
    # 顺时针50度， 左上角， 不缩放
    # M = cv.getRotationMatrix2D((0, 0), -50, 1.0)
    # rotated_2 = cv.warpAffine(img_colour, M, img_colour.shape[:2])
    # cv.imshow('center_-50', rotated_1)
    # cv.imshow('corner_50', rotated_2)

    # 图像平滑处理
    #   均值滤波
    # average = cv.blur(img_colour, (21,21))
    # res = np.hstack((img_colour, average))
    # cv.imshow('average', res)
    #   高斯滤波
    # gauss = cv.GaussianBlur(img_colour, (21, 21), 0)
    # res = np.hstack((img_colour, gauss))
    # cv.imshow('gauss', res)
    #   中值滤波
    # median = cv.medianBlur(img_colour, 21)
    # res = np.hstack((img_colour, median))
    # cv.imshow('median', res)

    # 卷积操作
    # kernal = np.array([[-1,-1,-1],
    #           [-1,8,-1],
    #           [-1,-1,-1]])
    # edge = cv.filter2D(img_colour, -1, kernal)
    # res = np.hstack((img_colour, edge))
    # cv.imshow('edge', res)




    # cv.imshow('double_fix', image_resize_5)
    cv.waitKey(0)
    cv.destroyAllWindows()
