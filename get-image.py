# -*- coding: UTF-8 -*-
'' '该程序用来识别一张图片当中有多少蚕茧，并且可以对单张图片中所有的单个蚕茧进行裁剪与保存的功能 '''

from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import cv2
import scipy.misc

position = []

image = Image.open('1.jpg')       #只是读取图片
img = image.filter(ImageFilter.GaussianBlur(radius=5))

img1 = img.convert("L")         #将图片转化为灰度图
imgarr = np.array(img1)        #将灰度图转化为数组，此时是一维
X = imgarr.shape[1]     #得到图片的长宽
Y = imgarr.shape[0]

#设定阈值进行二值化
binimg = (imgarr > 50).astype(np.uint8)
# binimg = cv2.threshold(img2arr, 75, 255, cv2.THRESH_BINARY)[1]

#找到连通区域，即找出蚕茧的位置及个数
cnts = cv2.findContours(binimg, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
number = len(cnts[1])
print(number)

#找出蚕茧的位置，并且把224*224的位置裁剪出来
for c in cnts[1]:
    M = cv2.moments(c)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if cx > 112 and cx < X -112 and cy > 112 and cy < Y -112:
        left = cx - 112
        right = cx + 112
        up = cy - 112
        down = cy + 112

    elif cx <= 112 and cy <=112 :
        left = 0
        right = 224
        up = 0
        down = 224

    elif cx <= 112 and cy >112 and cy < Y-112:
        left = 0
        right = 224
        up = cy - 112
        down = cy + 112

    elif cx <= 112 and cy > Y-112:
        left = 0
        right = 224
        up = Y - 224
        down = Y

    elif cx > 112 and cx < X - 112 and cy > Y -112:
        left = cx - 112
        right = cx + 112
        up = Y - 224
        down = Y

    elif cx >= X - 112 and cy > Y - 112:
        left = X - 224
        right = X
        up = Y - 224
        down = Y

    elif cx >= X - 112 and cy > 112 and cy < Y -112:
        left = X - 224
        right = X
        up = cy -112
        down = cy + 112

    elif cx >= X - 112 and cy < 112:
        left = X - 224
        right = X
        up = 0
        down = 224

    elif cx > 112 and cx < X - 112 and cy < 112:
        left = cx - 112
        right = cx + 112
        up = 0
        down = 224

    region = [left,up,right,down]
    position.append(region)

#裁剪各个蚕茧并保存图片
for each in range(number):
    print(position[each])
    new = image.crop(position[each])
    scipy.misc.imsave("image%s.jpg" % each, new)
    plt.imshow(new)
    plt.show()

#查看找出的蚕茧位置
cv2.drawContours(binimg, cnts[1], -1, (0, 0, 255), 3)
plt.imshow(binimg)
plt.show()