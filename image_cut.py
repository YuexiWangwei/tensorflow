# -*- coding: UTF-8 -*-

''' '

此程序是界面化剪裁图片，对一张图片16个图片进行剪裁
 
'''''


import  PIL
from PIL import ImageFilter
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import cv2
import scipy.misc

import os
from tkinter import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText


def close():
    exit(0)


def scan_files(file):           #扫描文件夹中所有文件并返回一个列表
    for root,dirs,files in os.walk(file):
        global filelists,path
        filelists = files
        path = root + "/"
        print(path)
    return filelists,path


def openfile():
    filedir = filedialog.askdirectory()  # 打开文件夹
    filelists = scan_files(filedir)  # 得到文件夹中的所有文件

    # 显示当前文件夹路径
    labelfile.delete(1.0, END)
    labelfile.insert(INSERT, path)

def chosepath():
    filedir = filedialog.askdirectory()  # 打开文件夹
    for root,dirs,files in os.walk(filedir):
        global savepath
        savepath = root + "/"

    # 显示当前文件夹路径
    labelpath.delete(1.0, END)
    labelpath.insert(INSERT, savepath)


# 找到每张图片中蚕茧的位置
def find_cocoon(temp_image):

    position = []
    image = PIL.Image.open("%s"%temp_image) # 只是读取图片
    img = image.filter(ImageFilter.GaussianBlur(radius=15))

    img1 = img.convert("L")  # 将图片转化为灰度图
    imgarr = np.array(img1)  # 将灰度图转化为数组，此时是一维
    X = imgarr.shape[1]  # 得到图片的长宽
    Y = imgarr.shape[0]

    # 设定阈值进行二值化
    binimg = (imgarr > 80).astype(np.uint8)
    # binimg = cv2.threshold(img2arr, 75, 255, cv2.THRESH_BINARY)[1]

    # 找到连通区域，即找出蚕茧的位置及个数
    cnts = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number = len(cnts[1])

    # 找出蚕茧的位置，并且把224*224的位置裁剪出来
    for c in cnts[1]:
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if cx > 112 and cx < X - 112 and cy > 112 and cy <= Y - 112:
            left = cx - 112
            right = cx + 112
            up = cy - 112
            down = cy + 112

        elif cx <= 112 and cy <= 112:
            left = 0
            right = 224
            up = 0
            down = 224

        elif cx <= 112 and cy > 112 and cy <= Y - 112:
            left = 0
            right = 224
            up = cy - 112
            down = cy + 112

        elif cx <= 112 and cy > Y - 112:
            left = 0
            right = 224
            up = Y - 224
            down = Y

        elif cx > 112 and cx < X - 112 and cy > Y - 112:
            left = cx - 112
            right = cx + 112
            up = Y - 224
            down = Y

        elif cx >= X - 112 and cy > Y - 112:
            left = X - 224
            right = X
            up = Y - 224
            down = Y

        elif cx >= X - 112 and cy > 112 and cy <= Y - 112:
            left = X - 224
            right = X
            up = cy - 112
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

        region = [left, up, right, down]
        position.append(region)

    return image,number,position


def start():

    image_path = []
    #获取输出框中的信息并且清除框中的信息
    name = entry_name.get()
    begin = entry_number.get()
    entry_name.delete(0, END)
    entry_number.delete(0,END)

    #判断是否选择了文件夹，若没有则提示错误
    #有则获取每个图片地址的完整信息,上面得到的filelists是一个列表，path是字符串
    try:
        for each in filelists:
            image = path + each
            image_path.append(image)
        savepath != ""
        os.chdir(savepath)
    except:
        textshow.insert(INSERT, "请选择文件所在文件夹以及保存路径！\n")
        textshow.see(END)
        textshow.update()

    #获取照片数量
    image_num = len(image_path)
    print(image_num)
    #保证名称和序号不能为空，若有则会提示出错
    if name=="" or begin =="":
        textshow.insert(INSERT,"请输入保存名称和开始序号！\n")
        textshow.see(END)
        textshow.update()

    #获得蚕茧的的进行剪裁和保存
    else:
        rank = int(begin) - 1
        for each in range(image_num):#image_num
            img,num,pos = find_cocoon(image_path[each])     #蚕茧的数量以及位置

            #当前图片完整信息
            states = image_path[each]
            print(states)
            # 进行图片裁剪以及重命名保存,num是该图片中蚕茧数量，pos是蚕茧四周位置,打开保存位置
            for each in range(num):
                rank = rank + 1
                # print("%s%04d.jpg" % (name,rank))
                new = img.crop(pos[each])
                scipy.misc.imsave("%s%04d.jpg" % (name,rank), new)

            textshow.insert(INSERT,"%s 已经剪裁完成\n"%states)
            textshow.see(END)
            textshow.update()


root= Tk()
root.title("剪裁蚕茧图片")

openbutton = Button(root,text="选择文件",command = openfile, font="幼圆")
labelfile = Text(root,width = 45,height = 1,font = "幼圆")
save_path = Button(root,text = "保存路径",command = chosepath,font = "幼圆")
labelname = Label(root,text = "保存名称:",font = "幼圆")
labelnumber = Label(root,text = "开始序号:",font = "幼圆")
labelpath = Text(root,width = 45,height = 1,font = "幼圆")
surebutton = Button(root, text=" 确   定 ", command=start, font="幼圆")
closebutton = Button(root, text=" 退   出 ", command=close, font="幼圆")
entry_name = Entry(root)
entry_number = Entry(root)
textshow = ScrolledText(root, width=60, height=20, font="幼圆")

openbutton.grid(row = 0,column = 0,padx =15,pady = 15)
labelfile.grid(row = 0,column=1,columnspan = 3,padx = 15,pady = 15)

save_path.grid(row = 1,column = 0,padx = 15,pady = 15)
labelpath.grid(row = 1,column = 1,columnspan = 3,padx = 15,pady = 15)

labelname.grid(row = 2,column = 0,padx = 15,pady = 15)
entry_name.grid(row = 2,column = 1,padx = 15,pady = 15)

labelnumber.grid(row = 2,column = 2,padx = 15,pady = 15)
entry_number.grid(row = 2,column = 3,padx = 15,pady = 15)

surebutton.grid(row = 3,column = 1,padx = 15,pady = 15)
closebutton.grid(row = 3,column = 2,padx = 15,pady = 15)
textshow.grid(rowspan=5, columnspan=4, padx=10, pady=25)

mainloop()