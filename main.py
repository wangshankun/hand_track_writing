# -*- coding:utf-8 -*- 
import os
import cv2
import sys
import time
import copy
import random
import thread
import numpy as np
from mnist import mnist
import deepwell as dp
from PIL import Image


#设置np打印数组显示的门限
np.set_printoptions(threshold=np.nan)
#切换进程执行目录
os.chdir(sys.path[0])


lable_to_vector = [
[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.],
[0.,0.,0.,1.,0.,0.,0.,0.,0.,0.],
[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]

width      = 640#摄像头尺寸
hight      = 480
status     = 0#进程状态
num_labels = []#预测的结果
timg       = np.zeros((hight, width), np.uint8)#用于跟踪笔迹调试窗口
org_img    = np.zeros((hight, width), np.uint8)#摄像头原始窗口
mnist_img  = np.zeros((256, 256))#用于显示处理后的笔迹窗口,也是进行识别图像的扩大版

#获取笔迹的数据和label
def get_trace_data(data_path):
    data_out  = []
    label_out = []
    files = os.listdir(data_path)
    for x in files:
        p = data_path + x
        data = np.load(p)
        data = data.flatten()
        data_out.append(data)
        label = int(x.split("_")[0])#取第一个为label
        label_out.append(lable_to_vector[label])

    data_out_mtx  = np.vstack(data_out)
    label_out_mtx = np.vstack(label_out)

    np.random.seed(701507)#伪随机
    numSum = data_out_mtx.shape[0]
    shuffle_idx = np.random.permutation(numSum)
    np.random.shuffle(shuffle_idx)

    data_out_mtx  = data_out_mtx[shuffle_idx,:]
    label_out_mtx = label_out_mtx[shuffle_idx,:]

    return data_out_mtx, label_out_mtx
'''
#获取mnist的数据和label
def get_mnist_data(num):
    data_out  = []
    label_out = []
    dataset = mnist("train")#仅仅使用train的数据
    for q in range (0, num):
        label, img = dataset.GetImage(q)
        data_out.append(img)
        label_out.append(lable_to_vector[int(label)])
    data_out_mtx  = np.vstack(data_out)
    label_out_mtx = np.vstack(label_out)
    np.random.seed(701507)#伪随机
    shuffle_idx = np.random.permutation(num)
    np.random.shuffle(shuffle_idx)    
    data_out_mtx  = data_out_mtx[shuffle_idx,:]
    label_out_mtx = label_out_mtx[shuffle_idx,:]

    return data_out_mtx, label_out_mtx
'''

sarry_img = 0
def pic_mnist(arry_img):
    global sarry_img, mnist_img
    kernel=np.uint8(np.zeros((5,5)))#膨胀的内核为5x5的十字
    for x in range(5):
        kernel[x,2]=1
        kernel[2,x]=1
    #arry_img需要再确保一次自己为二值化图,才能进入findContours
    retval, arry_img = cv2.threshold(arry_img, 1, 255, cv2.THRESH_BINARY)
    arry_img = arry_img.astype(np.uint8)
    #寻找轮廓,剔除较小轮廓(比如小于60，丢弃)
    contours, hierarchy = cv2.findContours(arry_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    back = np.zeros_like(arry_img, np.uint8)    
    ct_dict = {}
    for i in range(len(contours)):
        ct_dict[i] = len(contours[i])
    ct_sd_list =  sorted(ct_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    for i in ct_sd_list:
        if i[1] > 60:#大于60个点的轮廓进行显示
            cv2.drawContours(back, contours, i[0], (255,255,255), -1)#在全黑背景图上，用白色填充轮廓
    dilate           = cv2.dilate(arry_img, kernel, iterations = 2)
    back             = cv2.bitwise_and(back, dilate) 
    row_order_nozero = (np.transpose(np.nonzero(back)))
    col_order_nozero = row_order_nozero[np.lexsort(row_order_nozero[:,1:2].T)]
    if row_order_nozero.shape[0] < 80:#非零数小于80个点，无效
        return None
    min_row_nozero   = int(row_order_nozero[0,0:1])
    max_row_nozero   = int(row_order_nozero[-1,0:1])
    min_col_nozero   = int(col_order_nozero[0,1:2])
    max_col_nozero   = int(col_order_nozero[-1,1:2])
    cut_img          = back[min_row_nozero:max_row_nozero+1, min_col_nozero:max_col_nozero+1]
    #填充为正方形
    h, w          = cut_img.shape
    if w >= h:
        h_pad   = (w - h)/2
        h_pad_  = h_pad + (w - h)%2
        hp_top  = np.zeros([h_pad, w])
        hp_bot  = np.zeros([h_pad_, w])
        pad_img = np.vstack((hp_top, cut_img, hp_bot))
    elif w < h:
        w_pad   = (h - w)/2 
        w_pad_  = w_pad + (h - w)%2
        wp_lef  = np.zeros([h, w_pad])
        wp_rig  = np.zeros([h, w_pad_])
        pad_img = np.hstack((wp_lef, cut_img, wp_rig))
         
    mnist = cv2.resize(pad_img,(28,28),interpolation=cv2.INTER_AREA)
    mnist = mnist.astype(np.uint8)
    retval, mnist = cv2.threshold(mnist, 1, 255, cv2.THRESH_BINARY)
    mnist = mnist.astype(np.uint8)
    name = "./mnist" +"_"+ str(sarry_img)+".npy"
    #np.save(name, mnist)#采集数据
    sarry_img = sarry_img + 1
    mnist_img = cv2.resize(pad_img, (256, 256), interpolation=cv2.INTER_AREA)
    return mnist
    
#初始化deepwell和设置参数
l_scale     = 6
d_scale     = 3
y_scale     = 0
h_size      = 0x7ff#0x800 * 8 = 16K

dp.INIT()
dp.WD_EN(1)
clear_M_en = 1
'''   
#先使用deepwell 训练和测试一遍mnist
datas, labels = get_mnist_data(10000)
org_labels    = [x.argmax() for x in labels]
org_ar        = np.array(org_labels)
tdatas        = datas[0:1000,]
torg_ar       = org_ar[0:1000,]
s = time.time()
dp.Train(datas, labels, l_scale, d_scale, y_scale, h_size, clear_M_en)
e = time.time()
print "deepwell train 10000 using time:%f"%(e - s)
dp.wait_for_idle(0)
s = time.time()
res_labels = dp.Test(tdatas, l_scale, d_scale, y_scale, h_size)
e = time.time()
print "deepwell test 1000 using time:%f"%(e - s)
res_ar = np.array(res_labels)

cmp_ar  = torg_ar - res_ar
acc =  float(np.sum(cmp_ar==0))/len(torg_ar)
print acc
'''
#再用deepwell训练和测试一遍本项目采集笔迹图(越靠后训练图，权重越大)
t_datas, t_labels = get_trace_data("./data/")
t_org_labels      = [x.argmax() for x in t_labels]
t_org_ar          = np.array(t_org_labels)
clear_M_en        = 0
s = time.time()
dp.Train(t_datas, t_labels, l_scale, d_scale, y_scale, h_size, clear_M_en)
e = time.time()
print "deepwell train trace using time:%f"%(e - s)
dp.wait_for_idle(0)
s = time.time()
res_labels = dp.Test(t_datas, l_scale, d_scale, y_scale, h_size)
e = time.time()
print "deepwell test trace using time:%f"%(e - s)
res_ar = np.array(res_labels)

cmp_ar  = t_org_ar - res_ar
acc =  float(np.sum(cmp_ar==0))/len(t_org_ar)
print acc

#跟踪笔迹和识别的主线程
def detect_dpw():
    global status, org_img, timg, mnist_img, num_labels, width, hight
    trace_status = 0
    hit_contours = []
    h=346#采集区域尺寸
    w=404
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    cap            = cv2.VideoCapture(1)
    while True:
        if status == -1:#退出
            return
        ret, org_img = cap.read()
        #采框放到视频正中心位置
        x = width/2 - w/2
        y = hight/2 - h/2
        img_crop = org_img[y:y+h, x:x+w, :];
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2,50,50]), np.array([15,255,255]))#在hsv域上,采集皮肤颜色    
        erosion = cv2.erode(mask2, kernel_ellipse, iterations = 1) #腐蚀之后,转二值图
        retval, binary = cv2.threshold(erosion, 15, 255, cv2.THRESH_BINARY)
        nz = np.nonzero(binary)
        if len(nz[0]) != 0:
            ty = nz[0][0]
            tx = nz[1][0]
            hit_contours.append((tx, ty))
            if hit_contours.count((tx, ty)) > 3:#在一个点持续发现3次,认为开始
                tnz  = np.nonzero(timg)
                if len(tnz[0]) < 10:#timg没有东西的再刷新，防止中间停顿时间长，误删
                    timg = np.zeros((hight, width), np.uint8) 
                    hit_contours = []
                    trace_status = 1
                    mnist_img    = np.zeros((256, 256))#清除显示中的笔迹图
                    num_labels   = []#清除上次预测的label
            if trace_status == 1:
                cv2.circle(timg,(tx,ty), 6, (255,255,255),-1)
        else:#没有目标发现,连续20次,认为书写结束,开始识别
           hit_contours.append((-1, -1))
           if hit_contours.count((-1, -1)) > 20:
                mnist  = pic_mnist(timg)
                if mnist != None:
                    mnist_1v  = mnist.reshape((1,-1))
                    num_labels = dp.Test(mnist_1v, l_scale, d_scale, y_scale, h_size)
                    print num_labels#打印出识别结果
                timg = np.zeros((hight, width), np.uint8)
                hit_contours = []
                trace_status = 0
        cv2.rectangle(org_img,(x, y),(x + w,y + h),(55,255,155),3)

#UI显示的主线程
displayer_w = 1920#显示器尺寸
displayer_h = 1080
back_array  = np.zeros((displayer_h, displayer_w))#黑色
back_img    = Image.new('RGBA', (displayer_w, displayer_h))
back_img.paste(Image.fromarray(np.uint8(back_array)), (0,0))#初始化为黑色背景
def show():
    global status, org_img, timg, mnist_img, num_labels, back_img
    cv2.namedWindow(".", cv2.cv.CV_WINDOW_NORMAL)
    is_fullscreen = 1
    cv2.setWindowProperty(".", 0, is_fullscreen)#只有smnist全屏
    while True:
        if status == -1:#退出
            return
        cv2.imshow("show", org_img)#这两幅图，退出全屏，调试时候查看用
        cv2.imshow("timg", timg)#

        back_img.paste(Image.fromarray(np.uint8(mnist_img)), (int(displayer_w/2 - 128), int(displayer_h/2 - 128)))#将结果复制到中心位置   
        back_img_cv2 = cv2.cvtColor(np.asarray(back_img),cv2.COLOR_RGB2BGR)
        retval, back_img_cv2 = cv2.threshold(back_img_cv2, 1, 255, cv2.THRESH_BINARY)
        back_img_cv2 = back_img_cv2.astype(np.uint8)

        #将结果显示在
        if len(num_labels) != 0:
            cv2.putText(back_img_cv2, str(num_labels), (int(displayer_w/2 + 256), int(displayer_h/2 + 256)), cv2.FONT_HERSHEY_PLAIN, 3.0, (255,255,255), 5, 5)
        cv2.imshow(".", back_img_cv2)

        key    = cv2.waitKey(1)&0xFF
        if key == ord('q'):#按键q，退出程序
            status = -1;
        if key == ord('f'):#按键f，退出/进入 全屏
            if is_fullscreen == 1:
                cv2.setWindowProperty(".", 0, 0)#退出全屏
                is_fullscreen = 0
            else:
                cv2.setWindowProperty(".", 0, 1)#进入全屏
                is_fullscreen = 1

#main函数开始，启动两个线程，分别用于ui显示和后端检测
ret = thread.start_new_thread(show, ())
ret = thread.start_new_thread(detect_dpw, ())

while status != -1:#如果status为-1,结束主进程的while循环,主进程退出后,两线程自动退出
   pass
