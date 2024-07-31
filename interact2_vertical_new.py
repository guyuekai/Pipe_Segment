import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
import math
from scipy import ndimage
import pandas as pd
import time
import os
import sys
from segment_anything import sam_model_registry, SamPredictor
import tkinter as tk
import tkinter.filedialog
from PIL import Image


start = time.time()
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


def mul_theta(x):   # 垂直修正用
    return x * fw


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 20 / 255, 30 / 255, 0.1])
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


sys.path.append("..")

sam_checkpoint = "model_check/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

path = tkinter.filedialog.askdirectory(title="请选择文件夹", initialdir=".", mustexist=True)
folder_name = os.path.basename(os.path.normpath(path))

def get_number(filename):
    # 去掉文件名中的前缀和后缀
    filename = filename.replace(folder_name + "-", "").replace(".jpg", "")
    # 转换为整数
    return int(filename)


# 选择的文件夹的名字
files = os.listdir(path)
files = [file for file in files if file.endswith('.jpg')]
files.sort(key=get_number)# 获取文件夹内的所有文件名,并从小到大进行排序
# 获取第一个图像和最后一个图像的文件名
file_first = files[0]
file_last = files[-1]

# 对第一张图像取点
positive_first = []
negative_first = []
def onclick_first(event):
    # 打印鼠标的位置和按键
    print('you pressed', event.button, event.xdata, event.ydata)
    # 判断鼠标按键
    if event.button == 1:  # 左键
        color = 'red'
    elif event.button == 3:  # 右键
        color = 'blue'
    else: # 其他按键
        color = 'black'
    # 在图像上显示一个圆点和坐标
    plt.scatter(event.xdata, event.ydata, s=50, c=color, marker='o')
    plt.annotate(f'({event.xdata:.2f}, {event.ydata:.2f})', xy=(event.xdata, event.ydata), xytext=(event.xdata+10, event.ydata+10))
    if event.button == 1:  # 左键
        positive_first.append((event.xdata, event.ydata))
    if event.button == 3:  # 右键
        negative_first.append((event.xdata, event.ydata))
    # 刷新图像
    plt.draw()


# 第一张图
# cut_img = vertical_correction(path, file_first)
image_first = Image.open(path+'//'+file_first).convert('RGB')
image_first = np.asarray(image_first)
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.figure(figsize=(25,25))
plt.title('左键添加位置，右键排除位置')
plt.imshow(image_first)
plt.axis('off')
# 连接鼠标点击事件和函数
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick_first)
# 显示图形界面
plt.show()
# 添加正标签
input_point_first = np.array(positive_first)
input_label_first = np.ones(len(input_point_first))
# 添加负标签
negative_point = np.array(negative_first)
negative_label = np.zeros(len(negative_point))
input_point_first = np.concatenate((input_point_first, negative_point), axis=0)
input_label_first = np.append(input_label_first, negative_label)

# 对最后一张图像取点
positive_last = []
negative_last = []
pos_list = []
def onclick_last(event):
    # 打印鼠标的位置和按键
    print('you pressed', event.button, event.xdata, event.ydata)
    # 判断鼠标按键
    if event.button == 1:  # 左键
        color = 'red'
    elif event.button == 3:  # 右键
        color = 'blue'
    else: # 其他按键
        color = 'black'
    # 在图像上显示一个圆点和坐标
    plt.scatter(event.xdata, event.ydata, s=50, c=color, marker='o')
    plt.annotate(f'({event.xdata:.2f}, {event.ydata:.2f})', xy=(event.xdata, event.ydata), xytext=(event.xdata+10, event.ydata+10))
    if event.button == 1:  # 左键
        positive_last.append((event.xdata, event.ydata))
    if event.button == 3:  # 右键
        negative_last.append((event.xdata, event.ydata))
    # 刷新图像
    plt.draw()

# 最后一张图
image_last = Image.open(path+'//'+file_last).convert('RGB')
image_last = np.asarray(image_last)
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.figure(figsize=(25,25))
plt.title('左键添加位置，右键排除位置')
plt.imshow(image_last)
plt.axis('off')
# 连接鼠标点击事件和函数
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick_last)
# 显示图形界面
plt.show()
# 添加正标签
input_point_last = np.array(positive_last)
input_label_last = np.ones(len(input_point_last))
# 添加负标签
negative_point = np.array(negative_last)
negative_label = np.zeros(len(negative_point))
input_point_last = np.concatenate((input_point_last, negative_point), axis=0)
input_label_last = np.append(input_label_last, negative_label)



input_point_cnt = 0

for file in files:  # 遍历文件名【待添加：只读jpg文件】
    plt.close()
    image = Image.open(path + '//' + file).convert('RGB')
    image = np.asarray(image)
    predictor.set_image(image)
    try:
        input_point_cnt +=1
        input_point_sub = np.subtract(input_point_last, input_point_first)
        input_point_multiple = np.multiply(input_point_cnt/len(files),input_point_sub)
        input_point = np.add(input_point_first, input_point_multiple)
        # input_point_stack = np.stack([input_point_first, input_point_last], axis=0)
        # input_point = np.mean(input_point_stack, axis=0)
        input_label = input_label_last


        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]

        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # show_points(input_point, input_label, plt.gca())
        # plt.axis('on')
        # plt.show()

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )


        df = pd.DataFrame(columns=['行数', '左外径', '左内径', '左壁厚', '右外径', '右内径', '右壁厚', '内径', '外径'])
        image_height, image_width = image.shape[:2]
        print(image_width)
        print(image_height)
        remainder = math.floor(image_height / 200)
        row_list = [i * 200 for i in range(1, remainder + 1)]
        pos_list = []
        thickness_left = []
        thickness_right = []
        inner_diameter = []
        outside_diameter = []
        pos0_list = []

        for row in row_list:
            for pos in range(1, masks.shape[2]-2):
                if(masks[0][row][pos] == True and masks[0][row][pos+1] == False) or (masks[0][row][pos] == False and masks[0][row][pos+1] == True):
                    pos_list.append(pos)
                    for point in pos_list:
                        cv2.circle(image, (point,row), 3, (255, 0, 0), 5)
            temp = {'行数': row, '左外径': pos_list[0], '左内径': pos_list[1], '左壁厚': pos_list[1] - pos_list[0],
                        '右内径': pos_list[2], '右外径': pos_list[3], '右壁厚': pos_list[3]-pos_list[2],
                        '内径': pos_list[2]-pos_list[1], '外径': pos_list[3]-pos_list[0]}
            thickness_left.append(pos_list[1] - pos_list[0])
            thickness_right.append(pos_list[3] - pos_list[2])
            inner_diameter.append(pos_list[2] - pos_list[1])
            outside_diameter.append(pos_list[3] - pos_list[0])
            df = pd.concat([df, pd.DataFrame(temp, index=[0])], axis=0, ignore_index=True)
            pos0_list.append(pos_list[0])  # y坐标
            pos_list = []

        k, b = np.polyfit(row_list, pos0_list, 1)
        print("直线方程为: y = {:.4f}x + {:.2f}".format(k, b))
        theta = math.atan(k)
        fw = math.cos(theta)

        thickness_left = list(map(mul_theta, thickness_left))
        thickness_right = list(map(mul_theta, thickness_right))
        inner_diameter = list(map(mul_theta, inner_diameter))
        outside_diameter = list(map(mul_theta, outside_diameter))
        pos0_list = []

        thickness_left_mean = np.mean(thickness_left)
        thickness_right_mean = np.mean(thickness_right)
        inner_diameter_mean = np.mean(inner_diameter)
        outside_diameter_mean = np.mean(outside_diameter)

        thickness_left_std = np.std(thickness_left)
        thickness_right_std = np.std(thickness_right)
        inner_diameter_std = np.std(inner_diameter)
        outside_diameter_std = np.std(outside_diameter)

        thickness_left_percent = "%.2f%%" % (thickness_left_std / thickness_left_mean * 100)
        thickness_right_percent = "%.2f%%" % (thickness_right_std / thickness_right_mean * 100)
        inner_diameter_percent = "%.2f%%" % (inner_diameter_std / inner_diameter_mean * 100)
        outside_diameter_percent = "%.2f%%" % (outside_diameter_std / outside_diameter_mean * 100)

        df.loc[17] = ['均值', '', '', thickness_left_mean, '', '', thickness_right_mean, inner_diameter_mean,
                      outside_diameter_mean]
        df.loc[18] = ['标准差', '', '', thickness_left_std, '', '', thickness_right_std, inner_diameter_std,
                      outside_diameter_std]
        df.loc[19] = ['标准差百分比', '', '', thickness_left_percent, '', '', thickness_right_percent,
                      inner_diameter_percent, outside_diameter_percent]

        csv_name = file.replace('.jpg', '.csv')
        if not os.path.exists('D:\HK\TestPipeOutput_csv\\' + folder_name):  # 如果文件夹不存在
            os.mkdir('D:\HK\TestPipeOutput_csv\\' + folder_name)  # 创建文件夹
        df.to_csv('D:\HK\TestPipeOutput_csv\\' + folder_name + '\\' + csv_name, sep=',', index=False, encoding='utf_8_sig')  # 【待添加：自动创建文件夹】

        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')

        if not os.path.exists('D:\HK\TestPipeOutput_image\\' + folder_name):  # 如果文件夹不存在
            os.mkdir('D:\HK\TestPipeOutput_image\\' + folder_name)  # 创建文件夹
        plt.savefig('D:\HK\TestPipeOutput_image\\' + folder_name + '\\' +file, dpi=400)  # 【待添加：自动创建文件夹】
        # plt.show()
        print(file+"生成成功")
    except:
        print(file+"找不到四个峰值")
        continue

end = time.time()
print(f'the runing time is:{end-start} s')