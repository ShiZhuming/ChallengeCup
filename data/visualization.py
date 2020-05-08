# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import os
import multiprocessing as mp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from colour import Color


# %%
class Network(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Network, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.out(x)
        x = F.sigmoid(x)
        x = 100*x
        return x


# %%
fig = np.load("nancatch400fig.np.npy")


# %%
print(np.shape(fig))


# %%
# 初始化模型
filestr='./modelV'
network = torch.load(filestr)


# %%
Zero = network(torch.FloatTensor(np.zeros(21))).detach().numpy()


# %%
print(Zero)


# %%
eps = 0.00002


# %%
fig = fig.reshape((8,900*1800))
for i in range(8):
# for i in [1]:
    print(fig[i][:5])
    maxx = np.max(fig[i])
    print(maxx)
    for j in range(900*1800):
        if fig[i][j] > Zero[i] - eps and fig[i][j] < Zero[i] + eps:
            fig[i][j] = 0
        else :
            fig[i][j] = fig[i][j]*100/maxx
    print(fig[i][:5])
fig = fig.reshape((8,900,1800))


# %%
density_range = 100
img_width = 1800
img_height = 900
cmap = plt.get_cmap("rainbow") # 使用matplotlib获取颜色梯度
blue = Color("blue") # 使用Color来生成颜色梯度
hex_colors = list(blue.range_to(Color("red"), density_range))
rgb_colors = [[rgb * 255 for rgb in color.rgb] for color in hex_colors][::-1]


# %%
# 滤波轮次
for k in range(10,100,10):
    img = fig.reshape((8*900*1800))
    # 全图形滤波
    for i in range(8*900*1800):
        if img[i] >= k:
            img[i] = 99
        else:
            img[i] = img[i]*99/k
    img = img.reshape((8,900,1800))
    # 每个图形保存一张图
    for i in range(8):
        plt.imsave('/mnt/c/Users/shizh/Downloads/img'+str(i)+'dezore'+str(k)+'.png',img[i])
        frame = cv2.imread('surface2.jpg')
        if frame is None:
            print('open file error!')
            continue
        color_map = np.empty([img_height, img_width, 3], dtype=int)
        for h in range(img_height):
            for w in range(img_width):
                for j in range(3):
                    color_map[h][w][j] = rgb_colors[int(img[i][h][w])][j]
        cv2.imwrite('colormap'+str(i)+str(k)+'.png',color_map)
        heatmap = cv2.imread('colormap'+str(i)+str(k)+'.png')
        if heatmap is None:
            print('open file 2 error!')
            continue
        cv2.flip(heatmap,0 )
        alpha = 0.5 # 设置覆盖图片的透明度
        cv2.addWeighted(heatmap, alpha, frame, 1-alpha, 0, frame) # 将热度图覆盖到原图
        cv2.imwrite('/mnt/c/Users/shizh/Downloads/colorimg'+str(i)+'dezore'+str(k)+'.png',frame)
        print('finished (',i,',',k,')\n')


# %%


