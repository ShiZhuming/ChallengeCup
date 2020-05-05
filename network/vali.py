import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import numpy as np
import torch.optim as optim
import torch.utils.data as D
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

spec = []
res = []

def Curve_Fitting(x,y,deg):
    parameter = np.polyfit(x, y, deg)    #拟合deg次多项式
    p = np.poly1d(parameter)             #拟合deg次多项式
    aa=''                               #方程拼接  ——————————————————
    for i in range(deg+1): 
        bb=round(parameter[i],2)
        if bb>0:
            if i==0:
                bb=str(bb)
            else:
                bb='+'+str(bb)
        else:
            bb=str(bb)
        if deg==i:
            aa=aa+bb
        else:
            aa=aa+bb+'x^'+str(deg-i)    #方程拼接  ——————————————————
    plt.scatter(x, y)     #原始数据散点图
    plt.plot(x, p(x), color='g')  # 画拟合曲线
    plt.legend([aa,round(np.corrcoef(y, p(x))[0,1]**2,2)])   #拼接好的方程和R方放到图例
    plt.show()


inReader = csv.reader(open("/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/input_test.csv", "r"))
for i in inReader:
    tmp = [float(j) for j in i]
    spec.append(tmp)

outReader = csv.reader(open("/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/output_test.csv", "r"))
for i in outReader:
    tmp = [float(j) for j in i]
    res.append(tmp)

spec = torch.FloatTensor(spec)
res = torch.FloatTensor(res)


class Network(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Network, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.out(x)
        return x

filestr='/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/model'
network = torch.load(filestr)

predy=network(spec).detach().numpy()
trainy=res.numpy()


for i in range(8):
    #plt.scatter([item[i] for item in predy], [item[i] for item in trainy])
    Curve_Fitting([item[i] for item in predy],[item[i] for item in trainy],1)