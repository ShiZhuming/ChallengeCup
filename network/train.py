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

inReader = csv.reader(open("/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/input_update.csv", "r"))
for i in inReader:
    tmp = [float(j) for j in i]
    spec.append(tmp)

outReader = csv.reader(open("/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/output_update.csv", "r"))
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


network = Network(21, 20, 8)

BATCH_SIZE = 66
TRAIN_TIMES = 50000
losses = []
optimizer = optim.Adam(network.parameters(), lr=0.001)

datasets = D.TensorDataset(spec, res)

loader = D.DataLoader(
    dataset=datasets,
    batch_size=BATCH_SIZE,
    shuffle=True,

)

for epoch in range(TRAIN_TIMES):
    for batch in loader:
        specs, ress = batch

        preds = network(specs)
        loss = F.mse_loss(preds,ress)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if(epoch % 100 == 0):
        print("epoch: ",epoch," loss: ",losses[-1])
    
plt.plot(losses)
plt.show()

savestr='/Volumes/AnyShare/GroupDocuments.localized/ChallengeCup2020/ChallengeCup-master/network/model'
torch.save(network,savestr)

predy=network(spec).detach().numpy()
trainy=res.numpy()
plt.ion()
for i in range(8):
    plt.scatter([item[i] for item in predy], [item[i] for item in trainy])
    plt.pause(10)
    plt.cla()