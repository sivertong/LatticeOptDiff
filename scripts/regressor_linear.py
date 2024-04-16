import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch
import torch.nn as nn
import torch.utils.data as Data
import io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 现读取函数
data_structure = np.load('Output_12480.npy')  # design variables
data_label = np.load('Input_12480.npy')  # material properties

#当成一个数据集
data_structure = torch.Tensor(data_structure).float()
data_label = torch.Tensor(data_label[:,1:3]).float()
dataset = Data.TensorDataset(data_structure, data_label)

#自定义神经网络
class RegressorLinear(nn.Module):
    def __init__(self):
        super(RegressorLinear, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(11, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 2),
            ]
        )


    def forward(self, x):
        #         x = x_0
        for idx, linear_layer in enumerate(self.linears):
            x = linear_layer(x)
        return x




print('Training model...')
batch_size = 256
# dataset放到dataloader中
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 迭代周期
num_epoch = 1000
plt.rc('text', color='blue')
#实例化模型，传入一个数
model = RegressorLinear().to(device)  # 输入维度是11
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
for t in range(num_epoch):
    # dataloader遍历
    for idx, (batch_x, batch_y) in enumerate(dataloader):


        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 送入模型，得到t时刻的随机噪声预测值
        output = model(batch_x)

        # 计算误差
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(batch_y, output)
        optimizer.zero_grad()
        loss.backward()
        # 梯度clip，保持稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()


    # 每100步打印效果
    if (t % 10 == 0):
        print(loss)


