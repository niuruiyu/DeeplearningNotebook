# 开发人：牛瑞雨
# 开发时间： 2025/8/15 19:57
#加载数据
import copy
import time
from torchvision.datasets import FashionMNIST
import numpy as np
import torch
import torch.utils.data as Data
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
#导入模型
from model import AlexNet


#加载数据,处理训练集和验证集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227), transforms.ToTensor()]),
                              download=True)
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])
    train_dataloader=Data.DataLoader(dataset=train_data,batch_size=32,shuffle=True,pin_memory=True,num_workers=2)
    val_trainloader=Data.DataLoader(dataset=val_data,batch_size=32,shuffle=True,pin_memory=True,num_workers=2)
    return train_dataloader,val_trainloader
#模型训练的函数
def train_model_precess(model,train_dataloder,val_dataloder,num_epochs):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #定义损失函数:分类一般用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #将模型放到设备中
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    #初始化
    best_acc = 0.0
    #训练集列表
    train_loss_all = []
    train_acc_all = []
    #验证机列表
    val_loss_all = []
    val_acc_all = []
    #保存当前时间
    since = time.time()
#轮次训练
    for epoch in range(num_epochs):
        #设置输出
        print("epoch {}/{}".format(epoch,num_epochs-1))
        print('-'*10)

        #初始化参数
        #每轮刚开始重置
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        train_num = 0
        val_num = 0
        #一个的batch训练
        for step,(b_x,b_y) in enumerate(train_dataloder):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()
            output = model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)
            #将梯度初始化为0，防止梯度累计，每次在新的一批数据都要重置梯度
            optimizer.zero_grad()

            #反向传播
            loss.backward()
            #利用梯度下降法更新模型参数
            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)
            #如果预测正确，准确度加1
            train_corrects+=torch.sum(pre_lab == b_y.data)
            train_num+=b_x.size(0)
        for step,(b_x,b_y) in enumerate(val_dataloder):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab=torch.argmax(output,dim=1)
            loss=criterion(output,b_y)


            val_loss+=loss.item()*b_x.size(0)
            #如果预测正确，准确度加1
            val_corrects+=torch.sum(pre_lab == b_y.data)
            val_num+=b_x.size(0)
        #计算这个轮次的平均loss值
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))

#保存最优模型（验证精度越高越好，而不是训练精度）
        if val_acc_all[-1] > best_acc:
            best_acc=val_acc_all[-1]
            #保存最优的参数
            best_model_wts = copy.deepcopy(model.state_dict())
        #计算耗时
        time_use=time.time()-since
        print("训练耗费的时间：{:.0f}m{:.0f}s".format(time_use//60,time_use%60))
        #保存最优参数对应的模型

    torch.save(best_model_wts,'./best_model.pth')
    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})
    return train_process


#定义画图的过程
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()

if __name__=="__main__":
    AlexNet=AlexNet()
    #之后再用别的神经网络，直接导入别的像是alxenet等等就可以，训练过程都没变
    train_dataloader,val_dataloader=train_val_data_process()
    train_process=train_model_precess(AlexNet, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)










