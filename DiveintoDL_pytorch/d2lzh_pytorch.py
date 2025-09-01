from matplotlib_inline import backend_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import random
from torch import nn
def use_svg_display():  
    backend_inline.set_matplotlib_formats('svg')  

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)#打乱样本
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)
def linreg(X,w,b):
    return torch.mm(X,w)+b
def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2
    #这里返回的是y_hat大小的向量，pytorch的MSEloss没有除以2
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
def evaluate_accuracy(data_iter,net):
    acc_sum ,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1)==y).float().sum()
        n += y.shape[0]#获取当前批次的样本数
    return acc_sum/n

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


#训练的函数
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr =None,optimizer=None):
    for epoch in range(num_epochs):
        train_loss_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum()
            #先梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            #反向传播
            l.backward()
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()
            train_loss_sum+=l
            train_acc_sum+=(y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d,loss%.4f,train acc %.3f,test acc %.3f'
              %(epoch+1,train_loss_sum/n,train_acc_sum/n,test_acc))


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)