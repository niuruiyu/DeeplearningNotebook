from matplotlib_inline import backend_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import random
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