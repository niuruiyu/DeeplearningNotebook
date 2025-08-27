# 开发人：牛瑞雨
# 开发时间： 2025/8/18 16:35
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import  FashionMNIST
from model import GoogLeNet,Inception
#处理数据
def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)
    test_dataloader=Data.DataLoader(dataset=test_data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=0)
    return test_dataloader
#测试过程
def test_model_precess(model,test_dataloader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(device)
    #把模型放到设备中
    model = model.to(device)
    #设置初始化参数
    test_corrects=0.0
    test_num=0
    #只进行前向传播，不进行梯度计算
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            predict = torch.argmax(output,dim=1)
            test_corrects += torch.sum(predict == test_data_y.data)
            test_num += test_data_x.size(0)

    test_acc = test_corrects.double().item()/test_num
    print("该模型测试的精度为：",test_acc)

if __name__ == '__main__':
    #加载模型
    model = GoogLeNet(Inception)
    model.load_state_dict((torch.load('best_model.pth')))
    test_data_loader = test_data_process()
    # test_model_precess(model,test_data_loader)
    #推理
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(device)
    with torch.no_grad():
        for b_x,b_y in test_data_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为验证模式
            model.eval()
            output = model(b_x)
            predict = torch.argmax(output,dim=1)
            #取出数值
            result = predict.item()
            label = b_y.item()
            print("预测值为:",result,"------","真实值:",label)


