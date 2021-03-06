import matplotlib.pyplot as plt
import os.path
import sys
import numpy as np

epoch="epoch"
g_loss="g_loss"
d_loss="d_loss"
d_loss_real="d_loss_real"
d_loss_fake="d_loss_fake"
d_loss_fake_for_G="d_loss_fake_for_G"
rate_m="rate_M"
rate_d="rate_D"
alpha_G="rate"
step_num=1
EPOCH=True
val_G=5000
class Data:
    def __init__(self,reader,val_G):
        self.epoch=0
        self.val_G=val_G
        self.reader=reader
        self.g_loss=None
        self.d_loss=None
        self.d_loss_real=None
        self.d_loss_fake=None
        self.d_loss_fake_for_G=None
        self.rate_m=None
        self.rate_d=None
        self.alpha_G=None
        self.first=self.reader.readline()
        self.stop=False
        if self.first:
            self.start()
        else:
            self.stop=True

    def mostfast(self):
        if "val_g" in self.first:
            self.val_G=float(self.first.strip().split(" ")[1])
            self.first=self.reader.readline()

    def start(self):
        self.mostfast()
        self.epoch=int(self.first.split(" ")[0])
        line=self.reader.readline().strip().split("\t")
        for i in range(len(line)):
            target=line[i].split(" ")
            name=target[0]
            value=float(target[1])
            if name==g_loss:
                self.g_loss=value
            elif name==d_loss:
                self.d_loss=value
            elif name==d_loss_real:
                self.d_loss_real=value
            elif name==d_loss_fake:
                self.d_loss_fake=value
            elif name==d_loss_fake_for_G:
                self.d_loss_fake_for_G=value
            elif name==rate_m:
                self.rate_m=value
            elif name==rate_d:
                self.rate_d=value
            elif name==alpha_G:
                self.alpha_G=value
            else:
                print("error occurred with {}".format(name))

def make(name):
    val_G=5000
    f=open(name,"r")
    datas=[]
    flag=True
    while flag:
        new=Data(f,val_G)
        val_G=new.val_G
        flag= not new.stop
        if flag:
            datas.append(new)
    f.close()
    result=recognizer(datas)
    result=[np.array(result[i]) for i in range(len(result))]
    if EPOCH:
        for i in range(len(result)):
            if i<len(result)//2:
                result[i]=result[i]/step_num
    plt.subplot(231)
    plt.plot(result[0],result[9],"g")
    plt.title("g_loss")
    plt.subplot(233)
    plt.plot(result[4],result[13],"g")
    plt.title("d_loss_fake_for_G")
    plt.subplot(232)
    plt.plot(result[5],result[14],"g")
    plt.title("MSE_loss")
    plt.subplot(234)
    plt.plot(result[1],result[10])
    plt.title("d_loss")
    plt.subplot(235)
    plt.plot(result[2],result[11])
    plt.title("d_loss_real")
    plt.subplot(236)
    plt.plot(result[3],result[12])
    plt.title("d_loss_fake")
    """
    plt.subplot(254)
    plt.plot(result[6],result[15],"r")
    plt.title("gradient_M")
    plt.subplot(258)
    plt.plot(result[7],result[16],"r")
    plt.title("gradient_D")
    plt.subplot(259)
    plt.plot(result[8],result[17],"r")
    plt.title("gradient_rate")
    """

def recognizer(datas):
    epoch_g=[]
    g=[]
    epoch_d=[]
    d=[]
    epoch_d_r=[]
    d_r=[]
    epoch_d_f=[]
    d_f=[]
    epoch_d_fG=[]
    d_fG=[]
    epoch_MSE=[]
    g_MSE=[]
    epoch_rate_m=[]
    rate_m=[]
    epoch_rate_d=[]
    rate_d=[]
    epoch_alpha=[]
    alpha=[]
    for data in datas:
        if data.g_loss:
            epoch_g.append(data.epoch)
            g.append(data.g_loss)
        if data.d_loss:
            epoch_d.append(data.epoch)
            d.append(data.d_loss)
        if data.d_loss_real:
            epoch_d_r.append(data.epoch)
            d_r.append(data.d_loss_real)
        if data.d_loss_fake:
            epoch_d_f.append(data.epoch)
            d_f.append(data.d_loss_fake)
        if data.rate_m:
            epoch_rate_m.append(data.epoch)
            rate_m.append(data.rate_m)
        if data.rate_d:
            epoch_rate_d.append(data.epoch)
            rate_d.append(data.rate_d)
        if data.alpha_G:
            epoch_alpha.append(data.epoch)
            alpha.append(data.alpha_G)
        if data.d_loss_fake_for_G:
            epoch_d_fG.append(data.epoch)
            d_fG.append(data.d_loss_fake_for_G)
            if data.g_loss:
                epoch_MSE.append(data.epoch)
                g_MSE.append(data.g_loss - data.val_G*data.d_loss_fake_for_G)
    return [epoch_g,epoch_d,epoch_d_r,epoch_d_f,epoch_d_fG,epoch_MSE,epoch_rate_m,epoch_rate_d,epoch_alpha,g,d,d_r,d_f,d_fG,g_MSE,rate_m,rate_d,alpha]

def main(name):
    flag=True
    while flag:
        if not os.path.exists(name):
            print("no epochlog")
            sys.exit
        make(name)
        plt.show()
        flag=False

if __name__=="__main__":
    main("epoch_log.txt")

