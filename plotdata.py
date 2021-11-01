import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import os
dir = os.path.dirname(__file__)
print(dir)
data1 = scipy.io.loadmat(dir+'/amsgrad_cifar10.mat')
data2 = scipy.io.loadmat(dir+'/opt_admgrads_cifar10.mat')

x = data1['train_loss']
x1=data2['train_loss']
x=x[:,0]
x1=x1[:,0]
y=np.linspace(0,25,25)
y=y.reshape(25,1)
x=x.reshape(25,1)
x1=x1.reshape(25,1)
plt.plot(y,x,label= "AMSGrad",color='r')
plt.plot(y,x1,label= "Opt-AMSGrad",color='g')
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.legend(loc='best')
plt.savefig('./plots/Train_Loss.png', bbox_inches='tight')
plt.show()

x = data1['test_loss']
x1=data2['test_loss']
y=np.linspace(0,25,25)
y=y.reshape(25,1)
x=x.reshape(25,1)
x1=x1.reshape(25,1)
plt.plot(y,x,label= "AMSGrad",color='r')
plt.plot(y,x1,label= "Opt-AMSGrad",color='g')
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Testing Loss")
plt.legend(loc='best')
plt.savefig('./plots/Test_Loss.png', bbox_inches='tight')
plt.show()

x = data1['test_acc']
x1=data2['test_acc']
y=np.linspace(0,25,25)
y=y.reshape(25,1)
x=x.reshape(25,1)
x1=x1.reshape(25,1)
plt.plot(y,x,label= "AMSGrad",color='r')
plt.plot(y,x1,label= "Opt-AMSGrad",color='g')
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Testing Accuracy(%)")
plt.legend(loc='best')
plt.savefig('./plots/Test_Accuracy.png', bbox_inches='tight')
plt.show()

x =data1['train_acc']
x=x[:,0]
x1=data2['train_acc']
x1=x1[:,0]
y=np.linspace(0,25,25)
y=y.reshape(25,1)
x=x.reshape(25,1)
x1=x1.reshape(25,1)
plt.plot(y,x,label= "AMSGrad",color='r')
plt.plot(y,x1,label= "Opt-AMSGrad",color='g')
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy(%)")
plt.legend(loc='best')
plt.savefig('./plots/Train_Accuracy.png', bbox_inches='tight')
plt.show()