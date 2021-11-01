'''Some helper functions for PyTorch, including:'''
import os
import sys
import torch.nn.init as init
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import scipy.io as sio
import time
import os
import argparse

from torch.autograd import Variable

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
'''ResNet18'''

class BasicBlock(nn.Module):
    expansion = 1



    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes = num_classes)

def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes = num_classes)

def ResNet50(num_classes = 10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes)

def ResNet101(num_classes = 10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes)

def ResNet152(num_classes = 10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes = num_classes)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()

class Optadam_torch(Optimizer):

    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, span=5):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, span=span)
        super(Optadam_torch, self).__init__(params, defaults)

    def step(self, optimizer_aux):

        loss = None
        for (group, group_aux) in zip(self.param_groups, optimizer_aux.param_groups):
            for (p, q) in zip(group['params'], group_aux['params']):
                # print (p)
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                span = group['span']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).cuda()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).cuda()
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).cuda()
                    # Add More State Varialbes for Opt
                    a_size = p.data.size();
                    t = 1;
                    for i in range(len(a_size)):
                        t = t * a_size[i]
                    state['prev_predict_grad'] = torch.zeros_like(p.data).cuda()
                    state['w_ma'] = torch.zeros([span, t], dtype=torch.float64).cuda()
                    state['w_his'] = torch.zeros([span, t], dtype=torch.float64).cuda()
                    state['prev_w'] = p.data
                    state['aux_w'] = p.data
                ##
                aux_w = state['aux_w']
                a_size = p.data.size();
                t = 1;
                for i in range(len(a_size)):
                    t = t * a_size[i]
                prev_w, w_ma, w_his = state['prev_w'], state['w_ma'], state['w_his']
                prev_predict_grad = state['prev_predict_grad']

                w_diff = p.data - prev_w
                w_diff = torch.reshape(w_diff, (1, t))

                if (state['step'] >= 1 and state['step'] <= span):
                    w_ma[state['step'] - 1, :] = w_diff
                if (state['step'] > span):
                    w_ma[:-1, :] = w_ma[1:, :].clone()
                    w_ma[-1, :] = w_diff

                if (state['step'] < span):
                    w_his[state['step'], :] = torch.reshape(p.data, (1, t)).clone()
                else:
                    w_his[:-1, :] = w_his[1:, :].clone()
                    w_his[-1, :] = torch.reshape(p.data, (1, t)).clone()

                wtmp = torch.zeros_like(p.data)
                if (state['step'] >= span):
                    la = torch.mm(w_ma, w_ma.t())
                    la = torch.add(la, 0.001 * torch.eye(span, dtype=torch.float64).cuda())
                    lb = torch.ones([span, 1], dtype=torch.float64).cuda()
                    x, LU = torch.solve(lb, la)
                    x = x / sum(x)
                    wtmp = torch.mm(w_his.t(), x)
                    wtmp = torch.reshape(wtmp, a_size)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1)
                prev_predict_grad = q.grad.data  ###
                tmp_exp = torch.add(exp_avg, (1 - beta1), prev_predict_grad)
                exp_avg.add_(1 - beta1, grad)

                tmp = prev_predict_grad - grad  ###
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, tmp, tmp)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                aux_w.addcdiv_(-step_size, exp_avg, denom)
                prev_w = p.data  ###
                p.data = torch.addcdiv(aux_w, -step_size, tmp_exp.float(), denom)
                q.data = wtmp.float()  ###
        return loss

'''Train CIFAR10 with PyTorch.'''

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-f')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--logfile', default='Foo', type=str, help='filename of log file')
parser.add_argument('--span', default=5, type=int, help='number of previous gradients used for prediction')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--epochs', type=int, default=25, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')




args = parser.parse_args()
print (args.logfile)
betas = (args.beta1, args.beta2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./drive/My Drive/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./drive/My Drive/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18(num_classes = 10)
net = net.to(device)
net_aux = ResNet18(num_classes = 10)
net_aux = net_aux.to(device)
print(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    net_aux = torch.nn.DataParallel(net_aux)
    cudnn.benchmark = True

#if args.resume:
print('==> Resuming from checkpoint..')
assert os.path.isdir('./drive/My Drive/checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./drive/My Drive/checkpoint/ckpt2.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(start_epoch)

criterion = nn.CrossEntropyLoss()
optimizer     = Optadam_torch(net.parameters(), lr=args.lr, span = args.span, weight_decay = args.wd, betas = betas)
optimizer_aux = optim.SGD(net_aux.parameters(), lr=args.lr)

# Training
def train(epoch, trloss_rec, tracc_rec, time_rec, t0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    trloss_rec_aux = []
    tracc_rec_aux  = []
    time_rec_aux   = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        net_aux.to(device)
        optimizer_aux.zero_grad()
        output_aux = net_aux(inputs)
        loss_aux   = F.nll_loss(output_aux, targets)
        loss_aux.backward()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step(optimizer_aux)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if batch_idx % args.log_interval == 0:
            trloss_rec_aux.append( train_loss/(batch_idx+1) )
            tracc_rec_aux.append( 100.*correct/total )
            time_rec_aux.append( time.time()-t0 )

    trloss_rec.append( trloss_rec_aux )
    tracc_rec.append( tracc_rec_aux )
    time_rec.append( time_rec_aux )
    print(tracc_rec)

def test(epoch, tsloss_rec, tsacc_rec):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            num += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    tsloss_rec.append( test_loss/(num+1) )
    tsacc_rec.append( acc )

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./drive/My Drive/checkpoint'):
            os.mkdir('./drive/My Drive/checkpoint')
        torch.save(state, './drive/My Drive/checkpoint/ckpt2.pth')
        best_acc = acc


trloss_rec = []
tracc_rec  = []
time_rec   = []

tsloss_rec = []
tsacc_rec  = []

t0 = time.time()

for epoch in range(start_epoch, args.epochs):
    train(epoch, trloss_rec, tracc_rec, time_rec, t0)
    test(epoch, tsloss_rec, tsacc_rec)

sio.savemat(args.logfile, {'train_loss': trloss_rec,'train_acc':tracc_rec,'time_rec':time_rec,'test_loss':tsloss_rec,'test_acc':tsacc_rec})
