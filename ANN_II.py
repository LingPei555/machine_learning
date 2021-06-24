# Fig8 S
from __future__ import print_function

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms

from torch.autograd import Variable

import numpy as np

global params

from custom_data import custom_dset_II, collate_fn

from numpy.linalg import cholesky
from random import choice
import math as m


# 5-100-5000-5000-100-6
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(5,100 ) #30
        
        self.fc2 = nn.Linear(100, 5000)
        
        self.fc3 = nn.Linear(5000, 5000)
        
        #self.fc4 = nn.Linear(5000, 5000)
        
        self.fc4 = nn.Linear(5000, 100)
        
        self.fc5 = nn.Linear(100, 2)# L Q, exclude S
        
        
    
    def forward(self, x):

        x = x.view(-1, 5)
        
        x = F.tanh(self.fc1(x))
        
        x = F.tanh(self.fc2(x))
        
        x = F.tanh(self.fc3(x))
        
        #x = F.tanh(self.fc4(x))
        
        x = F.tanh(self.fc4(x))
        
        x = self.fc5(x)
        
        return x 
    

    
def train(args,model, device, train_loader, optimizer, epoch):

    model.train()


    for batch_idx, (data, target) in enumerate(train_loader):
        
        data = torch.Tensor(data)
            
        target = torch.Tensor(target)
        
        data, target = data.to(device), target.to(device)  
        
        optimizer.zero_grad()
        
        output = model(data)

        loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        loss = loss_fn(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))


    
def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data = torch.Tensor(data)
            
            target = torch.Tensor(target)
          
            data, target = data.to(device), target.to(device)  
            
            output = model(data)
            
            loss_fn = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

            test_loss += loss_fn(output, target).item()
            
            #********* modify the following  ***********#
            
            out=output.cpu().detach().numpy()
            tar=target.cpu().detach().numpy()               
            
            #'''
            for i in abs(output-target)/abs(target):
                if max(i[0:2])<0.1:
                    correct+=1
            #'''
            #********* modify the above  ***********#
 
    test_loss /= len(test_loader.dataset)
     
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def main():
        
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',

                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=23640, metavar='N',#1000

                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=1000, metavar='N',

                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.000001, metavar='LR',

                        help='learning rate (default: 0.001)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',

                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,

                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',

                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',

                        help='how many batches to wait before logging training status')

    

    parser.add_argument('--save-model', action='store_true', default=False,

                        help='For Saving the current Model')

    args = parser.parse_args()

    use_cuda = 1#not args.no_cuda and torch.cuda.is_available()



    torch.manual_seed(args.seed)



    device = torch.device("cuda" if use_cuda else "cpu")  # I change cuda to cpu use_cuda



    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    #************ Predict Sparameters L Q **************#
    
    train_loader = torch.utils.data.DataLoader(custom_dset_II('./ANN_II_Data_Freq', './ANN_II_Data_Freq/train_0.5.txt'),batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,**kwargs)

    test_loader = torch.utils.data.DataLoader(custom_dset_II('./ANN_II_Data_Freq', './ANN_II_Data_Freq/test_0.5.txt'),batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn,**kwargs)
    
    model = Net()
    model.load_state_dict(torch.load("ann_II_FullFreq_LQ_0.5.pt"))
    model.to(device) 
    
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)
        if not epoch%10:
            print('------------------Saving------------------')
            torch.save(model.state_dict(),"ann_II_FullFreq_LQ_0.5.pt")
        if np.round(test(args, model, device, test_loader),2) >99.9:
            break
        
        


    

if __name__ == '__main__':

    main()