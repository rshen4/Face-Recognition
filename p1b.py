import os
import argparse
import pickle
import numpy as np
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils

import matplotlib.pyplot as plt
import numpy as np
import random
import torch.optim as optim
import model2

def get_args():
    
    parser = argparse.ArgumentParser(description="This is the main test harness for my models.")
    parser.add_argument("--load", type=str, help="The model weight file to use for testing.")
    parser.add_argument("--save", type=str, help="Save the trained model weight")
    parser.add_argument("--augment", type=bool, help="use Augmented Pictures or not")

    args = parser.parse_args()

    return args


def check_args(args):

    if args.save is None and args.load is None:
        raise Exception("--You have to give at least one argument --load or --save, to choose mode of train or test")
    else:
        if args.save is None:
            return 'test'
        else:
        	return 'train'
            

def main():
    
    args = get_args()
    myTransform = None
    if args.augment == True:
        myTransform = model2.RandomTransform()
    mode = check_args(args)

    net = model2.siamese()
    dataLoader=model2.DataLoader(img_folder ='./lfw/', txt_file='./train.txt', transform = myTransform)
    net = net.cuda()
    if mode.lower() == "train":
         
        criterion = model2.ContrastiveLoss(13)
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        use_cuda = torch.cuda.is_available()
        loss_list = []
        
        for epoch in range(2):
            running_loss = 0.0
            for i in range(2200):
                images,label = dataLoader[i]
                inputImgs = Variable(images).cuda()
                target = Variable(label.view(-1,1)).cuda()
                optimizer.zero_grad()  
                output = net(inputImgs)
                loss = criterion(output.data,target.data)
                loss = Variable(loss,requires_grad=True)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if i % 2000 == 1999: 
                    print(running_loss.cpu().numpy()[0] / 2000)
                    loss_list.append(running_loss.cpu().numpy()[0] / 2000)
        if args.augment==True:
            torch.save(net.state_dict(),'p1b_AD')
            np.save('p1b_ADloss.npy',np.array(loss_list))
        else:
            torch.save(net.state_dict(),'p1b_NAD')
            np.save('p1bNADloss.npy',np.array(loss_list))
            
        accurate = 0
        for i in range(2200):
            images,label = dataLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0] > 8.5):
                predict = 0
            else:
                predict = 1
            if (target.data[0][0] - predict <0.5) and (target.data[0][0]-predict>-0.5):
                accurate = accurate + 1
        
        testLoader=model2.DataLoader(img_folder ='./lfw/', txt_file='./test.txt', transform = myTransform)
        accurate = 0
        for i in range(1000):
            images,label = testLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0] > 8.5):
                predict = 0
            else:
                predict = 1
            if (target.data[0][0] - predict <0.5) and (target.data[0][0]-predict>-0.5):
                accurate = accurate + 1

    if mode.lower() == "test":
        
        if args.augment == True:
            net.load_state_dict(torch.load('p1b_AD'))        
        else:
            net.load_state_dict(torch.load('p1b_NAD'))
            
        accurate = 0
        for i in range(2200):
            images,label = dataLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0] > 8.5):
                predict = 0
            else:
                predict = 1
            if (target.data[0][0] - predict <0.5) and (target.data[0][0]-predict>-0.5):
                accurate = accurate + 1

        testLoader=model2.DataLoader(img_folder ='./lfw/', txt_file='./test.txt', transform = myTransform)
        accurate = 0
        for i in range(1000):
            images,label = testLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0] > 8.5):
                predict = 0
            else:
                predict = 1
            if (target.data[0][0] - predict <0.5) and (target.data[0][0]-predict>-0.5):
                accurate = accurate + 1

if __name__ == "__main__":
    main()