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
import model1

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
        myTransform = model1.RandomTransform()
    mode = check_args(args)
    
    dataLoader=model1.DataLoader(img_folder ='./lfw/', txt_file='./train.txt', transform = myTransform)
    if mode.lower() == "train":
         
        net = model1.siamese()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        use_cuda = torch.cuda.is_available()

        net = net.cuda()

        loss_list = []
        for epoch in range(2):
            running_loss = 0.0
            for i in range(2200):
                images,label = dataLoader[i]
                inputImgs = Variable(images).cuda()
                target = Variable(label.view(-1,1)).cuda()
                optimizer.zero_grad()  
                output = net(inputImgs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if i % 2000 == 1999: 
                  #  print(running_loss / 2000)
                    loss_list.append(float(running_loss) / 2000)
        if args.augment==True:
            torch.save(net.state_dict(),'p1a_AD')
            np.save('p1a_ADloss.npy',np.array(loss_list))
        else:
            torch.save(net.state_dict(),'p1a_NAD')
            np.save('p1a_NADloss.npy',np.array(loss_list))
            
        accurate = 0
        for i in range(2200):
            images,label = dataLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0]-target.data[0][0] <0.5) and (output.data[0][0]-target.data[0][0]>-0.5):
                accurate = accurate + 1
        #print("train set accuracy")
        #print(accurate/2200.0)
        
        testLoader=model1.DataLoader(img_folder ='./lfw/', txt_file='./test.txt', transform = myTransform)
        accurate = 0
        for i in range(1000):
            images,label = testLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0]-target.data[0][0] <0.5) and (output.data[0][0]-target.data[0][0]>-0.5):
                accurate = accurate + 1
        #print("test set accuracy")
        #print(accurate/1000.0)

    if mode.lower() == "test":
        net = model1.siamese()
        if args.augment == True:
            net.load_state_dict(torch.load('p1a_AD'))        
        else:
            net.load_state_dict(torch.load('p1a_NAD'))
        	
        accurate = 0
        net = net.cuda()
        for i in range(2200):
            images,label = dataLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0]-target.data[0][0] <0.5) and (output.data[0][0]-target.data[0][0]>-0.5):
                accurate = accurate + 1
        #print("train set accuracy")
        #print(accurate/2200.0)

        testLoader=model1.DataLoader(img_folder ='./lfw/', txt_file='./test.txt', transform = myTransform)
        accurate = 0
        for i in range(1000):
            images,label = testLoader[i]
            inputImgs = Variable(images).cuda()
            target = Variable(label.view(-1,1)).cuda()
            output = net(inputImgs)
            if (output.data[0][0]-target.data[0][0] <0.5) and (output.data[0][0]-target.data[0][0]>-0.5):
                accurate = accurate + 1
        #print("test set accuracy")
        #print(accurate/1000.0)

if __name__ == "__main__":
    main()