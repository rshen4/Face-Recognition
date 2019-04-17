import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
import random 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from PIL import Image

class siamese(nn.Module):

    def __init__(self):
        super(siamese, self).__init__()
        # 1. Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, 5, (1,1),  2)
        # 3. Batch Normalization Layer
        self.BN1 = nn.BatchNorm2d(64)
        # 5. Convolution Layer
        self.conv2 = nn.Conv2d(64, 128, 5, (1,1), 2)
        # 7. Batch Normalization Layer
        self.BN2 = nn.BatchNorm2d(128)
        # 9. Convolution Layer
        self.conv3 = nn.Conv2d(128, 256, 3, (1,1), 1)
        # 11. Batch Normalization Layer
        self.BN3 = nn.BatchNorm2d(256)
        # 13. Convolution Layer
        self.conv4 = nn.Conv2d(256, 512, 3, (1,1), 1)
        # 15. Batch Normalization Layer
        self.BN4 = nn.BatchNorm2d(512)
        # 17. Fully Connected Layer
        self.fc1 = nn.Linear(131072, 1024)
        # 19. Batch Normalization Layer
        self.BN5 = nn.BatchNorm1d(1024)
        # 20. Fully Connected Layer
        self.fc2 = nn.Linear(2048,1)

    def forward(self, x):
        # go through (1. Convolution Layer), and get activation by relu (2. ReLU)
        x = F.relu(self.conv1(x))
        # go through (3. Batch Normalization Layer), and use (4. Max Pooling Layer) to filter x
        x = F.max_pool2d(self.BN1(x), (2, 2))
        
        # go through (5. Convolution Layer), and get activation by relu (6. ReLU)
        x = F.relu(self.conv2(x))
        # go through (7. Batch Normalization Layer), and use (8. Max Pooling Layer) to filter x
        x = F.max_pool2d(self.BN2(x), (2, 2))
        
        # go through (9. Convolution Layer), and get activation by relu (10. ReLU)
        x = F.relu(self.conv3(x))
        # go through (11. Batch Normalization Layer), and use (12. Max Pooling Layer) to filter x
        x = F.max_pool2d(self.BN3(x), (2, 2))
        
        # go through (13. Convolution Layer), and get activation by relu (14. ReLU)
        x = F.relu(self.conv4(x))
        # go through (15. Batch Normalization Layer)
        x = self.BN4(x)
        
        # go through (16. Flatten Layer)
        x = x.view(-1, self.num_flat_features(x))
        
        # go through (17. fully connected layer), and et activation by relu (18. ReLU)
        x = F.relu(self.fc1(x))
        x1,x2=torch.chunk(x,2,0)
        distance = nn.PairwiseDistance(p=2)
        x = distance(x1,x2)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class RandomTransform(object):
    """Randomly transform the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        self.flip = transforms.RandomHorizontalFlip()

    def __call__(self, image):
        # randomly rotate the image
        if (random.random()<0.7):
            image = image.rotate(random.randint(-30,30))
        # randomly scale the image 
        if (random.random()<0.7):
            scale_rate = random.uniform(0.7,1.3)
            # 250 is the initial size of the picture, which is already checked in my test code
            new_size = int(scale_rate*250)
            scale = transforms.CenterCrop(new_size)
            image = scale(image)
        # randomly flip the image, the RandomHorizontalFlip itself has a probability of 0.5
            image = self.flip(image)
        # random image translation
        if (random.random()<0.7):
            x_translation = random.randint(-10,10) 
            y_translation = random.randint(-10,10)
           
            image = image.transform(image.size, Image.AFFINE, (1, 0, x_translation, 0, 1, y_translation))
            
        return image


class DataLoader(Dataset):
    def __init__(self, img_folder, txt_file, transform = None):
        self.img_folder = img_folder
        self.transform = transform
        self.data_list=[]
        self.transform = transform
        self.rescale = torchvision.transforms.Scale(128)
        self.toTensor = torchvision.transforms.ToTensor()
        with open(txt_file) as reader:
            for example_index, line in enumerate(reader):
                if len(line.strip()) == 0:
                    continue
                # Divide the line into images path and label.
                split_line = line.split(" ")
                self.data_list.append(split_line)
                
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,index):
        img0_path = os.path.join(self.img_folder,self.data_list[index][0])
        img1_path = os.path.join(self.img_folder,self.data_list[index][1])
        
        label = torch.from_numpy(np.array([float(self.data_list[index][2])])).float()
        
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        # if the parameter is not given in the initialization, then skip this
        if (random.random()<0.7): 
            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)
        
        img0 = self.rescale(img0)
        img1 = self.rescale(img1)
        
        img0 = self.toTensor(img0)
        img1 = self.toTensor(img1)
        
        img0 = img0.numpy()
        img1 = img1.numpy()
        
        img0 = np.expand_dims(img0,axis=0)
        img1 = np.expand_dims(img1,axis=0)
        # here to add transform
        IMG = np.concatenate((img0,img1),axis=0)
        
        IMG = torch.from_numpy(IMG)
        
        return IMG,label

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,euclidean_distance,label):
        loss_contrastive = label * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0,max=1000), 2)

        return loss_contrastive