import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
from models import *
from datasets import *
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="dataset", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

transforms_ = [transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

transform_image = transforms.Compose(transforms_)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)
generator = GeneratorUNet()

def get_test_img(file,transform_image):
	img = Image.open(file)
	w, h = img.size
	img_A = img.crop((0, 0, w/2, h))
	img_A = img_A.convert('RGB')
	img_B = img.crop((w/2, 0, w, h))
	img_B = img_B.convert('RGB')
	img_A = transform_image(img_A)
	img_B = transform_image(img_B)

	img_A = img_A.view(-1, 3, 512, 512)
	img_B = img_B.view(-1, 3, 512, 512)

	return {'img_A': img_A, 'img_B': img_B}


generator.load_state_dict(torch.load('saved_models/generator.pth', map_location=torch.device('cpu')))
print(generator)
files = glob.glob('test/*.png')
print(files)

for file in files:
    print(file)
	imgs = get_test_img(file, transform_image)
	img_A = Variable(imgs['img_A'])
	img_B = Variable(imgs['img_B'])
	fake = generator(img_A)
	img_sample = torch.cat((img_A.data, fake.data, img_B.data), -2)
	os.makedirs('test_results', exist_ok=True)
	save_image(img_sample, 'test_results/%s' % file[5:], normalize=True)
	print('%s is tested' % file)