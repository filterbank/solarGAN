#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 10:40:39 2018

@author: Yi Huang
"""

from MeasureFunction import Measure
from MeasureFunction import Load_img
import numpy as np
import argparse
import os
import json

'''---------------------------Perceptual evaluation-------------------------------------'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perception-Evaluation')

    parser.add_argument('--path', type=str, metavar='img_dir',default='',
                        help='input image  dir')

    args = parser.parse_args()

    measure = Measure()
    load_img = Load_img()

    orig_dir=os.path.join(args.path,'Orig')
    blur_dir=os.path.join(args.path,'blur-x')
    clean_dir=os.path.join(args.path,'clean-x')
    if not os.path.exists(orig_dir):
        print('error input orig dir: '+args.ref)
        exit(1)
    if not os.path.exists(blur_dir):
        print('error input blur dir: '+args.ref)
        exit(1)
    if not os.path.exists(clean_dir):
        print('error input clean dir: '+args.ref)
        exit(1)

    inputOrigs=[]

    for root, dirs, files in os.walk(orig_dir, topdown=True):
        for f in files:
            if f.split('.')[-1]=='png':
                fName=os.path.join(root,f)
                inputOrigs.append(fName)
                # print(inputFiles[-1])
    print('find '+inputOrigs.__len__().__str__()+' input orig files')

    rst=[]

    for f in inputOrigs:
        try:
            origF=f
            blurF=os.path.join(blur_dir,os.path.split(f)[-1])
            cleanF=os.path.join(clean_dir,os.path.split(f)[-1])
            ref_img = load_img.read_png(f)[:,:,0]
            #ref_img = ref_img[384:639,384:639]
            #ref_img = ref_img.resize(512,512)
            ref_img = load_img.norm(ref_img)
            blur_img = load_img.read_png(blurF)[:,:,0]
            #blur_img = blur_img.resize(512,512)
            clean_img= load_img.read_png(cleanF)[:,:,0]
            cmpF = [blur_img,clean_img]
            r=[os.path.split(f)[-1]]
            for observe_img in cmpF:
                size = 256 - 1
                temp = []
                for i in range(0, 16):
                    for j in range(0, 16):
                        patch = observe_img[i * 16:i * 16 + size, j * 16:j * 16 + size]
                        patch = load_img.norm(patch)
                        l = measure.Preception(ref_img, patch)
                        #print(l)
                        temp.append(l)
                PE = float(np.mean(temp))
                print(PE)
                r.append(PE)
            print(f)
            #print(f+' : '+rst[1].__str__()+' \t '+rst[2].__str__())
            rst.append(r)
        except Exception as err:
            print('error occur in processing file : ' + f)
            print(err.__str__())
    json.dump(rst,open('rst.txt','w'))

