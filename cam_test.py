#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/15 13:45
# @Author  : Wslsdx
# @FileName: cam_test.py
# @Software: PyCharm
# @Github  ：https://github.com/Wslsdx
from torchvision.models import  resnet,vgg
from grad_cam import GradCam
from utils import cv_imread,show_cams_on_images,preprocess_image
import torch
import cv2
import  numpy as np
import os

if __name__=="__main__":

	image_dir="./photos/"
	image_name="both.png"
	image=cv2.resize(cv_imread(image_dir+image_name), (224, 224))
	image_arr =np.array(image).astype(np.float)/255.0
	img_batch=preprocess_image(image)

	device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
	model=vgg.vgg19(pretrained=True).to(device)
	#model = resnet.resnet34(pretrained=True).to(device)
	print(model)
	gcam = GradCam(model, device, activation="sigmoid")
	input = torch.from_numpy(img_batch).float().to(device)

	#target_layer_names = ["features.30""features.32","features.34"]
	target_layer_names = ["features.32"]
	#添加钩子
	gcam.register_hooks(target_layer_names)
	#计算cam
	cams = gcam(input)
	#移除钩子
	gcam.remove_hook_handles()

	#可视化
	for idx,cam_batch,layer_name  in  zip( range(len(cams)),cams,target_layer_names):

		img_batch=image_arr[np.newaxis,:,:,:]
		cam_batch=cam_batch.detach().cpu().numpy()
		save_name=layer_name+"_"+image_name
		name_batch=[save_name]
		show_cams_on_images(img_batch,cam_batch,name_batch,os.path.abspath(image_dir))

