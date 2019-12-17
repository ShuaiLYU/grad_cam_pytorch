#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/15 13:45
# @Author  : Wslsdx
# @FileName: cam_test.py
# @Software: PyCharm
# @Github  ：https://github.com/Wslsdx
import  os
import cv2
import numpy as np

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	return preprocessed_img[np.newaxis,:,:,:]


def show_cams_on_images(img_batch, mask_batch,filenames,save_dirs):
	"""
	:param img_batch:   [B,H,W,C] C==3
	:param mask_batch:     [B,H1,W1,1]
	:param filenames:    list[filename]
	:param save_dirs:    dir of save
	:return:
	"""
	if img_batch.ndim!=4:img_batch=img_batch.unsqueeze(0)
	if img_batch.shape[-1] != 3: raise Exception("image[{}] must be RGB!".format(img_batch.shape))
	mask_batch=mask_batch.squeeze(1)
	batch=  len(img_batch) if isinstance(img_batch, list) else img_batch.shape[0]
	img_height,img_width=img_batch.shape[1:3]
	save_dirs=[save_dirs]*batch if not isinstance(save_dirs, list) else save_dirs
	for i, filename in enumerate(filenames):
		save_dir=save_dirs[i]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		#filename = str(filename).split("'")[-2].replace("/","_")
		filename=filename.decode("utf-8")  if not isinstance(filename, str) else filename
		heatmap = cv2.applyColorMap(np.uint8(255 * mask_batch[i]), cv2.COLORMAP_JET)
		heatmap=cv2.resize(heatmap,(img_width,img_height))
		heatmap = np.float32(heatmap) / 255
		#img_show=cv2.cvtColor( np.uint8(255 * img_batch[i]), cv2.COLOR_GRAY2BGR)
		img_show =np.uint8(255 * img_batch[i])
		cam = heatmap + np.float32(img_show)/255
		#cam=np.float32(img_show) / 255
		cam = cam / np.max(cam)
		cam=np.uint8(255 * cam)
		visualization_path = os.path.join(save_dir,filename)
		print("write to {}".format(visualization_path))
		cv2.imwrite(visualization_path, cam)

def cv_imread(file_path, flag=1):
	"""
	解决cv包含中文路径的问题
	:param file_path:  路径
	:param flag:
	:return:
	"""
	cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
	return cv_img
