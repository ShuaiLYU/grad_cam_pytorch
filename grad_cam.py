#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/15 13:45
# @Author  : Wslsdx
# @FileName: cam_test.py
# @Software: PyCharm
# @Github  ：https://github.com/Wslsdx
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
from torch import nn
import cv2
import sys
import numpy as np
from torchvision.models import  resnet

class GradCam(object):
    def __init__(self, model,device,activation="normalize"):
        self.model = model
        self.device=device
        self.activation=activation
        if self.activation  not in ("normalize","sigmoid","tanh"):
            raise Exception("Got an unexpected activation mode!!!".format(self.activation))
        self.features=[]
        self.grads=[]

    def __call__(self,input,target=None):
        """

        :param input:   Tensor[B,3,H,W]
        :param target:    Tensor[B]   if target is None ,
        :return:    list [  Tensor[B,1,H,W]  ] ,    len(list) equal the number of "target_layer_names"
        Note : The order of the output is related to the order of the network layer rather than the order of "target_layer_names"
        """

        self.features=[]
        self.grads=[]
        logits=self.model(input)
        assert (logits.dim()==2)
        predict=torch.argmax(logits,dim=-1,keepdim=False)
        target=predict if target is None else target
        one_hot=np.eye(logits.size()[-1])[target.cpu()]
        one_hot =torch.from_numpy(one_hot).type_as(logits).to(self.device)
        one_hot =(one_hot * logits).sum()
        self.model.zero_grad()
        one_hot.backward()
        #one_hot=one_hot.sum(dim=-1,keepdim=False)
        #self.model.zero_grad()
        # one_hot.backward(torch.ones_like(one_hot))
        cams=[  self.get_cam(i) for  i in range(len(self.target_layer_names))]
        return cams

    def get_cam(self,layer_idx=0):
        """
        :param layer_idx:  layer index
        :return:  cam_batch:  numpy[batch,1,H,W]
        """
        feature=self.features[layer_idx]
        grad =self.grads[layer_idx]
        weights=torch.mean(grad,dim=[2,3],keepdim=True)
        cam=torch.sum(feature*weights,dim=1,keepdim=True)
        # cam=cam.detach().cpu().numpy()
        if self.activation is "sigmoid":
            cam=torch.sigmoid(cam)
        elif self.activation is "tanh":
            cam = torch.tanh(cam)
        elif self.activation is "normalize":
            cam=self.normalize(cam)
        return cam
    def normalize(self,input):
        B,C,H,W=input.size()
        input=input.view( B,C,-1)
        min_value,_=torch.min(input,dim=-1,keepdim=True)
        input=input-min_value
        max_value,_=torch.max(input,dim=-1,keepdim=True)
        input=input/max_value
        return input.view(B,C,H,W)

    def register_hooks(self,target_layer_names):
        """
        :param target_layer_names:     list[ layer_path ]
        :return:
        """

        self.target_layer_names=target_layer_names
        self.hook_handles=[]
        for layer_name in target_layer_names:
            pointer=self.get_pointer(layer_name)
            handle = pointer.register_forward_hook(self.forward_hook)
            self.hook_handles.append(handle)
            handle = pointer.register_backward_hook(self.backward_hook)
            self.hook_handles.append(handle)
    def remove_hook_handles(self):
        for handle in self.hook_handles:
            handle.remove()


    def forward_hook(self,module, input, output):
        """
        :param module:  example:  <class 'torch.nn.modules.conv.Conv2d'>
        :param input:     tuple( input )
        :param output:    <class torch.nn.Tensor(B,C,H,W)>
        :return:
        """
        # print(type(module))
        # print(type(input))
        # if isinstance(input,tuple):
        #     print(len(input))
        #     for item in input:
        #         print(item.size())
        # else:
        #     print(input.size())
        # if isinstance(output, tuple):
        #     print(len(output))
        #     for item in output:
        #         print(item.size())
        # else:
        #     print(output.size())
        self.features.append(output)
    def backward_hook(self,module,input_grad, output_grad):
        """
        :param module:    example:  <class 'torch.nn.modules.conv.Conv2d'>
        :param input_grad:   tuple( input_grad, wights_grad,bias_grad )
        :param output_grad:  tuple( output_grad)
        :return:
        """
        # print(type(module))
        # print(type(input_grad))
        # if isinstance(input_grad, tuple):
        #     print(len(input_grad))
        #     for item in input_grad:
        #         print(item.size())
        # else:
        #     print(input_grad.size())
        # if isinstance(output_grad, tuple):
        #     print(len(output_grad))
        #     for item in output_grad:
        #         print(item.size())
        # else:
        #     print(output_grad.size())
        #[0]：output_grad是一个元组
        self.grads.append(output_grad[0])

    def get_pointer(self,layers):
        pointer=self.model
        layers=layers.split(".")
        for layer in layers:
            if isinstance(pointer,nn.Sequential):
                pointer=eval("pointer[{}]".format(layer))
            else:
                pointer=eval("pointer.{}".format(layer))
        return  pointer




# if __name__=="__main__":
#
#     device=torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
#     model=resnet.resnet18(pretrained=True).to(device)
#     # for name, module, in model._modules.items():
#     #     print(name)
#
#     target_layer_names=["layer4"]
#     gram=GradCam(model,device,target_layer_names=target_layer_names,activation="tanh")
#     input=torch.from_numpy( np.random.rand(5,3,224,224)).float().to(device)
#     cam=gram(input)


