import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import numpy as np
import cv2
# from networks import unet
from sys import argv
import time

def get_mask(input_tensor,model_unet,output_size = (800,704)):
	# checkpoints_directory_unet="checkpoints_unet"
	
	# python api.py /path/to/image/imagename.extension 
	#will give output in the folder containing script as Output_unet.png

	# checkpoints_unet= os.listdir(checkpoints_directory_unet)
	# checkpoints_unet.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
	# start_time = time.time()
	model_unet.eval()
	# model_unet.cuda()
	if torch.cuda.is_available(): #use gpu if available
		model_unet.cuda() 
	# print(time.time()-start_time)

	# image1 = cv2.imread(img_path1)
	# orig_width,orig_height=image1.shape[0],image1.shape[1]
	# input_unet1 = image1

	input_unet1 = cv2.resize(input_tensor[0,:,:,:3],(256,256), interpolation = cv2.INTER_CUBIC)
	input_unet1 = input_unet1.reshape((256,256,3,1))

	input_unet1 = input_unet1.transpose((3, 2, 0, 1))

	input_unet1.astype(float)
	# input_unet1=input_unet1/255

	input_unet1 = torch.from_numpy(input_unet1)


	input_unet1=input_unet1.type(torch.FloatTensor)

	if torch.cuda.is_available(): #use gpu if available
	    input_unet1 = Variable(input_unet1.cuda(),volatile = True) 
	else:
		input_unet1 = Variable(input_unet1, volatile = True)

	# image2 = cv2.imread(img_path2)
	# orig_width,orig_height=image2.shape[0],image2.shape[1]
	# input_unet2 = image2

	input_unet2 = cv2.resize(input_tensor[0,:,:,3:],(256,256), interpolation = cv2.INTER_CUBIC)
	input_unet2 = input_unet2.reshape((256,256,3,1))

	input_unet2 = input_unet2.transpose((3, 2, 0, 1))

	input_unet2.astype(float)
	# input_unet2=input_unet2/255

	input_unet2 = torch.from_numpy(input_unet2)


	input_unet2=input_unet2.type(torch.FloatTensor)

	if torch.cuda.is_available(): #use gpu if available
	    input_unet2 = Variable(input_unet2.cuda()) 
	else:
		input_unet2 = Variable(input_unet2,volatile = True)

        
	input_unet = torch.cat((input_unet1,input_unet2), dim= 1)
	#start_time = time.time()
        out_unet = model_unet(input_unet)
	#print(time.time()-start_time)

	out_unet =  out_unet.cpu().data.numpy()


	# out_unet = out_unet*255


	out_unet = out_unet.transpose((2,3,0,1))
	out_unet= out_unet.reshape((256,256,1))
	out_unet= cv2.resize(out_unet,output_size, interpolation=cv2.INTER_CUBIC)

	# (thresh, im_bw) = cv2.threshold(out_unet, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# im_bw = out_unet
	# thresh = 25
	# im_bw = cv2.threshold(out_unet, thresh, 255, cv2.THRESH_BINARY)[1]
	# out_unet = out_unet.astype('uint8')
	# im_bw = cv2.adaptiveThreshold(out_unet,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
	# y = np.expand_dims(a,axis = 0) 
	# y = np.expand_dims(y,axis = -1) 
	# y = np.repeat(y,6,axis =3)
	return np.repeat(np.expand_dims(np.expand_dims(out_unet,axis = 0),axis = -1),6,axis = 3)
	# cv2.imwrite("Output_unet_2.png", im_bw)
if __name__ == '__main__':
	input_tensor = np.zeros((1,700,800,6))
	model_unet = torch.load('model_epoch_170.pt')

	start_time = time.time()
	output = get_mask(input_tensor,model_unet)
	# print(time.time()-start_time)
	# print(output.shape)
