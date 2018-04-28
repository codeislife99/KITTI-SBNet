import os 
import glob


train_file = open('train.txt')
train = []

for line in train_file:
	if line[-1] == '\n':
		train.append(line[:-1])
	else:
		train.append(line)
# print(train)

valid_file = open('val.txt')
valid = []

for line in valid_file:
	if line[-1] == '\n':
		valid.append(line[:-1])
	else:
		valid.append(line)
# print(valid)

for element in sorted(train):
	if element == "000000":
		continue
	os.rename("./bev_samples/"+str(element)+"_1.jpg", "./bev_raw/train/"+str(element)+"_1.jpg")
	os.rename("./bev_samples/"+str(element)+"_0.jpg", "./bev_raw/train/"+str(element)+"_0.jpg")
	os.rename("./gt_masks/"+str(element)+".jpg", "./gt_masks_raw/train/"+str(element)+".jpg")

for element in sorted(valid):
	os.rename("./bev_samples/"+str(element)+"_1.jpg", "./bev_raw/val/"+str(element)+"_1.jpg")
	os.rename("./bev_samples/"+str(element)+"_0.jpg", "./bev_raw/val/"+str(element)+"_0.jpg")
	os.rename("./gt_masks/"+str(element)+".jpg", "./gt_masks_raw/val/"+str(element)+".jpg")



