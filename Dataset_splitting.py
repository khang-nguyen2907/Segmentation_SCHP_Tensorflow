import os
import random
import glob
import shutil
imgs_path = "G:\\retain\AIP391\ATR\humanparsing\JPEGImages"
segs_path = "G:\\retain\AIP391\ATR\humanparsing\SegmentationClassAug"

train_img_path = "G:\\retain\AIP391\ATR\humanparsing\\train\image"
train_label_path = "G:\\retain\AIP391\ATR\humanparsing\\train\label"

val_img_path = "G:\\retain\AIP391\ATR\humanparsing\\val\image"
val_label_path = "G:\\retain\AIP391\ATR\humanparsing\\val\label"

test_img_path = "G:\\retain\AIP391\ATR\humanparsing\\test\image"
test_label_path = "G:\\retain\AIP391\ATR\humanparsing\\test\label"

# imgs = os.listdir(imgs_path)
# train_img = os.listdir(train_img_path)
val_img = os.listdir(val_img_path)

# list_train = [i.split('.')[0] for i in train_img]
list_val = [i.split('.')[0] for i in val_img]


# train_text = open("G:\\retain\AIP391\ATR\humanparsing\\train_id.txt", "w")
#
# for element in list_train:
#     train_text.write(element+"\n")
# train_text.close()
#
val_text = open("G:\\retain\AIP391\ATR\humanparsing\\val_id.txt", "w")
for element in list_val:
    val_text.write(element+"\n")
val_text.close()

#
# train_length = int(len(imgs)*0.8)
# test_length = len(imgs) - train_length
#
# random.shuffle(imgs)
#
# train_data = imgs[:train_length]
# val_data = imgs[train_length:train_length+test_length//2]
# test_data = imgs[train_length+test_length//2:]
#
# print(len(train_data), len(test_data), len(val_data))
#
# print()
# #14164 1771 1771
# #train folder
# for data in train_data:
#     name = data.split('.')[0]
#     src_img = os.path.join(imgs_path, data)
#     src_seg = os.path.join(segs_path, name + ".png")
#     des_img = os.path.join(train_img_path,data )
#     des_seg = os.path.join(train_label_path, name+".png")
#     shutil.copy(src_img,des_img)
#     shutil.copy(src_seg, des_seg)
# #val folder
# for data in val_data:
#     name = data.split('.')[0]
#     src_img = os.path.join(imgs_path, data)
#     src_seg = os.path.join(segs_path, name + ".png")
#     des_img = os.path.join(val_img_path,data )
#     des_seg = os.path.join(val_label_path, name+".png")
#     shutil.copy(src_img,des_img)
#     shutil.copy(src_seg, des_seg)
# #test folder
# for data in test_data:
#     name = data.split('.')[0]
#     src_img = os.path.join(imgs_path, data)
#     src_seg = os.path.join(segs_path, name + ".png")
#     des_img = os.path.join(test_img_path,data )
#     des_seg = os.path.join(test_label_path, name+".png")
#     shutil.copy(src_img,des_img)
#     shutil.copy(src_seg, des_seg)
#
#
# print(len(os.listdir(train_img_path)), len(os.listdir(train_label_path)))
# print(len(os.listdir(val_img_path)), len(os.listdir(val_label_path)))
# print(len(os.listdir(test_img_path)), len(os.listdir(test_label_path)))

