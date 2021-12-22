import tensorflow as tf
import numpy as np
import os
import cv2

def transforms(image):
    """
    convert image to tensor, and normalize it with mean and std
    This is rewrite from the code:

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    :param image -- numpy array
    :return:
        normalized tensor
    """
    img = tf.convert_to_tensor(image)
    norm_img = tf.image.per_image_standardization(img)

    return norm_img

def xywh2cs(x, y, w, h, aspect_ratio, ):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w <aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
    return center, scale
def box2cs(box):
    x, y, w, h = box[:4]
    return xywh2cs(x, y, w, h)
def load_image_processed(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    im = cv2.imread(img, cv2.IMREAD_COLOR)
    h, w, _ = im.shape
    parsing_anno = np.zeros((h, w), dtype=np.long)
    person_center, s = box2cs([0, 0, w - 1, h - 1])
    r = 0

    # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

    #train

    #val
    return img

def load_data(root):
    """

    :param root -- data/ATR
    :return:
        train_dataset, val_dataset
    """
    train_image_list = os.listdir(os.path.join(root,"train_images"))
    train_seg_list = os.listdir(os.path.join(root, "train_segmentations"))
    val_image_list = os.listdir(os.path.join(root, "val_images"))
    val_seg_list = os.listdir(os.path.join(root, "val_segmentations"))

    train_image_path = [os.path.join(root,"train_images", i) for i in train_image_list]
    train_seg_path = [os.path.join(root,"train_segmentations", i) for i in train_seg_list]
    val_image_path = [os.path.join(root,"val_images", i) for i in val_image_list]
    val_seg_path = [os.path.join(root,"val_segmentations", i) for i in val_seg_list]

    # train_image_ds = tf.data.Dataset.list_files(train_image_path, shuffle=False)
    # train_seg_ds = tf.data.Dataset.list_files(train_seg_path, shuffle=False)
    # val_image_ds = tf.data.Dataset.list_files(val_image_path, shuffle=False)
    # val_seg_ds = tf.data.Dataset.list_files(val_seg_path, shuffle=False)
    train_image = tf.constant(train_image_path)
    train_seg = tf.constant(train_seg_path)
    val_image = tf.constant(val_image_path)
    val_seg = tf.constant(val_seg_path)

    train_ds = tf.data.Dataset.from_tensor_slices((train_image, train_seg))
    val_ds = tf.data.Dataset.from_tensor_slices((val_image, val_seg))

if __name__ == "__main__":
    img = tf.io.read_file("Grace_Hopper.jpg")
    img = tf.image.decode_jpeg(img, channels=3)
    im = cv2.imread(img.numpy(), cv2.IMREAD_COLOR)
    print(img.numpy())

