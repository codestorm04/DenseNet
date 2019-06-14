import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from sklearn.utils import shuffle
import cv2


def resizeImg(path='JS_Data/', img_rows=300, img_cols=300):
    listing_dir = os.listdir(path)
    for category in listing_dir:
        listing = os.listdir(path + category)
        for file in listing:
            filename = path + category +'/' + file
            print(filename)
            img = Image.open(filename)
            img = img.resize((img_cols, img_rows))
            # grayimg = img.convert('L')
            img.save(filename)


def load_js_data(path='JS_Data/', img_rows=300, img_cols=300):
    class_names = os.listdir(path)
    nb_classes = len(class_names)
    imlist = []
    Y = []

    for idx in range(nb_classes):
        for file in os.listdir(path + class_names[idx]):
            filename = path + class_names[idx] +'/' + file
            imlist.append(filename)
            Y.append(idx)

    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(images)).flatten()
                         for images in imlist], dtype = 'f')
    Y = np.array([[y] for y in Y])
    immatrix = immatrix.astype('uint8')
    Y = Y.astype('uint8')
    
    X, Y = shuffle(immatrix, Y, random_state=2)
     
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    return X_train, Y_train, X_test, Y_test, class_names

# Crop the only object in a figure
def bounding_crop(path='JS_Data/', img_rows=300, img_cols=300):
    listing_dir = os.listdir(path)
    for category in listing_dir:
        listing = os.listdir(path + category)
        for file in listing:
            filename = path + category +'/' + file
            print(filename)

            image = cv2.imread(filename)
            # image = cv2.resize(image, (img_rows, img_cols))
            img = cv2.Canny(image, 120, 200)
            # img = cv2.medianBlur(img, 3)

            tmp = np.sum(img, axis=0)  
            x, w = _hill_filter(tmp)  
            tmp = np.sum(img, axis=1)
            y, h = _hill_filter(tmp)  

            if x is not None and y is not None and w is not None and h is not None:
                x, y, w, h = _aspect_ratio_rectify(x, y, w, h, img.shape[1], img.shape[0])
                image = image[y: y+h, x: x+w, : ]
                image = cv2.resize(image, (img_rows, img_cols))

            cv2.imwrite(filename, image)
            # cv2.imshow(str(random.random()), img)
            # cv2.imshow(str(random.random()), image) 


# filer the small hills of accumulative array to remove noise points
def _hill_filter(arr):
    arr = arr/255
    acc = np.empty([0, 3], dtype=int)
    tmp = 0
    start = -1
    end = -1
    for i in range(len(arr)):
        if arr[i] > 0:
            tmp += arr[i]
            if start == -1:
                start = i
            if i == len(arr) - 1:
                end = i
                acc = np.concatenate((acc, [np.array([tmp, start, end])]), axis=0)
        elif tmp != 0:
            end = i - 1
            acc = np.concatenate((acc, [np.array([tmp, start, end])]), axis=0)
            tmp = 0
            start = -1
    if len(acc) > 0:
        thresh = np.mean(acc, axis = 0)[0] * 0.15
        x = 0
        x2 = 0
        for hill in acc:
            if hill[0] > thresh:
                x = hill[1]
                break
        for hill in acc[::-1, :]:
            if hill[0] > thresh:
                x2 = hill[2]
                return int(x), int(x2 - x)
    return None, None

# If the aspect ratio more than ratio_thresh then keep padding
def _aspect_ratio_rectify(x, y, w, h, width, height):
    ratio_thresh = 0.667
    if w > h * ratio_thresh and h > w * ratio_thresh:
        return x, y, w, h
    if w <= h * ratio_thresh:
        delta = int((h * ratio_thresh - w) / 2)
        w = int(h * ratio_thresh)
        x = max(0, x - delta)
        w = min(w, width)
    else:
        delta = int((w * ratio_thresh - h) / 2)
        h = int(w * ratio_thresh)
        y = max(0, y - delta)
        h = min(h, width)
    return x, y, w, h


    
if __name__ == '__main__':
    # load_js_data()

    resizeImg(path='/home/lyz/desktop/js_data_512x512/', img_rows=512, img_cols=512)
    # load_js_data(path='JS_Data/', img_rows=200, img_cols=320)
    # bounding_crop(path="/home/lyz/desktop/goods_data_croped/", img_rows=299, img_cols=299)
