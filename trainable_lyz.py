from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import backend as K
import process_img_lyz
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime

batch_size = 10
nb_classes = 21
nb_epoch = 150

img_rows, img_cols = 300, 300
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation
weights_file = "weights/densenet-goods.h5"
log_dir = "logs/densenet-goods.log"
path = "goodsdata/data_all/"
model = ""

def buildmodel():
    model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block, growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
    print("Model created")

#    model.summary()
    optimizer = Adam(lr=1e-3) # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiled.")
    return model


def loaddata():
    print('loading data...')
    # (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX, trainY, testX, testY, class_names = process_img_lyz.load_js_data(path=path, img_rows=img_rows, img_cols=img_cols)
    # img = Image.fromarray(trainX[0]).convert('RGB')
    # plt.imshow(img)
    # plt.show()
    # exit(0)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    trainX = densenet.preprocess_input(trainX)
    testX = densenet.preprocess_input(testX)

    trainY = np_utils.to_categorical(trainY, nb_classes)
    testY = np_utils.to_categorical(testY, nb_classes)
    print('data loaded.')
    return trainX, trainY, testX, testY, class_names 

def train(trainX, trainY, testX, testY):
    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./img_cols,
                                   height_shift_range=5./img_rows,
                                   horizontal_flip=True)

    generator.fit(trainX, seed=0)
    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                        cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                      save_weights_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    callbacks=[lr_reducer, model_checkpoint, tensorboard]

    print('training...')
    model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size),
                        steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, testY),
                        validation_steps=testX.shape[0] // batch_size, verbose=1)

    print('saving weights...')
#    filename = str(datetime.datetime.now())
    model.save_weights(weights_file,overwrite=True)


def test(testX, testY, classnames):
    model = load_weights(model)
    print('testing...')
    predY = model.predict(testX)
    predY = np.argmax(predY, axis=1)
    testY = np.argmax(testY, axis=1)

    accuracy = metrics.accuracy_score(testY, predY) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    confusion_mat = confusion_matrix(testY, predY)
    print(class_names)
    print(confusion_mat)

def predict(model, testX, classnames):
    print('predicting...')
    testX = np.expand_dims(testX, axis=0) 
    predY = model.predict(testX)[0]
    print(predY)
    Y = np.argmax(predY, axis=0)
    print(str(Y) + '---' + classnames[Y] + ': ' + str(predY[Y]) )
    return Y, predY[Y], classnames[Y]


def load_weights(model):
    # Load model
    if os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print("Model loaded")
        return model
    else:
        print("Model " + weights_file + " not found, Exited!")
        return None



def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')

if __name__ == '__main__':
    model = buildmodel()
    trainX, trainY, testX, testY, class_names = loaddata()
    train(trainX, trainY, testX, testY)
    test(trainX, trainY, class_names)

# ssh -L 6006:127.0.0.1:6006 root@192.168.8.2 -p2222