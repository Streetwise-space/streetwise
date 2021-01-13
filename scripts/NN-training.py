'''
# Script for training neural network to compare two images and find the winner, based on the results of a crowdsourcing campaign.
# More info on https://github.com/Streetwise/streetwise-data/wiki
'''

# Import necessary libraries
import tensorflow
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D,Activation, Concatenate
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.applications import resnet50, vgg19
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import gc
import json
import csv
from sklearn.preprocessing import LabelBinarizer
from random import shuffle

# select which GPU to use for training
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### Params ###
# Set here the basic params to choose which data to use for the training
path = "images-atmos/" # The folder where the images are located
cspath = "crowdsourcing/atmosphere-final.json" # The file where the crowdsourcing results are located
output_path = "models/full_model_atmos.h5" # The file where the trained model will be saved

# Load all images
tempImages = {}
for f in os.listdir(path):
    tempImages[f[:-4]] = preprocess_input(cv2.imread(path+f))

# Shuffle dataset, to have it more balanced
def shuffle_data(left, right, lab):
  shuffled = list(zip(left, right, lab))
  shuffle(shuffled)
  return zip(*shuffled)

lb = LabelBinarizer()

imgsLeft = []
imgsRight = []
labels = []

# Load data from json file
with open(cspath) as f:
    data = json.load(f)
    for d in data:
        if (d["winner"] == "left" or d["winner"] == "right") and d["campaign"]["id"] == 2:
            imgsLeft.append(d["left_image"]["filename"])
            imgsRight.append(d["right_image"]["filename"])
            labels.append(d["winner"])
            
print(labels[0])

# Transform "right" to (0,1) and "left" to (1,0)        
first_label = labels[0]
lb.fit(labels)
labels = lb.transform(labels)
labels = np.hstack((labels, 1 - labels)) if first_label == "right" else np.hstack((1-labels, labels))

print(labels[0])

dataset = list(zip(imgsLeft, imgsRight, labels))
shuffle(dataset)


# #### Create training and testing dataset

# Validation and training sets - With training and validation split - uncomment this and comment next part
'''
validationSet = dataset[int(0.8*len(dataset)+1):]
trainingSet = dataset[:int(0.8*len(dataset))]
'''

# Validation and training sets for pure training 
# (NOTE: use this if you want a more accurate network, but obviously validation accuracy does not mean anything in this case,
# the validation sert here is just a dummy and overlaps with the training set)
validationSet = dataset[int(0.99*len(dataset)+1):]
trainingSet = dataset

# Unzip the sets
imgsLeftTest, imgsRightTest, labelsTest = zip(*validationSet)
imgsLeft, imgsRight, labels = zip(*trainingSet)

# Extend training with inversed comparisons (img i vs img j, i wins => img j vs img i, i wins)
ilt = imgsLeft
irt = imgsRight
imgsLeft = imgsLeft + irt
imgsRight = imgsRight + ilt
labels = np.concatenate((np.array(labels), 1-np.array(labels)))

print(labels.shape)

# Extend validation with inversed comparisons (img i vs img j, i wins => img j vs img i, i wins)
labelsTest = np.array(labelsTest)
ilt = imgsLeftTest
irt = imgsRightTest
imgsLeftTest = imgsLeftTest + irt
imgsRightTest = imgsRightTest + ilt
labelsTest = np.concatenate((np.array(labelsTest), 1-np.array(labelsTest)))

print(labelsTest.shape)

## Function to load a batch of data
def load_data(leftIm, rightIm, labs, idx, batch_size, augmentation = True):
  id = idx*batch_size
  
  lftImgs = leftIm[id%len(leftIm):id%len(leftIm)+batch_size]
  rgtImgs = rightIm[id%len(leftIm):id%len(leftIm)+batch_size]
    
  lft = np.squeeze(np.array([tempImages[l[:-4]] for l in lftImgs]))
  rgt = np.squeeze(np.array([tempImages[r[:-4]] for r in rgtImgs]))
  lbls = np.squeeze(np.array(labs[id%len(leftIm):id%len(leftIm)+batch_size]))
    
  return ([lft, rgt], lbls)

def batch_generator(leftIm, rightIm, labs, batch_size, steps, augmentation = True):
  leftIm, rightIm, labs = shuffle_data(leftIm, rightIm, labs) # Is this done automatically by model.fit()?
  idx=1
  while True:
    yield load_data(leftIm, rightIm, labs, (idx-1), batch_size, augmentation)## Yields data
    if idx<steps:
      idx+=1
    else:
      idx=1


# Generate NN model

# Two inputs one each - left and right image
lInput = Input((224,224,3))
rInput = Input((224,224,3))

#Import Resnet architecture from keras application and initializing each layer with pretrained imagenet weights.
convnet = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Add the final fully connected layers
x = convnet.output
convnet = Model(inputs=convnet.input, outputs=x)

#Applying above model for both the left and right images
encoded_l = convnet(lInput)
encoded_r = convnet(rInput)

# Fusion layers
init = tensorflow.keras.initializers.RandomNormal(stddev=0.001)
fused = Concatenate()([encoded_l, encoded_r])

# Convolutional layers
fusion = Conv2D(filters=8, kernel_size=(3,3), kernel_initializer=init)(fused)

fusion = Conv2D(filters=8, kernel_size=(3,3), kernel_initializer=init)(fusion)
fusion = MaxPooling2D((3,3))(fusion)

fusion = Flatten()(fusion)
prediction = Dense(2, activation="softmax",
                kernel_initializer=init,
                )(fusion)

#Define the network with the left and right inputs and the ouput prediction
siamese_net = Model(inputs=[lInput,rInput],outputs=prediction)

### Warm-up of the network ###
# Train only last 4 layers...
for layer in siamese_net.layers[:-5]:
    layer.trainable = False

# ...with relatively high learning rate
optim = optimizers.SGD(lr=0.02)

#compile the network using binary cross entropy loss and the above optimizer
siamese_net.compile(loss="binary_crossentropy",optimizer=optim,metrics=['accuracy'])

siamese_net.summary()

# Run warm-up training for a small nuber of epochs
batch_size = 64
nb_epoch = 4
steps_per_epoch=np.floor(len(imgsLeft)/batch_size)
validation_steps = np.floor(len(imgsLeftTest)/batch_size)
# Generator objects for train and validation
training_batch_generator = batch_generator(imgsLeft, imgsRight, labels, batch_size, steps_per_epoch, True)
validation_batch_generator = batch_generator(imgsLeftTest, imgsRightTest, labelsTest, batch_size, validation_steps, False)

class MyCustomCallback(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()

gc.collect()

siamese_net.fit(training_batch_generator,
    epochs=nb_epoch,
    steps_per_epoch=steps_per_epoch,
    verbose=1, 
    validation_data=validation_batch_generator,
    validation_steps=validation_steps,
    callbacks=[MyCustomCallback()])

### Fine tuning of the model ###

# Set all layers to trainable...
for layer in siamese_net.layers[:]:
    layer.trainable = True
    
# ... with a very small learning rate
optim = optimizers.SGD(lr=0.000001)

#compile the network using binary cross entropy loss and the above optimizer
siamese_net.compile(loss="binary_crossentropy",optimizer=optim,metrics=['accuracy'])

# Run fine-tuning training for a larger number of epochs
batch_size = 16
nb_epoch = 20
steps_per_epoch=np.floor(len(imgsLeft)/batch_size)
validation_steps = np.floor(len(imgsLeftTest)/batch_size)### Generator objects for train and validation
training_batch_generator = batch_generator(imgsLeft, imgsRight, labels, batch_size, steps_per_epoch, True)
validation_batch_generator = batch_generator(imgsLeftTest, imgsRightTest, labelsTest, batch_size, validation_steps, False)

gc.collect()

siamese_net.fit(training_batch_generator,
    epochs=nb_epoch,
    steps_per_epoch=steps_per_epoch,
    verbose=1, 
    validation_data=validation_batch_generator,
    validation_steps=validation_steps,
    callbacks=[MyCustomCallback()])

# summarize history for accuracy
plt.plot(siamese_net.history.history['accuracy'])
plt.plot(siamese_net.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(siamese_net.history.history['loss'])
plt.plot(siamese_net.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save trained model
siamese_net.save(output_path)
