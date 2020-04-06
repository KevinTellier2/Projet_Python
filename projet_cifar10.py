
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#        Projet M2 ESA - Machine Learning with Python - AMANCY Alexis - MAZLOUM Sabrina - TELLIER Kevin         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# Make sure you have all the following libraries and tools installed in your system :

'''
# TensorFlow is an end-to-end open source platform for machine learning.
# pip install tensorflow

# NumPy is the fundamental package for scientific computing with Python :
# pip install numpy

# A library for Keras for investigating architectures and parameters of sequential models :
# pip install keras-sequential-ascii

# Pillow is a free library for the Python programming language that adds support for
# opening, manipulating, and saving many different image file formats :
# pip install Pillow

# SciPy is open-source software for mathematics, science, and engineering :
# conda install -c anaconda scipy=1.2.1

# Matplotlib is a comprehensive library for creating static, animated, and
# interactive visualizations in Python
# pip install matplotlib

# OpenCV-Python is a library of Python bindings designed to solve computer vision problems :
# pip install opencv-python
'''



###################################################################################################
###################################################################################################
#################################             IMPORTANT           #################################
###################################################################################################
###################################################################################################
###########      Make sure to paste the images .JPG to your working directory first ! #############
######    You can get your working directory by submitting the following instruction :  ###########
#################################             os.getcwd()           ###############################
######  You will find the 10 images downloaded in our Github repo, in the "Images" folder.   ######
###########        Github link : https://github.com/KevinTellier2/Projet_Python         ###########
###################################################################################################
###################################################################################################



# Importing packages :

from __future__ import print_function
import numpy as np
import keras
from keras import regularizers
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from scipy.misc import toimage
import cv2



# Model configuration :

nb_class = 10
img_rows, img_cols = 32, 32
loss = 'categorical_crossentropy' # Name of the loss function.
batch_size = 32 # Highest number that your machine has memory for.
epoch = 60 # Number of iterations until the network stops learning.
lr = 0.0001 # Learning rate.
decay = lr / epoch
verbose = 1
weight_decay = 1e-4
padding = 'same' # "same" results in padding the input such that the output has the same length as the original input.
active_function = 'relu' # Relu activation for the Activation layer.



# Loading the CIFAR-10 data :
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



# Blurring images :
'''
Image blurring is achieved by convolving the image with a low-pass filter kernel. I'ts useful for removing noises.
It actually removes high frequency content, i.e. noises, edges from the image resulting in edges being blurred
when this filter is applied. This is done by convolving the image with a normalized box filter.
So, here, it simply takes the average of all the pixels under kernel area and replaces the
central element with this average, doing this for all the 50 000 images in the train sample.
'''

for i in range(0,50000):
    imag = x_train[i]
    if np.argmax(y_train[i]) == 2:
        x_train[i] = cv2.blur(imag, (32, 32))

# Examining the dataset :
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Assigning the type of float point number for our train/test samples :

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Z-score normalization :
'''
It's a strategy of normalizing data that avoids the outlier issue.
'''
mean = np.mean(x_train, axis = (0, 1, 2, 3))
std = np.std(x_train, axis = (0, 1, 2, 3))
x_train = (x_train - mean) / (std)
x_test = (x_test - mean) / (std)

# Data conversion :
'''
The following two lines will convert integer targets into categorical targets.
'''
y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)



# Let's now define the Convolutional Neural Network model :

'''
# We will use a model with six convolutional input layers followed by max pooling and
# a flattening out to make predictions.

# 1 - Convolution with 32 different filters with a size of 3x3, with a Kernel_regularizer,
#     which means that it allows to apply penalties on layer parameters during optimization.
# 2 - Convolution with 32 different filters with a size of 3x3, with a Kernel_regularizer.
# 3 - MaxPool layer with a size of 2×2.
# 4 - Dropout set to 25 %.
# 5 - Convolution with 64 different filters with a size of 3x3, with a Kernel_regularizer.
# 6 - Convolution with 64 different filters with a size of 3x3, with a Kernel_regularizer.
# 7 - MaxPool layer with a size of 2×2.
# 8 - Dropout set to 35 %.
# 9 - Convolution with 128 different filters with a size of 3x3, with a Kernel_regularizer.
# 10 - Convolution with 128 different filters with a size of 3x3, with a Kernel_regularizer.
# 11 - MaxPool layer with a size of 2×2.
# 12 - Dropout set to 50 %. 
# 13 - Fully connected output layer with 10 units and a softmax activation function.
'''
model = Sequential()

# Conv2D means a 2-dimensional convolutional layer
model.add(Conv2D(32, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay),
                 input_shape = x_train.shape[1:]))
model.add(Activation(active_function))


model.add(Conv2D(32, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation(active_function))
# The Maxpooling objective is to down-sample an input representation, reducing
# its dimensionality and allowing for assumptions to be made about features
# contained in the sub-regions binned, and to retain the most important information.
model.add(MaxPooling2D(pool_size=(2, 2)))
# The key idea is to randomly drop units along with their connections from the
# neural network during training. The reduction in number of parameters has
# effect of regularization --> Dropout.
model.add(Dropout(0.25))


model.add(Conv2D(64, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation(active_function))
model.add(Conv2D(64, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation(active_function))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.35))


model.add(Conv2D(128, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation(active_function))
model.add(Conv2D(128, (3, 3), padding = padding,
                 kernel_regularizer = regularizers.l2(weight_decay)))
model.add(Activation(active_function))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))


# The role of the Flatten layer in Keras is super simple :
# A flatten operation on a tensor reshapes the tensor to have the shape that
# is equal to the number of elements contained in tensor non including the
# batch dimension.
model.add(Flatten())
model.add(Dense(nb_class, activation = 'softmax'))

# See next what our CNN model looks like :
model.summary()



# Image Data Augmentation :
'''
Image data augmentation is a technique that can be used to artificially expand the size of a
training dataset by creating modified versions of images in the dataset.
Training neural network models on more data can result in better models,
and the augmentation techniques create variations of the images in order to improve the
ability of the fit models to generalize well what they have learned to new images.
'''
data_augm = ImageDataGenerator(
        zca_epsilon=1e-6, # The epsilon for ZCA whitening.
        width_shift_range=0.1, # Randomly shift images horizontally.
        height_shift_range=0.1, # Randomly shift images vertically.
        horizontal_flip=True, # Randomly flip images horizontally.
        data_format=None) # Format of the image data, either 'channels_first' or 'channels_last'.
data_augm.fit(x_train)



# Let's create our optimizer with the RMSprop algorithm :
'''
The RMSprop optimizer is like to the gradient descent algorithm but it restricts the oscillations
in the vertical direction.
'''
optimizer = keras.optimizers.RMSprop(learning_rate = lr, decay = decay)



# Let's train the model using RMSprop :

model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])



# Fitting the model on the batches generated by data_augm.flow() :

model_cnn_fit = model.fit_generator(data_augm.flow(x_train, y_train, batch_size = batch_size),
                        epochs = epoch, validation_data = (x_test, y_test), workers = 4)



# Let's now check the results by plotting for the accuracy and loss for our two samples :
# Accuracy :

plt.figure(0)
plt.plot(model_cnn_fit.history['accuracy'],'blue')
plt.plot(model_cnn_fit.history['val_accuracy'],'red')
plt.xticks(np.arange(0, 61, 1.0))
plt.rcParams['figure.figsize'] = (10, 8)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(['Train','Validation'])
 
# Loss :

plt.figure(1)
plt.plot(model_cnn_fit.history['loss'],'blue')
plt.plot(model_cnn_fit.history['val_loss'],'red')
plt.xticks(np.arange(0, 61, 1.0))
plt.rcParams['figure.figsize'] = (10, 8)
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(['Train','Validation'])

plt.show()

'''
In approximately 82% of the cases, our model was correct. This is in line with the validation accuracies visualized across the epochs.
At first, loss went down pretty fast, and then always continues to go down, around 0.6.
So our model doesn't seem to overfit the data. Thus, our model may have good results on data it has never seen before.
'''



# Saving the file to our disk :

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('cifar10_project.h5') 



# Final evaluation of the model :

scores = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = verbose)
print("Loss on test data is: %.3f" % (scores[0]))
print("Accuracy on test data is : %.3f%%" % (100 * scores[1]))
'''
Loss on test data is : 0.588
Accuracy on test data is : 81.940%
'''


# Let's see if our training was good enough to recognize images :

def show_imgs(X):
    plt.figure(1)
    a = 0
    for q in range(0,3):
        for r in range(0,4):
            plt.subplot2grid((3,4),(q,r))
            plt.imshow(toimage(X[a]))
            a = a+1
    # show the plot
    plt.show()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Plotting 12 images we're going to predict the good label :
show_imgs(x_test[:12])



# Loading the trained CNN model :

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('cifar10_project.h5')


labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
names = np.argmax(model.predict(x_test[:12]), 1)
print([labels[k] for k in names])

'''
Seven out of twelve images were well predicted, so only 58.33 %...
'''



# Now, let's test the model with some random input images, found on https://www.pexels.com/fr-fr/ !!

# Give the link of the images here to test :

test_image_1 = image.load_img('airplane1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('automobile1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('bird1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('cat1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('deer1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Uncorrectly predicted.



test_image_1 = image.load_img('dog1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('frog1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('horse1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



test_image_1 = image.load_img('ship1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Uncorrectly predicted.



test_image_1 = image.load_img('truck1.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image_1)
test_image = np.expand_dims(test_image, axis = 0) 
result = model.predict(test_image) 
print(result) 
if result[0][0]==1: 
    print("Airplane") 
elif result[0][1]==1: 
    print('Automobile') 
elif result[0][2]==1: 
    print('Bird') 
elif result[0][3]==1: 
    print('Cat') 
elif result[0][4]==1: 
    print('Deer') 
elif result[0][5]==1: 
    print('Dog') 
elif result[0][6]==1: 
    print('Frog') 
elif result[0][7]==1: 
    print('Horse') 
elif result[0][8]==1: 
    print('Ship') 
elif result[0][9]==1: 
    print('Truck') 
else: 
    print('Error')

plt.imshow(test_image_1)
# Correctly predicted.



'''
Conclusion :
Eight images out ten correctly predicted on collected images.
'''
