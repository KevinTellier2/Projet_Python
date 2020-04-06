# CIFAR-10 Prediction In Keras using a Convolutional Neural Network


## Motivations :
This project is an example of image classification using a Convolutional Neural Network, using Keras and the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The Convolutional Neural Network model was made using Python and trained with the CIFAR-10 dataset. We try to correctly classify images into ten categories, such as a: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The final accuracy is roughly 81 %;


## Steps :

### 1. Loading the dataset

#### Examining the dataset

Shape of the training data : (50000, 32, 32, 3)

Shape of the test data : (10000, 32, 32, 3)

So we have 50000 training and 10000 test images in this dataset. The images have a structure of (32, 32, 3) which correspond to (width, height, RGB).


### 2. Preparing the dataset
We first assign the type of float point number for our train and test samples, followed by a Z-score normalization to avoid the outlier issue. Then we convert integer targets into categorical targets, to represent the labels of the samples instead of class indices.

### 3. CNN classifier

#### Model configuration
In order to have an easy code reproducibility, we configure the model parameters at the start, like the number of epochs, the batch size, the loss function...

#### Defining the Convolutional Neural Network model
We will use six convolutional input layers (two with 32 filters, two with 64 filters and two with 128 filters, all with a size of 3x3) followed by max pooling and a flattening out to make predictions. We've used a Kernel regularizer, which means that it allows to apply penalties on layer parameters during the optimization, as well as the ReLU activation function.
Then we have a fully connected output layer with ten units and a softmax activation function.

#### Image Data Augmentation
Image data augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of images in the dataset, and to result in better models.

#### Creation of the optimizer and compilation
The RMSprop optimizer is like to the gradient descent algorithm but it restricts the oscillations in the vertical direction.
The model was compiled with the 'categorical_crossentropy' loss function, the RMSprop optimizer and the 'accuracy' metric. 

#### Fitting the model







## Important points :

## Possible future development paths :


### License

[MIT](https://github.com/KevinTellier2/Projet_Python/blob/master/LICENSE)

