# CIFAR-10 Prediction In Keras using a Convolutional Neural Network

-------------------------------------------------------------------------

## Motivations :	

<div align="justify">
  
This project is an example of image classification using a Convolutional Neural Network, using Keras and the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).	

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.	
The Convolutional Neural Network model was made using Python and trained with the CIFAR-10 dataset. We try to correctly classify images into ten categories, such as a: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.	

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.	

The final accuracy is roughly 82 %.	
</div>

-------------------------------------------------------------------------

## Steps :	

### 1. Loading the dataset	

#### Examining the dataset	

<div align="justify">

* Shape of the training data : (50000, 32, 32, 3)	

* Shape of the test data : (10000, 32, 32, 3)	

So we have 50000 training and 10000 test images in this dataset. The images have a structure of (32, 32, 3) which correspond to (width, height, RGB).	

</div>

### 2. Preparing the dataset	

<div align="justify">

We first assign the type of float point number for our train and test samples, followed by a Z-score normalization to avoid the outlier issue. Then we convert integer targets into categorical targets, to represent the labels of the samples instead of class indices.	

</div>

### 3. CNN classifier	

#### Model configuration	

<div align="justify">

In order to have an easy code reproducibility, we configure the model parameters at the start, like the number of epochs, the batch size, the loss function...	

</div>

#### Defining the Convolutional Neural Network model	

<div align="justify">

We will use six convolutional input layers (two with 32 filters, two with 64 filters and two with 128 filters, all with a size of 3x3) followed by max pooling and a flattening out to make predictions. We've used a Kernel regularizer, which means that it allows to apply penalties on layer parameters during the optimization, as well as the ReLU activation function.	
Then we have a fully connected output layer with ten units and a softmax activation function.	

</div>

#### Image Data Augmentation	

<div align="justify">

Image data augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of images in the dataset, and to result in better models.	

</div>

#### Creation of the optimizer and compilation	

<div align="justify">

The RMSprop optimizer is like to the gradient descent algorithm but it restricts the oscillations in the vertical direction.	
The model was compiled with the 'categorical_crossentropy' loss function, the RMSprop optimizer and the 'accuracy' metric. 	

</div>

#### Fitting the model

<div align="justify">

We've fit the CNN model with dropout and data augmentation, and we've obtained the following results :	

![Training vs Validation Accuracy](https://github.com/KevinTellier2/Projet_Python/blob/master/Training%20vs%20Validation%20Accuracy.png?raw=true)



![Training vs Validation Loss](https://github.com/KevinTellier2/Projet_Python/blob/master/Training%20vs%20Validation%20Loss.png?raw=true)



In approximately 82% of the cases, our model was correct. This is in line with the validation accuracies visualized across the epochs.	
At first, loss went down pretty fast, and then always continues to go down, around 0.6.	
So our model doesn't seem to overfit the data. Thus, our model may have good results on data it has never seen before.	

</div>

#### Evaluating the CNN model

<div align="justify">

* The accuracy on test data is 81.940 %.  
* The loss on test data is 0.588.


We then wanted to see if our training was good enough to recognize images, so we decided to plot the first twelve images we're going to predict the good label :  
![12_Images_from samples_we're_gonna_predict](https://github.com/KevinTellier2/Projet_Python/blob/master/12_images_gonna_predict(from%20samples).png?raw=true)  
We got the following results from our CNN trained model :  
['dog', 'automobile', 'airplane', 'airplane', 'frog', 'dog', 'automobile', 'frog', 'dog', 'truck', 'airplane', 'truck'].  

We only got seven out of twelve images were well predicted, so only 58.33 %...

</div>

#### Testing the model with some random input images

<div align="justify">

We wanted to apply the model also on examples collected by us. So we found ten high quality images on https://www.pexels.com/fr-fr/.  
We tested each of the ten images in turn and got *good results*. __Eight images out ten correctly predicted.__  
The deer was confused with the horse, and the boat gave the message Error, because the model was not 100% safe.  
<br>
For example, with that following input image :  
![Airplane_demonstration_README](https://github.com/KevinTellier2/Projet_Python/blob/master/test_airplane_in_readme.png?raw=true)  
__The result printed was :__  
[[1.0000000e+00 0.0000000e+00 0.0000000e+00 4.3595235e-22 0.0000000e+00  
  1.3940576e-29 0.0000000e+00 1.8409132e-19 0.0000000e+00 0.0000000e+00]]  
Airplane

</div>



__You can obviously try with your own images, you will find the code to do it at the bottom of the page.__


-------------------------------------------------------------------------

## Important points :

<div align="justify">

I have used Keras with Tensorflow backend to implement all the architectures. The model was trained on my computer with Nvidia Geforce GTX 970, and with Intel Core I7 4790k. But I've tested it on my I5 laptop too, it was really slower to train the model...  
Because I don't have more powerful machine to train larger networks, I decided to decrease the epochs to 60, which decreased the accuracy.

</div>

-------------------------------------------------------------------------

## Possible future development paths :

<div align="justify">

__It must be noted that the model could be modified to get maybe over 90% of accuracy.__  
To enhance model performance, we first can set up the epoch to 100 or 150 if the machine allow it.  
The model doesn't seem to overfit the training data. But using different thresholds of dropout and use Batch Normalization would resolve the problem of non having better accuracy.  
*Note that the Batch Normalization allows us to use much higher learning rates and be less careful about initialization.*  
<br>
Moreover, giving a weight initialization or tune the learning rate, batches and epochs can enhance the accuracy of the model. To improve the performances, you can inquire here : [How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/).

</div>

-------------------------------------------------------------------------


## Additional informations :

The Github repository contains 1 folder and 10 files. In the folder, you will have the 10 images from https://www.pexels.com/fr-fr/.
Among the files, you'll find :
* The file .py containing the full Python code.
* Four .png files (only needed for the [README.md](https://github.com/KevinTellier2/Projet_Python/blob/master/README.md) file.
* A README file containing the detailed explanatory note.
* A INSTALL file, telling you what to install on your machine before running the code.
* A .h5 file, it's the model files after the learning completes successfully (we asked to save it on the computer).
* A .json file, which contains the JSON format of the model.



-------------------------------------------------------------------------


### License	

[MIT](https://github.com/KevinTellier2/Projet_Python/blob/master/LICENSE)	
