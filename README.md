# Deep-MNIST
This is a repository with implementations of several deep Convolutional Neural Network (CNN) architectures for classifying the MNIST dataset.
## MNIST Dataset
The MNIST dataset consists of images of hand-written digits. Each image has a resolution of 28 x 28 pixels and a single color channel (grayscale). The dataset consists of 60,000 training images and 10,000 test images. The NIST dataset is the parent of the MNIST dataset - in other words, the MNIST is a subset of the NIST dataset. The dataset can be found [here](http://yann.lecun.com/exdb/mnist/). An example of the images of hand-written digits in the dataset are shown below.
![](sample_imgs/Input1.png) ![](sample_imgs/Input2.png) ![](sample_imgs/Input3.png) ![](sample_imgs/Input4.png) ![](sample_imgs/Input5.png)
## Convolutional Neural Networks
Deep Convolutional Neural Networks (CNN) are currently the state-of-the-art in the domain of image classification. However, there are several hyper-parameters in CNNs - the learning rate, the optimization algorithm (like SGD, Adagrad, Momentum, Adam, etc.), the batch size, the number of epochs and so on. In fact, the entire architecture of the network is a hyper-parameter. The number of Convolutional layers, the number of filters in each convolutional layer, the kernel size of each convolutional layer, the activation function used, the number of Fully Connected (FC or Dense) layers and normalization requirement of layers are all parameters that need to be experimented with, in order to be able to construct a robust model for any particular type of data. This repository aims at providing such an elaborate experimentation on the MNIST dataset.
## Software
Python is the primary software used. Packages used in addition are Theano and Keras.
