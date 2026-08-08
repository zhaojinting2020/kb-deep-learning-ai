---
title: DataLoader 完全指南
url: https://blog.paperspace.com/dataloaders-abstractions-pytorch/
fetch_source: internet_archive
fetched_at: '2026-06-27T16:26:59+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# DataLoader 完全指南

In this post, we'll deal with one of the most challenging problems in the fields of Machine Learning and Deep Learning: the struggle of loading and handling different types of data.

Say you’re already familiar with coding Neural Networks in PyTorch, and now you’re working on predicting a number using the MNIST dataset with a multilayer perceptron. In that case, you probably used the torch `DataLoader` class to directly load and convert the images to tensors. But now, in this post, we’ll learn how to go beyond the `DataLoader` class and follow the best practices that can be used while dealing with various forms of data, such as CSV files, images, text, etc. Below are the topics that we'll be covering.
- Working on Datasets
- Data Loading in PyTorch
- Looking at the MNIST Dataset in-Depth
- Transforms and Rescaling the Data
- Creating Custom Datasets in PyTorch
- Summary

You can follow along with the code and run it for free on a Gradient Community Notebook from the ML Showcase.

Bring this project to life

## Working on Datasets
If you are working on a real-time project involving Deep Learning, it's common that most of your time goes into handling data, rather than the neural network that you would build. This is because data is like fuel for your network: the more appropriate it is, the faster and the more accurate the results are! One of the main reasons for your neural network to underperform might be due to bad, or poorly understood data. Hence it is important to understand, preprocess, and load your data into the network in a more intuitive way.

In many cases, we train neural networks on default or well-known datasets like MNIST or CIFAR. While working on these, we can easily achieve accuracy greater than 90% for prediction- and classification-type problems. The reason being, these datasets are neatly organized and easy to preprocess. But when you are working on a dataset of your own, it’s quite tricky and challenging to achieve high accuracy. We’ll learn about working on custom datasets in the next sections. Before that, we’ll have a quick look at the datasets that are included in the PyTorch library.

PyTorch comes with several built-in datasets, all of which are pre-loaded in the class `torch.datasets`. Does that ring any bells? In the previous example, when we were classifying MNIST images, we used the same class to download our images. What’s in the package `torch` and `torchvision`? The package `torch` consists of all the core classes and methods required to implement neural networks, while `torchvision` is a supporting package consisting of popular datasets, model architectures, and common image transformations for computer vision. There is one more package named `torchtext` which has all the basic utilities of PyTorch Natural Language Processing. This package consists of datasets that are related to text.

Here’s a quick overview of datasets that are included in the classes `torchvision` and `torchtext`.

### Datasets in Torchvision
**MNIST:** MNIST is a dataset consisting of handwritten images that are normalized and center-cropped. It has over 60,000 training images and 10,000 test images. This is one of the most-used datasets for learning and experimenting purposes. To load and use the dataset you can import using the below syntax after the `torchvision` package is installed.
- `torchvision.datasets.MNIST()`

**Fashion MNIST: **This dataset is similar to MNIST, but instead of handwritten digits, this dataset includes clothing items like T-shirts, trousers, bags, etc. The number of training and testing samples is 60,000 and 10,000 respectively. Below is the location of FMNIST class.
- `torchvision.datasets.FashionMNIST()`

**CIFAR: **The** **CIFAR dataset has two versions, CIFAR10 and CIFAR100. CIFAR10 consists of images of 10 different labels, while CIFAR100 has 100 different classes. These include common images like trucks, frogs, boats, cars, deer, and others. This dataset is recommended for building CNNs.
- `torchvision.datasets.CIFAR10()`
- `torchvision.datasets.CIFAR100()`

**COCO: **This dataset consists of over 100,000 everyday objects like people, bottles, stationery, books, etc. This dataset of images is widely used for object detection and image captioning applications. Below is the location from which COCO can be loaded:
- `torchvision.datasets.CocoCaptions()`

**EMNIST: **This dataset is an advanced version of the MNIST dataset. It consists of images including both numbers and alphabets. If you are working on a problem that is based on recognizing text from images, this is the right dataset to train with. Below is the class:
- `torchvision.datasets.EMNIST()`

**IMAGE-NET: **ImageNet is one of the flagship datasets that is used to train high-end neural networks. It consists of over 1.2 million images spread across 10,000 classes. Usually, this dataset is loaded on a high-end hardware system as a CPU alone cannot handle datasets this big in size. Below is the class to load the ImageNet dataset:
- `torchvision.datasets.ImageNet()`

These are a few datasets that are the most frequently used while building neural networks in PyTorch. A few others include KMNIST, QMNIST, LSUN, STL10, SVHN, PhotoTour, SBU, Cityscapes, SBD, USPS, Kinetics-400. You can learn more about these from the PyTorch official documentation.

### Datasets in Torchtext
As discussed previously, `torchtext` is a supporting package that consists of all the basic utilities for Natural Language Processing. If you are new to NLP, it is a subfield of Artificial Intelligence that processes and analyzes large amounts of natural language data (mostly relating to text).

Now let's take a look at a few popular text datasets to experiment and work with.
**IMDB: **This is a dataset for sentiment classification that contains a set of 25,000 highly polar movie reviews for training, and another 25,000 for testing. We can load this data by using the following class from `torchtext`:
- `torchtext.datasets.IMDB()`

**WikiText2:** This language modelling dataset is a collection of over 100 million tokens. It is extracted from Wikipedia and retains the punctuation and the actual letter case. It is widely used in applications that involve long-term dependencies. This data can be loaded from `torchtext` as follows:
- `torchtext.datasets.WikiText2()`

Besides the above two popular datasets, there are still many more available in the `torchtext` library, such as SST, TREC, SNLI, MultiNLI, WikiText-2, WikiText103, PennTreebank, Multi30k, etc.

So far, we’ve seen datasets that are based on a predefined set of images and text. But what if you have your own? How do you load it? For now let's learn the `ImageFolder` class, which you can use to load your own image datasets.

### ImageFolder Class
`ImageFolder` is a generic data loader class in `torchvision` that helps you load your own image dataset. Let’s imagine you are working on a classification problem and building a neural network to identify if a given image is an apple or an orange. To do this in PyTorch, the first step is to arrange images in a default folder structure as shown below:
root├── orange│   ├── orange_image1.png│   └── orange_image1.png├── apple│   └── apple_image1.png│   └── apple_image2.png│   └── apple_image3.png After you arrange your dataset as shown, you can use the `ImageLoader` class to load all these images. Below is the code snippet you would use to do so:
`torchvision.datasets.ImageFolder(root, transform)`In the next section, let’s see how to load data into our programs.

## Data Loading in PyTorch
Data loading is one of the first steps in building a Deep Learning pipeline, or training a model. This task becomes more challenging when the complexity of the data increases. In this section, we will learn about the `DataLoader` class in PyTorch that helps us to load and iterate over elements in a dataset. This class is available as `DataLoader` in the `torch.utils.data` module. `DataLoader` can be imported as follows:
`from torch.utils.data import DataLoader`Let’s now discuss in detail the parameters that the `DataLoader` class accepts, shown below.
from torch.utils.data import DataLoader DataLoader(     dataset,     batch_size=1,     shuffle=False,     num_workers=0,     collate_fn=None,     pin_memory=False,  )
**1. Dataset: **The first parameter in the `DataLoader` class is the `dataset`. This is where we load the data from.
**2. Batching the data:** `batch_size` refers to the number of training samples used in one iteration. Usually we split our data into training and testing sets, and we may have different batch sizes for each.
**3. Shuffling the data:** `shuffle` is another argument passed to the `DataLoader` class. The argument takes in a Boolean value (True/False). If shuffle is set to `True`, then all the samples are shuffled and loaded in batches. Otherwise they are sent one-by-one without any shuffling.
**4. Allowing multi-processing:** As deep learning involves training models with a lot of data, running only single processes ends up taking a lot of time. In PyTorch, you can increase the number of processes running simultaneously by allowing multiprocessing with the argument `num_workers`. This also depends on the batch size, but I wouldn’t set `num_workers` to the same number because each worker loads a single batch, and returns it only once it’s ready.
- `num_workers=0`
- `num_workers=1`

**5. Merging datasets:** The `collate_fn` argument is used if we want to merge datasets. This argument is optional, and mostly used when batches are loaded from map-styled datasets.
**6. Loading data on CUDA tensors: **You can directly load datasets as CUDA tensors using the `pin_memory` argument. It is an optional parameter that takes in a Boolean value; if set to `True`, the `DataLoader` class copies Tensors into CUDA-pinned memory before returning them.

Let’s take a look at an example to better understand the usual data loading pipeline.

## Looking at the MNIST Dataset in-Depth
PyTorch’s `torchvision` repository hosts a handful of standard datasets, MNIST being one of the most popular. Now we'll see how PyTorch loads the MNIST dataset from the pytorch/vision repository. Let's first download the dataset and load it in a variable named `data_train`. Then we'll print a sample image.

# Import MNIST
from torchvision.datasets import MNIST

# Download and Save MNIST
data_train = MNIST('~/mnist_data', train=True, download=True)

# Print Data
print(data_train) print(data_train[12])

### Output
`Dataset MNIST Number of datapoints: 60000 Root location: /Users/viharkurama/mnist_data Split: Train (<PIL.Image.Image image mode=L size=28x28 at 0x11164A100>, 3)`Let’s now try extracting the tuple wherein the first value would correspond to the image, and the second value would correspond to its respective label. Below is the code snippet:
import matplotlib.pyplot as plt random_image = data_train[0][0] random_image_label = data_train[0][1]

# Print the Image using Matplotlib
plt.imshow(random_image) print("The label of the image is:", random_image_label) Most of the time you wouldn’t be accessing images with indices, but rather sending matrices containing the images to your model. This comes in handy when you need to prepare data batches (and perhaps, shuffle them before every run). Now let’s see how this works in real-time. Let’s use the `DataLoader` class to load the dataset, as shown below.
import torch from torchvision import transforms data_train = torch.utils.data.DataLoader(     MNIST(           '~/mnist_data', train=True, download=True,           transform = transforms.Compose([               transforms.ToTensor()           ])),           batch_size=64,           shuffle=True           ) for batch_idx, samples in enumerate(data_train):
      print(batch_idx, samples) This is how we load a simple dataset using `DataLoader`. However, we can’t always rely on `DataLoader` for every dataset. We often deal with large or irregular datasets containing images of asymmetric resolutions, and this is where GPUs play an important role.

### Loading the Data on GPUs
We can enable GPUs for faster training of our models. Let’s now look at the configuration of `CUDA` (GPU support for PyTorch) that can be used while loading the data. Here is an example code snippet:
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {} train_loader = torch.utils.data.DataLoader(   torchvision.datasets.MNIST('/files/', train=True, download=True),   batch_size=batch_size_train, **kwargs) test_loader = torch.utils.data.DataLoader(   torchvision.datasets.MNIST('files/', train=False, download=True),   batch_size=batch_size, **kwargs) In the above, we declared a new variable named `device`. Next, we write a simple `if` condition that checks the current hardware configuration. If it supports `GPU`, it would set the `device` to `cuda`, else it would set it to `cpu`. The variable `num_workers` denotes the number of processes that generate batches in parallel. For data loading, passing `pin_memory=True` to the `DataLoader` class will automatically put the fetched data tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.

In the next section we’ll learn about Transforms, which define the preprocessing steps for loading the data.

## Transforms and Rescaling the Data
PyTorch transforms define simple image transformation techniques that convert the whole dataset into a unique format. For example, consider a dataset containing pictures of different cars in various resolutions. While training, all the images in our train dataset should have the same resolution size. It's time-consuming if we manually convert all the images into the required input size, so we can use transforms instead; with a few lines of PyTorch code, all the images in our dataset can be converted to the desired input size and resolution. You can also resize them using the `transforms` module. The few most commonly used operations are `transforms.Resize()` to resize images, `transforms.CenterCrop()` to crop the images from the center, and `transforms.RandomResizedCrop()` to randomly resize all the images in the dataset.

Let’s now load CIFAR10 from `torchvision.datasets` and apply the following transforms:
- Resizing all the images to 32×32
- Applying a center crop transform to the images
- Converting the cropped images to tensors
- Normalizing the images

First we import the necessary modules, as well as `transforms` from the `torchvision` module. The NumPy and Matplotlib libraries are used to visualize the dataset.
import torch import torchvision import torchvision.transforms as transforms import matplotlib.pyplot as plt import numpy as np Next we'll define a variable named `transforms`, in which we write all the preprocessing steps in a sequential manner. We used the `Compose` class to chain together all the transform operations.
transform = transforms.Compose([     # resize     transforms.Resize(32),     # center-crop     transforms.CenterCrop(32),     # to-tensor     transforms.ToTensor(),     # normalize     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) ])
- `resize`: This- `Resize`transform converts all images to the defined size. In this case we want to resize all images to 32×32. Hence, we pass- `32`as an argument.
- `center-crop`: Next we crop the images using the- `CenterCrop`transform. The argument we send is also the resolution/size, but since we already resized the image to- `32x32`, the images would be center-aligned with this crop. This means the images would be cropped by 32 units from the center (both vertically and horizontally).
- `to-tensor`: We used the method- `ToTensor()`to convert the images to the- `Tensor`datatype.
- `normalize`: This normalizes all the values in the tensor so that they lie between 0.5 and 1.

In the next step, we'll load the `CIFAR` dataset into `trainset` using `trainloader`, after performing the transformations we just defined.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,                                         download=True, transform=transform) trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,                                           shuffle=False) We fetched the CIFAR dataset from `torchvision.datasets`, setting the `train` and `download` arguments to `True`. Next, we set the transform argument to the defined `transform` variable. The `DataLoader` iterable was initialized, and we passed `trainset` as an argument to it. The `batch_size` was set to `4`, and shuffle to `False`. Next, we can visualize the images using the below code snippet. Check out the corresponding Gradient Community Notebook on the ML Showcase to run the code and see the results.
classes = ('plane', 'car', 'bird', 'cat',            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') def imshow(img):
     img = img / 2 + 0.5      npimg = img.numpy()      plt.imshow(np.transpose(npimg, (1, 2, 0)))      plt.show() dataiter = iter(trainloader) images, labels = dataiter.next() imshow(torchvision.utils.make_grid(images)) print(' '.join('%5s' % classes[labels[j]] for j in range(4))) Besides `Resize()`, `CenterCrop()`, and `RandomResizedCrop()`, there are various other `Transform` classes available. Let’s look at the most-used ones.

## Transform Classes
- `RandomCrop`: This class in PyTorch crops the given PIL Image at a random location. The following are the arguments that- `RandomCrop`accepts:

`torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0)`- `size`: This argument takes an integer which indicates the desired output size of the random crop. For example, if the size is set to 32, the output will be a randomly cropped image of size 32×32.
- `padding`: This is an integer argument which is initially set to- `None`. If set to and integer, it adds an additional border to the image. For example, if the padding is set to- `4`, it pads the left, top, right, and bottom borders by 4 units each.
- `pad_if_needed`: This is an optional parameter which takes a Boolean value. If it’s set to- `True`, then it pads a smaller area around the image to avoid minimal resolution errors. By default, this parameter is set to- `False`.
- `fill`: This constant value initializes the values of all the padded pixels. The default fill value is- `0`.
2. `RandomHorizontalFlip`: Sometimes, to make the model robust while training, we flip the images randomly. The class `RandomHorizontalFlip` is used to achieve such results. It has one default argument, `p`, which indicates the probability of the image being flipped (between 0 and 1). The default value is `0.5`.

`torchvision.transforms.RandomHorizontalFlip(p=0.5)`3. `Normalize`:** **This normalizes the images, with the mean and standard deviation given as arguments. This class takes four arguments, shown below:
`torchvision.transforms.functional.normalize(tensor, mean, std, inplace=False)`- The `tensor`argument takes in a Tensor with three values: C, H, and W. They stand for the number of channels, height, and width, respectively. Based on the given argument, all the pixel values of the input images are normalized.
- The `mean`and`std`argument takes in a sequence of means and standard deviations with respect to each channel.
- The `inplace`argument is a Boolean value. If set to`True`, all the operations shall be computed in-place.
4. `ToTensor`:** **This class converts the PIL Image or a NumPy *n*-dimensional array to a tensor.

`torchvision.transforms.functional.to_tensor(img)`Now let’s understand the mechanisms behind loading a custom dataset, rather than using the built-in datasets.

## Creating Custom Datasets in PyTorch
So far, we’ve learned to load datasets along with various ways to preprocess the data. In this section, we’ll create a simple custom dataset consisting of numbers and text. We’ll talk about the `Dataset` object in PyTorch that helps to handle numerical and text files, and how one could go about optimizing the pipeline for a certain task. The trick here is to abstract the `__getitem__()` and `__len__()` methods in the Dataset class.
- The `__getitem__()`method returns the selected sample in the dataset by indexing.
- The  `__len__()`method returns the total size of the dataset. For example, if your dataset contains 1,00,000 samples, the`len`method should return 1,00,000.

Note that at this point, the data is not yet loaded into memory.

Below is an abstract view explaining the implementations of `__getitem__()` and `__len__()` methods:
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError     def __len__(self):
        raise NotImplementedError Creating a custom dataset isn’t complex, but as an additional step to the typical procedure of loading data, it is necessary to build an interface in order to get a nice abstraction (a nice syntactic sugar to say the least). Now we’ll create a new dataset that has numbers and their squared values. Let us call our dataset SquareDataset. Its purpose is to return squares of values in the range `[a,b]`. Below is the relevant code:
import torch import torchvision from torch.utils.data import Dataset, DataLoader from torchvision import datasets, transforms
class SquareDataset(Dataset):
     def __init__(self, a=0, b=1):
         super(Dataset, self).__init__()          assert a <= b          self.a = a          self.b = b      def __len__(self):
         return self.b - self.a + 1      def __getitem__(self, index):
        assert self.a <= index <= self.b
        return index, index**2 data_train = SquareDataset(a=1,b=64) data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True) print(len(data_train)) In the above code block, we created a Python class named SquareDataset** **that inherits the Dataset class from PyTorch. Next, we called an `__init__()` constructor where `a` and `b` were initialized to `0` and `1`, respectively. The `super` class is used to access the `len` and `get_item` methods from the inherited `Dataset` class. Next we used the `assert` statement to check if `a` is less than or equal to `b`, as we want to create a dataset wherein the values would lie between `a` and `b`.

We then created a dataset using the `SquareDataset`** **class, where the data values lie in the range 1 to 64. We loaded this into a variable named `data_train`. Lastly, the `Dataloader` class created an iterator over the data stored in `data_train_loader` with a `batch_size` initialized to 64, and `shuffle` set to `True`.

Data loaders exploit the goodness of Python by employing pieces of object-oriented programming concepts. A good exercise would be to go through a variety of data loaders with a number of popular datasets including CelebA, PIMA, COCO, ImageNet, CIFAR-10/100, etc.

## Summary
In this post, we’ve learned about data loading and abstraction. We started with the datasets available in the packages `torchvision`** **and** **`torchtext`, and reviewed a few popular datasets. We then learned about the `DataLoader` class, and its significance in handling the data neatly by organizing it in accordance with the given parameters. Later we analyzed the MNIST dataset in-depth by looking at various possible techniques to call it into our workspace. Data Loaders and Transforms have been introduced as well, their importance cited in the MNIST example. A deeper insight into Transforms and its classes has been put forth by explaining it through the `RandomCrop`, `RandomHorizontalFlip`, `Normalize`, `ToTensor`, and `RandomRotate` classes. Thereafter, the reasons for GPUs having an upper hand over CPUs have been explained through examples with PyTorch CUDA. The creation of a custom dataset isn’t a complex task, and this statement has been justified using a short snippet of code. The concepts and fundamentals that you’ve learned in this tutorial are all fundamental to using PyTorch.

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/DataLoader-原理深解|DataLoader 原理深解]]
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/Dataset-子集-Subset|Dataset 子集 Subset]]
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/不平衡类别-WeightedRandomSampler|不平衡类别 WeightedRandomSampler]]
