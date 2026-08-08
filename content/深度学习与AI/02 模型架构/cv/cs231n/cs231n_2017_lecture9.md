---
title: cs231n_2017_lecture9
source: converted:attachments/documents/AI_CNN-b4e086da9e0b/cs231n_2017_lecture9.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-b4e086da9e0b/cs231n_2017_lecture9.pdf
  title: cs231n_2017_lecture9.pdf
---

### Lecture 9: CNN Architectures 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 1 

**May 2, 2017** 

#### Administrative 

**A2** due Thu May 4 

**Midterm** : In-class Tue May 9. Covers material through Thu May 4 lecture. 

**Poster session** : Tue June 6, 12-3pm 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 2 May 2, 2017 

###### Last time: Deep learning frameworks 

Paddle (Baidu) 

Caffe2 (Facebook) 

Caffe Caffe2 (UC Berkeley) (Facebook) Torch PyTorch (NYU / Facebook) (Facebook) 

Theano (U Montreal) 

TensorFlow (Google) 

CNTK (Microsoft) 

MXNet (Amazon) 

Developed by U Washington, CMU, MIT, Hong Kong U, etc but main framework of choice at AWS 

And others... 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 3 May 2, 2017 

###### Last time: Deep learning frameworks 

(1) Easily build big computational graphs (2) Easily compute gradients in computational graphs (3) Run it all efficiently on GPU (wrap cuDNN, cuBLAS, etc) 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 4 May 2, 2017 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires grad=False) 

~~>~~ 

model = torch.nn.Sequential( 

torch.nn.Linear(Din, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size average=False) 

learning_rate = le-4 

for t in range(500): 

y_pred = model(x) loss = loss fn(y pred, y) 

model.zero_grad() loss. backward( ) 

for param in model.parameters(): param.data -= learning rate * param.grad.data 

##### Today: CNN Architectures 

###### Case Studies 

- AlexNet 

- VGG 

- GoogLeNet 

- ResNet 

###### Also.... 

- NiN (Network in Network) 

- Wide ResNet 

- ResNeXT 

   - DenseNet 

   - FractalNet 

   - SqueezeNet 

- Stochastic Depth 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 7 

May 2, 2017 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

* Gb x 55x50 ~ mum of parameters * each fitter (lxllx3 totally Qbx Wx IkK3= BK pawcunctews Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Gb x 2} x2F param ckev : None 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

None 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

af, 255 

###### A 

HASH =2} 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, 2012. Reproduced with permission. 

Figure copyright Kaiming He, 2016. Reproduced with permission. 

Figure copyright Kaiming He, 2016. Reproduced with permission. 

##### Case Study: VGGNet 

_[Simonyan and Zisserman, 2014]_ 

Small filters, Deeper networks 

8 layers (AlexNet) -> 16 - 19 layers (VGG16Net) 

Only 3x3 CONV stride 1, pad 1 and  2x2 MAX POOL stride 2 

11.7% top 5 error in ILSVRC’13 (ZFNet) -> 7.3% top 5 error in ILSVRC’14 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 26 

**May 2, 2017** 

##### Case Study: VGGNet 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 27 **May 2, 2017** 

##### Case Study: VGGNet 

_[Simonyan and Zisserman, 2014]_ 

Q: Why use smaller filters? (3x3 conv) 

Stack of three 3x3 conv (stride 1) layers has same as **effective receptive field** one 7x7 conv layer 

Q: What is the effective receptive field of three 3x3 conv (stride 1) layers? 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 28 

**May 2, 2017** 

##### Case Study: VGGNet 

_[Simonyan and Zisserman, 2014]_ 

Q: Why use smaller filters? (3x3 conv) 

Stack of three 3x3 conv (stride 1) layers has same as **effective receptive field** one 7x7 conv layer 

[7x7] 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 29 **May 2, 2017** 

##### Case Study: VGGNet 

_[Simonyan and Zisserman, 2014]_ 

Q: Why use smaller filters? (3x3 conv) 

Stack of three 3x3 conv (stride 1) layers has same as **effective receptive field** one 7x7 conv layer 

But deeper, more non-linearities 

And fewer parameters: 3 * (3<sup>2</sup> C<sup>2</sup> ) vs. 7<sup>2</sup> C<sup>2</sup> for C channels per layer 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 30 **May 2, 2017** 

(not counting biases) 

INPUT: [224x224x3]        memory:  224*224*3=150K params: 0 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*3)*64 = 1,728 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*64)*64 = 36,864 POOL2: [112x112x64]  memory:  112*112*64=800K params: 0 

CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*64)*128 = 73,728 CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*128)*128 = 147,456 POOL2: [56x56x128]  memory:  56*56*128=400K params: 0 

CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*128)*256 = 294,912 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 POOL2: [28x28x256]  memory:  28*28*256=200K params: 0 

CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*256)*512 = 1,179,648 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 POOL2: [14x14x512]  memory:  14*14*512=100K params: 0 

CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 POOL2: [7x7x512]  memory:  7*7*512=25K params: 0 

FC: [1x1x4096]  memory:  4096 params: 7*7*512*4096 = 102,760,448 FC: [1x1x4096]  memory:  4096 params: 4096*4096 = 16,777,216 FC: [1x1x1000]  memory:  1000 params: 4096*1000 = 4,096,000 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 31 

VGG16 

**May 2, 2017** 

(not counting biases) 

INPUT: [224x224x3]        memory:  224*224*3=150K params: 0 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*3)*64 = 1,728 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*64)*64 = 36,864 POOL2: [112x112x64]  memory:  112*112*64=800K params: 0 

CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*64)*128 = 73,728 CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*128)*128 = 147,456 POOL2: [56x56x128]  memory:  56*56*128=400K params: 0 

CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*128)*256 = 294,912 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 POOL2: [28x28x256]  memory:  28*28*256=200K params: 0 

CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*256)*512 = 1,179,648 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 POOL2: [14x14x512]  memory:  14*14*512=100K params: 0 

CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 POOL2: [7x7x512]  memory:  7*7*512=25K params: 0 

FC: [1x1x4096]  memory:  4096 params: 7*7*512*4096 = 102,760,448 FC: [1x1x4096]  memory:  4096 params: 4096*4096 = 16,777,216 FC: [1x1x1000]  memory:  1000 params: 4096*1000 = 4,096,000 

VGG16 

TOTAL memory: 24M * 4 bytes ~= 96MB / image (only forward! ~*2 for bwd) TOTAL params: 138M parameters 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 32 

**May 2, 2017** 

(not counting biases) 

INPUT: [224x224x3]        memory:  224*224*3=150K params: 0 CONV3-64: [224x224x64]  memory: **224*224*64=3.2M** params: (3*3*3)*64 = 1,728 CONV3-64: [224x224x64]  memory: **224*224*64=3.2M** params: (3*3*64)*64 = 36,864 POOL2: [112x112x64]  memory:  112*112*64=800K params: 0 

CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*64)*128 = 73,728 CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*128)*128 = 147,456 POOL2: [56x56x128]  memory:  56*56*128=400K params: 0 

Note: 

Most memory is in early CONV 

CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*128)*256 = 294,912 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 POOL2: [28x28x256]  memory:  28*28*256=200K params: 0 

CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*256)*512 = 1,179,648 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 POOL2: [14x14x512]  memory:  14*14*512=100K params: 0 

CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 POOL2: [7x7x512]  memory:  7*7*512=25K params: 0 FC: [1x1x4096]  memory:  4096 params: 7*7*512*4096 = **102,760,448** FC: [1x1x4096]  memory:  4096 params: 4096*4096 = 16,777,216 FC: [1x1x1000]  memory:  1000 params: 4096*1000 = 4,096,000 

Most params are in late FC 

TOTAL memory: 24M * 4 bytes ~= 96MB / image (only forward! ~*2 for bwd) TOTAL params: 138M parameters 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 33 

**May 2, 2017** 

(not counting biases) 

INPUT: [224x224x3]        memory:  224*224*3=150K params: 0 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*3)*64 = 1,728 CONV3-64: [224x224x64]  memory:  224*224*64=3.2M params: (3*3*64)*64 = 36,864 POOL2: [112x112x64]  memory:  112*112*64=800K params: 0 

CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*64)*128 = 73,728 CONV3-128: [112x112x128]  memory:  112*112*128=1.6M params: (3*3*128)*128 = 147,456 POOL2: [56x56x128]  memory:  56*56*128=400K params: 0 

CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*128)*256 = 294,912 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 CONV3-256: [56x56x256]  memory:  56*56*256=800K params: (3*3*256)*256 = 589,824 POOL2: [28x28x256]  memory:  28*28*256=200K params: 0 

CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*256)*512 = 1,179,648 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 CONV3-512: [28x28x512]  memory:  28*28*512=400K params: (3*3*512)*512 = 2,359,296 POOL2: [14x14x512]  memory:  14*14*512=100K params: 0 

CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 CONV3-512: [14x14x512]  memory:  14*14*512=100K params: (3*3*512)*512 = 2,359,296 POOL2: [7x7x512]  memory:  7*7*512=25K params: 0 

FC: [1x1x4096]  memory:  4096 params: 7*7*512*4096 = 102,760,448 FC: [1x1x4096]  memory:  4096 params: 4096*4096 = 16,777,216 FC: [1x1x1000]  memory:  1000 params: 4096*1000 = 4,096,000 

TOTAL memory: 24M * 4 bytes ~= 96MB / image (only forward! ~*2 for bwd) TOTAL params: 138M parameters 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 34 **May 2, 2017** 

##### Case Study: VGGNet 

###### _[Simonyan and Zisserman, 2014]_ 

###### Details: 

- ILSVRC’14 2nd in classification, 1st in localization 

- Similar training procedure as Krizhevsky 2012 

- No Local Response Normalisation (LRN) 

- - Use VGG16 or VGG19 (VGG19 only slightly better, more memory) 

- Use ensembles for best results 

- - FC7 features generalize well to other tasks 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 35 **May 2, 2017** 

Figure copyright Kaiming He, 2016. Reproduced with permission. 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Apply parallel filter operations on the input from previous layer: 

- Multiple receptive field sizes for convolution (1x1, 3x3, 5x5) 

- Pooling operation (3x3) 

###### Naive Inception module 

Concatenate all filter outputs together depth-wise 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 39 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Apply parallel filter operations on the input from previous layer: 

- Multiple receptive field sizes for convolution (1x1, 3x3, 5x5) 

- Pooling operation (3x3) 

###### Naive Inception module 

Concatenate all filter outputs together depth-wise 

Q: What is the problem with this? [Hint: Computational complexity] 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 40 **May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Example: 

Q: What is the problem with this? [Hint: Computational complexity] 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 41 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q1: What is the output size of the Example: 1x1 conv, with 128 filters? 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 42 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q1: What is the output size of the Example: 1x1 conv, with 128 filters? 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 43 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q2: What are the output sizes of Example: all different filter operations? 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 44 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q2: What are the output sizes of Example: all different filter operations? 

Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 45 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q3:What is output size after Example: filter concatenation? 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 46 

**May 2, 2017** 

##### Case Study: GoogLeNet 

###### _[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q3:What is output size after Example: filter concatenation? 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 47 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q3:What is output size after Example: filter concatenation? 

**Conv Ops:** [1x1 conv, 128]  28x28x128x1x1x256 [3x3 conv, 192]  28x28x192x3x3x256 [5x5 conv, 96]  28x28x96x5x5x256 **Total: 854M ops** 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 48 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

###### Naive Inception module 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

Q: What is the problem with this? [Hint: Computational complexity] 

**Conv Ops:** 

[1x1 conv, 128]  28x28x128x1x1x256 [3x3 conv, 192]  28x28x192x3x3x256 [5x5 conv, 96]  28x28x96x5x5x256 **Total: 854M ops** 

Very expensive compute 

Pooling layer also preserves feature depth, which means total depth after concatenation can only grow at every layer! 

**Lecture 9 -** 49 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

Q: What is the problem with this? [Hint: Computational complexity] 

Q3:What is output size after Example: filter concatenation? 

Solution: “bottleneck” layers that use 1x1 convolutions to reduce feature depth 

###### Naive Inception module 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 50 

**May 2, 2017** 

##### Reminder: 1x1 convolutions 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 51 

**May 2, 2017** 

##### Reminder: 1x1 convolutions 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 52 

**May 2, 2017** 

Case Study: GoogLeNet _[Szegedy et al., 2014]_ 

###### Naive Inception module 

###### Inception module with dimension reduction 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 53 

**May 2, 2017** 

##### Case Study: GoogLeNet 

_[Szegedy et al., 2014]_ 

1x1 conv “bottleneck” layers 

###### Naive Inception module 

###### Inception module with dimension reduction 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 54 

**May 2, 2017** 

Using same parallel layers as naive example, and adding “1x1 conv, 64 filter” bottlenecks: 

**Conv Ops:** 

[1x1 conv, 64]  28x28x64x1x1x256 [1x1 conv, 64]  28x28x64x1x1x256 [1x1 conv, 128]  28x28x128x1x1x256 [3x3 conv, 192]  28x28x192x3x3x64 [5x5 conv, 96]  28x28x96x5x5x64 [1x1 conv, 64]  28x28x64x1x1x256 **Total: 358M ops** 

Compared to 854M ops for naive version Bottleneck can also reduce depth after pooling layer 

###### Inception module with dimension reduction 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 55 

**May 2, 2017** 

# ~~<u><mark>tt</mark></u>~~ 

## ~~<u><mark>e</mark></u>~~ <u><mark>e</mark></u> ~~<u><mark>?</mark></u>~~ 

GML layers have sepanate we4g its 

Figure copyright Kaiming He, 2016. Reproduced with permission. 

##### Case Study: ResNet 

_[He et al., 2015]_ 

Very deep networks using residual connections 

- 152-layer model for ImageNet 

- - ILSVRC’15 classification winner (3.57% top 5 error) 

- - Swept all classification and detection competitions in ILSVRC’15 and COCO’15! 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 65 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

What happens when we continue stacking deeper layers on a “plain” convolutional neural network? 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 66 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

What happens when we continue stacking deeper layers on a “plain” convolutional neural network? 

Q: What’s strange about these training and test curves? [Hint: look at the order of the curves] 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 67 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

What happens when we continue stacking deeper layers on a “plain” convolutional neural network? 

- 56-layer model performs worse on both training and test error 

- -> The deeper model performs worse, but it’s not caused by overfitting! 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 68 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

Hypothesis: the problem is an _optimization_ problem, deeper models are harder to optimize 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 69 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

Hypothesis: the problem is an _optimization_ problem, deeper models are harder to optimize 

The deeper model should be able to perform at least as well as the shallower model. 

A solution by construction is copying the learned layers from the shallower model and setting additional layers to identity mapping. 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 70 

**May 2, 2017** 

##### Case Study: ResNet 

###### _[He et al., 2015]_ 

Solution: Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 71 **May 2, 2017** 

##### Case Study: ResNet 

###### _[He et al., 2015]_ 

Solution: Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 - 72 May 2, 2017** 

Case Study: ResNet _[He et al., 2015]_ 

- Full ResNet architecture: - Stack residual blocks 

- - Every residual block has two 3x3 conv layers 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 73 **May 2, 2017** 

<mark>Softmax FC 1000 Pool 3x3 conv, 512 3x3 conv, 512 3x3 conv, 512 3x3 conv, 512 3x3 conv, 512 3x3 conv, 512, /2</mark> .. . <mark>3x3 conv, 128 3x3 conv, 128</mark> 3x3 conv, 128 X <mark>3x3 conv, 128</mark> filters, /2 <mark>3x3 conv, 128</mark> identity spatially with <mark>3x3 conv, 128</mark> stride 2 <mark>3x3 conv, 128, / 2 3x3 conv, 64</mark> 3x3 conv, 64 <mark>3x3 conv, 64</mark> filters <mark>3x3 conv, 64 3x3 conv, 64 3x3 conv, 64 3x3 conv, 64 Pool 7x7 conv, 64, / 2 Input</mark> **Lecture 9 -** 74 **May 2, 2017** 

Case Study: ResNet _[He et al., 2015]_ 

- Full ResNet architecture: - Stack residual blocks 

- - Every residual block has two 3x3 conv layers 

- - Periodically, double # of filters and downsample spatially using stride 2 (/2 in each dimension) 

relu F(x) + x 3x3 conv X F(x) relu identity 3x3 conv X Residual block 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

Case Study: ResNet _[He et al., 2015]_ 

Full ResNet architecture: relu - Stack residual blocks F(x) + x - Every residual block has two 3x3 conv layers - 3x3 conv Periodically, double # of filters and downsample F(x) relu X identity spatially using stride 2 3x3 conv (/2 in each dimension) - Additional conv layer at the beginning X Residual block 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

Case Study: ResNet _[He et al., 2015]_ 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

Total depths of 34, 50, 101, or 152 layers for ImageNet 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 77 

**May 2, 2017** 

##### Case Study: ResNet 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 79 

**May 2, 2017** 

##### Case Study: ResNet 

_[He et al., 2015]_ 

###### Training ResNet in practice: 

- Batch Normalization after every CONV layer 

- Xavier/2 initialization from He et al. 

- SGD + Momentum (0.9) 

- Learning rate: 0.1, divided by 10 when validation error plateaus 

- Mini-batch size 256 

- Weight decay of 1e-5 

- No dropout used 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 -** 80 

**May 2, 2017** 

MSRA @ ILSVRC & COCO 2015 Competitions 

* 1st places in all five main tracks 

* ImageNet Classification: “Ultra-deep” (quote Yann) 152-layer nets * ImageNet Detection: 16% better than 2nd * ImageNet Localization: 27% better than 2nd 

- COCO Detection: 11% better than 2nd 

- COCO Segmentation: 12% better than 2nd 

MSRA @ ILSVRC & COCO 2015 Competitions 

* 1st places in all five main tracks 

* ImageNet Classification: “Ultra-deep” (quote Yann) 152-layer nets * ImageNet Detection: 16% better than 2nd * ImageNet Localization: 27% better than 2nd 

- COCO Detection: 11% better than 2nd 

- COCO Segmentation: 12% better than 2nd 

Figure copyright Kaiming He, 2016. Reproduced with permission. 

##### Other architectures to know... 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 9 - 91 

May 2, 2017 

( 

###### Improving ResNets... 

##### Identity Mappings in Deep Residual Networks 

_[He et al. 2016]_ 

- Improved ResNet block design from creators of ResNet 

- Creates a more direct path for propagating information throughout network (moves activation to residual mapping pathway) 

- Gives better performance 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 93 

May 2, 2017 

###### Improving ResNets... 

##### Wide Residual Networks 

_[Zagoruyko et al. 2016]_ 

- Argues that residuals are the important factor, not depth 

- User wider residual blocks (F x k filters instead of F filters in each layer) 

- - 50-layer wide ResNet outperforms 152-layer original ResNet 

- - Increasing width instead of depth more computationally efficient (parallelizable) 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 94 

May 2, 2017 

##### Improving ResNets... Aggregated Residual Transformations for Deep Neural Networks (ResNeXt) 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 95 May 2, 2017 

###### Improving ResNets... 

##### Deep Networks with Stochastic Depth 

_[Huang et al. 2016]_ 

- Motivation: reduce vanishing gradients and training time through short networks during training 

- Randomly drop a subset of layers during each training pass 

- Bypass with identity function 

- Use full deep network at test time 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 96 May 2, 2017 

###### Beyond ResNets... 

###### Densely Connected Convolutional Networks 

_[Huang et al. 2017]_ 

- Dense blocks where each layer is connected to every other layer in feedforward fashion 

- Alleviates vanishing gradient, strengthens feature propagation, encourages feature reuse 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 9 -** 

**May 2, 2017** 

##### Summary: CNN Architectures 

###### Case Studies 

- AlexNet 

- VGG 

- GoogLeNet 

- ResNet 

###### Also.... 

- NiN (Network in Network) 

- Wide ResNet 

- ResNeXT 

   - DenseNet 

   - FractalNet 

   - SqueezeNet 

- Stochastic Depth 

Fei-Fei Li & Justin Johnson & Serena Yeung 

10 Lecture 9 - 0 

May 2, 2017 

##### Summary: CNN Architectures 

- VGG, GoogLeNet, ResNet all in wide use, available in model zoos 

- ResNet current best default 

- Trend towards extremely deep networks 

- Significant research centers around design of layer / skip connections and improving gradient flow 

- Even more recent trend towards examining necessity of depth vs. width and residual connections 

- Next time: Recurrent neural networks 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 9 - 101 May 2, 2017

---

## 源文件

- [cs231n_2017_lecture9.pdf](attachments/documents/AI_CNN-b4e086da9e0b/cs231n_2017_lecture9.pdf)
