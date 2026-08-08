---
title: cs231n_2017_lecture7
source: converted:attachments/documents/AI_CNN-bb073e5c3535/cs231n_2017_lecture7.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-bb073e5c3535/cs231n_2017_lecture7.pdf
  title: cs231n_2017_lecture7.pdf
---

# Lecture 7: Training Neural Networks, Part 2 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 1 

**April 25, 2017** 

### Administrative 

- Assignment 1 is being graded, stay tuned 

- Project proposals due today by 11:59pm 

- Assignment 2 is out, due Thursday May 4 at 11:59pm 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 2 

**April 25, 2017** 

### Administrative: Google Cloud 

- STOP YOUR INSTANCES when not in use! 

- Keep track of your spending! 

- GPU instances are much more expensive than CPU instances - only use GPU instance when you need it 

   - (e.g. for A2 only on TensorFlow / PyTorch notebooks) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 4 **April 25, 2017** 

<mark>o(0) max(0.lx, x = pha</mark> | <mark>Ole)</mark> / <mark>x + bi, w4x + be) tanh(x) —fmax(wi max(0,</mark> >0 <mark>x) _/ Se 1) ; <0 Sf</mark> 

> <sup><mark>ite</mark></sup> <mark>o(x)</mark><sup><mark>=</mark></sup> | <mark>tanh(x) Jf. max(0, x) /</mark> 

<mark>( ) max(0.1z, x</mark> /<sup><mark>bi, w4x+be)</mark></sup> <mark>max(wi</mark><sup><mark>x +</mark></sup> > 0 <mark>Se 1) ; <0 Sf</mark> 

### Last time: Data Preprocessing 

**Before normalization** : classification loss very sensitive to changes in weight matrix; hard to optimize 

**After normalization** : less sensitive to small changes in weights; easier to optimize 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 9 

**April 25, 2017** 

### Today 

#### - Fancier optimization - Regularization - Transfer Learning 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 13 

**April 25, 2017** 

# Vanilla Gradient Descent 

while True: weights grad = evaluate gradient(loss fun, data, weights) weights += - step size * weights grad # perform parameter update 

### Optimization: Problems with SGD 

What if loss changes quickly in one direction and slowly in another? What does gradient descent do? 

Loss function has high **condition number** : ratio of largest to smallest singular value of the Hessian matrix is large 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 15 **April 25, 2017** 

### Optimization: Problems with SGD 

What if loss changes quickly in one direction and slowly in another? What does gradient descent do? 

Very slow progress along shallow dimension, jitter along steep direction 

Loss function has high **condition number** : ratio of largest to smallest singular value of the Hessian matrix is large 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 16 **April 25, 2017** 

### Optimization: Problems with SGD 

What if the loss function has a **local minima** or **saddle point** ? 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 17 **April 25, 2017** 

### Optimization: Problems with SGD 

What if the loss function has a **local minima** or **saddle point** ? 

Zero gradient, gradient descent gets stuck 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 18 

**April 25, 2017** 

### Optimization: Problems with SGD 

What if the loss function has a **local minima** or **saddle point** ? 

###### Saddle points much more common in high dimension 

Dauphin et al, “Identifying and attacking the saddle point problem in high-dimensional non-convex optimization”, NIPS 2014 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 19 

**April 25, 2017** 

Vo = 0 

tayrent velocity iS OF b previous velocity. Ut+1 = (OUt + f(a) @u Lt+1 = Lt — AVE41 — Dan hebptw ovss.the Saddle point vx = 0 @)22 Solve pooror conoliHanningHanningning while True: © nse fom minibatchbatch gets averaged dx = compute_gradient(x) vx = rho * vx + dx x += learning_rate * vx 

<mark>Lt41 = Lt — aV f(x)</mark> whilejj True: dx = compute_gradient(x) x += learning_rate * dx 

### SGD + Momentum 

###### Gradient Noise 

Local Minima Saddle points 

Poor Conditioning 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 22 **April 25, 2017** 

### SGD + Momentum 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 23 

**April 25, 2017** 

### Nesterov Momentum 

Nesterov, “A method of solving a convex programming problem with convergence rate O(1/k^2)”, 1983 Nesterov, “Introductory lectures on convex optimization: a basic course”, 2004 Sutskever et al, “On the importance of initialization and momentum in deel learning”, ICML 2013 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 24 

**April 25, 2017** 

Uis1 = PUt — aV f <mark>(ve + pvr</mark> 

<mark>xr, Vf (xt)</mark> 

Vi41 = pz — Af <mark>V(Xz + Pry]</mark> Lt+1 = Le + Ve4+1 

Ly = Xe + pry Wid = Poa = aV f (Zt) Le+1-= tt — Pve + + (1 + p44 p44 =F + VUt41 + p(ve4 = Ut) 

dx = compute_gradient(x) old_v =v_v =v =v v = rho * v - learning_rate * dx x += -rho * old_v + (1 + rho) * v 

~~es a es~~ to land in @ flat yninvma ‘ Usually we vant + Momentum will jump onan fuse very Sharp mimi ay which usally Coma tem tem overt ting will dlis op op pea with lavgor tray tray data. 

grad_squared = 0 while True: dx = compute_gradient(x) <mark>grad_squared += dx</mark> * <mark>dx</mark> x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7) 

A 1 AXw oo eee SOO Me > » A 2 olxe+t AX 4 d\n + \xe-t Ne + AX; +. +AXn 

grad_squared = 0 while True: dx = compute_gradient(x) grad_squared += dx * dx 

grad_squared = 0 while True: dx = compute_gradient(x) grad_squared += dx * dx 

grad_squared = 0 while True: dx = compute_gradient(x) x -= learning_rate * dx / (np.sqrt(grad_squared) + ie-7) 

grad_squared = 0 while True: dx = compute_gradient(x x -= learning_rate * dx / (np.sqrt(grad_squared) + i1e-7) 

first_moment = 0 second_moment = 0 while True: dx = compute_gradient(x) first_moment = betai * first_moment + (1 - betai) * dx second_moment = beta second_moment + - beta dx ax x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)) \\\ or cl ; by 2ev0 alee sure. a a cele N BK © Wwe ta-betar) : oO first <mark>=(4-</mark> beta) <mark>Ax</mark> Sseond <mark>=</mark> (4- betaz) <mark>dX</mark> reer very close to 2eW0 

first_moment = 0 second_moment = 0 for t in range(num_iterations): dx_= compute gradient(x second_unbias = second_moment / (i - beta2 ** t 

first_moment = 0 second_moment = 0 for t in range(i, num_iterations): dx = compute _gradient(x 

move jommest in SGD Momentum less LOMINM IN adam. a=— age _ pp. first id nest leavning vate without” deog , then “try adjust lea C not fist ovalai’ hu par parameter) a= ag /(1 tT kt) 

### First-Order Optimization 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 43 **April 25, 2017** 

### First-Order Optimization 

(1) Use gradient form linear approximation (2) Step to minimize the approximation Loss 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 44 **April 25, 2017** 

### Second-Order Optimization 

- (1) Use gradient **and Hessian** to form **quadratic** approximation (2) Step to the **minima** of the approximation 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 45 **April 25, 2017** 

— newton step 

J(8) 1 = J(80) + (8 — @) ' VoI(8o) + 5 (9 — &)'H(8 — 8) 

0* =0, —H <mark>'VoJ(8o)</mark> 

nod learning rote 

J(0) <mark>=</mark> J(8p) <mark>+</mark> (0 ~ 6) " Vo (Oo) <mark>+</mark> 5 (8 — %)"H(0 — <mark>0)</mark> 

(O J(Oo) + (8 —@) ' <mark>Vo</mark> et (8) + <mark>=(0</mark> —@) <mark>'H(0</mark> — 8 

0* =0, <mark>-H 'VoJ</mark> (80) 

##### 6* =0,—H ‘VoJ <mark>(9)</mark> 

##### 6* =0,—H ‘VoJ <mark>(9)</mark> 

#### **L-BFGS** 

- **Usually works very well in full batch, deterministic mode** i.e. if you have a single, deterministic f(x) then L-BFGS will probably work very nicely 

- **Does not transfer very well to mini-batch setting** . Gives bad results. Adapting L-BFGS to large-scale, stochastic setting is an active area of research. 

Le et al, “On optimization methods for deep learning, ICML 2011” 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 51 

**April 25, 2017** 

### <mark>In practice:</mark> 

- **Adam** is a good default choice in most cases 

- If you can afford to do full batch updates then try out **L-BFGS** (and don’t forget to disable all sources of noise) 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 52 

**April 25, 2017** 

### Model Ensembles 

#### 1. Train multiple independent models 2. At test time average their results 

#### Enjoy 2% extra performance 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 54 

**April 25, 2017** 

data batch = dataset.sampledata batch() loss = network. forward(data batch) dx = network.backward() x += - learning rate * dx x test G.390"x% test 0.005*x 

not too commen in packer 

<mark>L=5 int do j4y, max(0, f(wi; “er ~ Flee W )y, + 1) +|AR()</mark> 

RWW) = do, Wi, R(W) = doy 2a 2a |Wee| 

### Regularization: Dropout 

In each forward pass, randomly set some neurons to zero Probability of dropping is a hyperparameter; 0.5 is common 

Srivastava et al, “Dropout: A simple way to prevent neural networks from overfitting”, JMLR 2014 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**April 25, 2017** 

**Lecture 7 -** 60 

p = 0.5 # probability of keeping a unit active. higher = less dropout 

def train_step(X): """ X contains the data """ 

H1 = np.maximum(0, np.dot(W1, X) + bl) U1 = np.random.rand(*Hl.shape) < p # first dropout mas! H1 *= U1 # drop! = of then ding ext el activations H2 = np.maximum(0, np.dot(W2, H1) + b2) U2 = np.random.rand(*H2.shape) < p # second dropout mash H2 *= U2 # drop! out = np.dot(W3, H2) + b3 

Regularization: Dropout How can this possibly be a good idea? Forces the network to have a redundant representation; Prevents co-adaptation of features 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 62 

**April 25, 2017** 

### Regularization: Dropout How can this possibly be a good idea? 

Another interpretation: 

Dropout is training a large **ensemble** of models (that share parameters). Each binary mask is one model 

An FC layer with 4096 units has 2<sup>4096</sup> ~ 10<sup>1233</sup> possible masks! Only ~ 10<sup>82</sup> atoms in the universe... 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 63 

**April 25, 2017** 

y = fle) = E.[f(2,2)] = <mark>/</mark> p(2) f(a, z)dz 

## y = fle) = E.[f(a,2)] = <mark>/</mark> pz) f(a, 2)de 

y= f(a) = E,[t(e,2] = | vle) Fle, 2)ae 

~~A~~ 

E la] = wW1xr + Wey 

y= f(a) = E,[t(e,2] = | vle) Fle, 2)ae 

with 40°h aropo 

E la] = W1r+ Woy Ela =F (wie + wey) + (wis + Oy) (wie + wey) + (wis + Oy) + wey) + (wis + Oy) wey) + (wis + Oy) + (wis + Oy) (wis + Oy) + Oy) Oy) + 7 (02 + Oy) + (02 + Oy) + + Oy) + Oy) + + 7 (02 + wy) (02 + wy) + wy) =S(wia + wey) + wey) wey) 

y= f(a) = E,[t(e,2] = | vle) Fle, 2)ae 

E la] = W1r+ Woy Ela =F (wie + wey) + (wis + Oy) (wie + wey) + (wis + Oy) + wey) + (wis + Oy) wey) + (wis + Oy) + (wis + Oy) (wis + Oy) + Oy) Oy) + “(0 “(0 + Oy) + Oy) + + 7 (Ox + way) 1 =5 (wiz + way) 

###### def predict(X): 

Hl = np.maximum(0, np.dot(W1l, X) + bl) * p # NOTE: scale the activations H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations out = np.dot(W3, H2) + b3 

(see notes below) 

""" 

""""Vanilla Dropout: Not recommended implementation 

p = 0.5 # probability of keepii tive. highe ess dr def train_step(X): """"X contains the data """ H1 = np.maximum(9, np.dot(W1, X) + bl) Ul = np.random.rand(*H1l.shape) < p first ash Hl *= Ul # drop! H2 = np.maximum(0, np.dot(W2, HI) + D U2 = np.random.rand(*H2.shape) < p # second dropout mas} H2 *= U2 # : out = np.dot(W3, H2) + b3 

def predict(X): 

H1 = np.maximum(0, np.dot(W1l, X) + bl)]* p # NOTE: scale t iC ati H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NO the out = np.dot(W3, H2) + b3 

y = fle) = E.[f(2,2)] = <mark>/</mark> p(2) f(a, 2)dz 

y = fw(z, Zz) 

<mark>y=</mark> f(a) = E.[f(e.2)| = E.[f(e.2)| E.[f(e.2)| <mark>=</mark> f e)fl@,2)de e)fl@,2)de eee 

### Regularization: Data Augmentation 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 74 

**April 25, 2017** 

### Regularization: Data Augmentation 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 75 

**April 25, 2017** 

### Data Augmentation Random crops and scales 

- **Training** : sample random crops / scales ResNet: 

1. Pick random L in range [256, 480] 2. Resize training image, short side = L 3. Sample random 224 x 224 patch 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 77 

**April 25, 2017** 

### Data Augmentation Random crops and scales 

- **Training** : sample random crops / scales ResNet: 

1. Pick random L in range [256, 480] 2. Resize training image, short side = L 3. Sample random 224 x 224 patch 

- **Testing** : average a fixed set of crops ResNet: 

1. Resize image at 5 scales:  {224, 256, 384, 480, 640} 2. For each size, use 10 224 x 224 crops: 4 corners + center, + flips 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 78 

**April 25, 2017** 

### Data Augmentation Get creative for your problem! 

Random mix/combinations of : 

- translation 

- - rotation 

- stretching - shearing, - lens distortions, …  (go crazy) 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 81 **April 25, 2017** 

Regularization: A common pattern **Training** : Add random noise **Testing** : Marginalize over the noise 

: **Examples** Dropout Batch Normalization Data Augmentation 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 82 

**April 25, 2017** 

Regularization: A common pattern **Training** : Add random noise **Testing** : Marginalize over the noise 

: **Examples** Dropout Batch Normalization Data Augmentation DropConnect 

Wan et al, “Regularization of Neural Networks using DropConnect”, ICML 2013 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 83 

**April 25, 2017** 

Regularization: A common pattern **Training** : Add random noise **Testing** : Marginalize over the noise : **Examples** Dropout Batch Normalization Data Augmentation DropConnect Fractional Max Pooling Stochastic Depth 

Huang et al, “Deep Networks with Stochastic Depth”, ECCV 2016 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 85 **April 25, 2017** 

# Transfer Learning 

#### “You need a lot of a data if you want to train/use CNNs” 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 86 

**April 25, 2017** 

# Transfer Learning 

“You need a lot of a data if you want to train/use CNNs” **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 87 

**April 25, 2017** 

###### Transfer Learning with CNNs 

Donahue et al, “DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition”, ICML 2014 Razavian et al, “CNN Features Off-the-Shelf: An Astounding Baseline for Recognition”, CVPR Workshops 2014 

###### 1. Train on Imagenet 

**<mark>FC-1000 FC-4096 FC-4096 MaxPool Conv-512 Conv-512 MaxPool Conv-512 Conv-512 MaxPool Conv-256 Conv-256 MaxPool Conv-128 Conv-128 MaxPool Conv-64 Conv-64 Image</mark>** 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 88 

**April 25, 2017** 

Donahue et al, “DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition”, ICML 2014 Razavian et al, “CNN Features Off-the-Shelf: An Astounding Baseline for Recognition”, CVPR Workshops 2014 

###### Transfer Learning with CNNs 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 89 

**April 25, 2017** 

###### Transfer Learning with CNNs 

###### 1. Train on Imagenet 

###### 2. Small Dataset (C classes) 

Donahue et al, “DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition”, ICML 2014 Razavian et al, “CNN Features Off-the-Shelf: An Astounding Baseline for Recognition”, CVPR Workshops 2014 

###### 3. Bigger dataset 

|**FC-1000**|**FC-C**||**FC-C**||
|---|---|---|---|---|
|**FC-4096**|**FC-4096**|Reinitialize|**FC-4096**|Train these|
|**FC-4096**|**FC-4096**|<br>this and train|**FC-4096**||
|**MaxPool**|**MaxPool**||**MaxPool**||
|**Conv-512**|**Conv-512**||**Conv-512**|With bigger|
|**Conv-512**|**Conv-512**||**Conv-512**|<br>dataset, train|
|**MaxPool**|**MaxPool**||**MaxPool**|more layers|
|**Conv-512**|**Conv-512**||**Conv-512**||
|**Conv-512**|**Conv-512**||**Conv-512**||
|**MaxPool**|**MaxPool**|Freeze these|**MaxPool**||
|**Conv-256**|**Conv-256**||**Conv-256**|Freeze these|
|**Conv-256**|**Conv-256**||**Conv-256**||
|**MaxPool**|**MaxPool**||**MaxPool**||
|**Conv-128**|**Conv-128**||**Conv-128**|Lower learning rate|
|**Conv-128**|**Conv-128**||**Conv-128**|when finetuning;|
|**MaxPool**|**MaxPool**||**MaxPool**|1/10 of original LR|
|**Conv-64**|**Conv-64**||**Conv-64**|is good starting|
|**Conv-64**|**Conv-64**||**Conv-64**|<br>point|
|**Image**|**Image**||**Image**||

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 7 -** 90 

**April 25, 2017** 

|**FC-4096**<br>**FC-4096**<br>**FC-1000**|||**very similar**<br>**dataset**|**very different**<br>**dataset**|
|---|---|---|---|---|
|**MaxPool**|||||
|**Conv-512**|||||
|**Conv-512**<br>**MaxPool**<br>**Conv-512**|More specific|**very little data**|?|?|
|**Conv-512**|||||
|**MaxPool**|||||
|**Conv-256**|||||
|**Conv-256**|More generic||||
|**MaxPool**|||||
|**Conv-128**<br>**Conv-128**||**quite a lot of**|?|?|
|**Conv-64**<br>**MaxPool**||**data**|||
|**Conv-64**|||||
|**Image**|||||

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 91 

**April 25, 2017** 

|**FC-4096**<br>**FC-4096**<br>**FC-1000**|||**very similar**<br>**dataset**|**very different**<br>**dataset**|
|---|---|---|---|---|
|**MaxPool**|||||
|**Conv-512**|||||
|**Conv-512**<br>**MaxPool**<br>**Conv-512**|More specific|**very little data**|Use Linear<br>Classifier on|?|
|**MaxPool**<br>**Conv-512**|||top layer||
|**Conv-256**|||||
|**Conv-256**|More generic||||
|**MaxPool**|||||
|**Conv-128**<br>**Conv-128**||**quite a lot of**|Finetune a|?|
|**Conv-64**<br>**MaxPool**||**data**|few layers||
|**Conv-64**|||||
|**Image**|||||

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 92 

**April 25, 2017** 

|**FC-4096**<br>**FC-4096**<br>**FC-1000**|||**very similar**<br>**dataset**|**very different**<br>**dataset**|
|---|---|---|---|---|
|**MaxPool**|||||
|**Conv-512**|||||
|**Conv-512**<br>**MaxPool**<br>**Conv-512**|More specific|**very little data**|Use Linear<br>Classifier on|You’re in<br>trouble… Try|
|**Conv-256**<br>**MaxPool**<br>**Conv-512**|||top layer|linear classifier<br>from different|
|**Conv-256**|More generic|||stages|
|**MaxPool**|||||
|**Conv-128**<br>**Conv-128**||**quite a lot of**|Finetune a|Finetune a|
|**Image**<br>**Conv-64**<br>**Conv-64**<br>**MaxPool**||**data**|few layers|larger number<br>of layers|

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 93 

**April 25, 2017** 

**Takeaway for your projects and beyond:** Have some dataset of interest but it has < ~1M images? 

1. Find a very large dataset that has similar data, train a big ConvNet there 

2. Transfer learn to your dataset 

Deep learning frameworks provide a “Model Zoo” of pretrained models so you don’t need to train your own - Caffe: https://github.com/BVLC/caffe/wiki/Model <u>Zoo</u> TensorFlow: https://github.com/tensorflow/models PyTorch: https://github.com/pytorch/vision 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 97 **April 25, 2017** 

### Summary 

- Optimization 

#### - Momentum, RMSProp, Adam, etc 

#### - Regularization - Dropout, etc 

- Transfer learning - Use this for your projects! 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 98 **April 25, 2017** 

### Next time: Deep Learning Software! 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 7 -** 99 

**April 25, 2017**

---

## 源文件

- [cs231n_2017_lecture7.pdf](attachments/documents/AI_CNN-bb073e5c3535/cs231n_2017_lecture7.pdf)
