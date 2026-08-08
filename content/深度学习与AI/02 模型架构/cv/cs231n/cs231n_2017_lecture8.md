---
title: cs231n_2017_lecture8
source: converted:attachments/documents/AI_CNN-7a2d1dec3d03/cs231n_2017_lecture8.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-7a2d1dec3d03/cs231n_2017_lecture8.pdf
  title: cs231n_2017_lecture8.pdf
---

## Lecture 8: Deep Learning Software 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 1 **April 27, 2017** 

#### Administrative 

- Project proposals were due Tuesday 

- We are assigning TAs to projects, stay tuned 

- - We are grading A1 

- A2 is due Thursday 5/4 

   - Remember to **stop your instances** when not in use 

   - Only use GPU instances for the **last notebook** 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -2 2 April 27, 2017 

#### Last time 

###### **Regularization:** Dropout 

###### **Transfer Learning** 

**Optimization** : SGD+Momentum, Nesterov, RMSProp, Adam 

**<mark>FC-C FC-4096</mark>** Reinitialize **<mark>FC-4096</mark>** this and train **<mark>MaxPool Conv-512 Conv-512 MaxPool Conv-512 Conv-512 MaxPool</mark>** Freeze these **<mark>Conv-256 Conv-256 MaxPool Conv-128 Conv-128 MaxPool Conv-64 Conv-64 Image</mark>** 

**Regularization** : Add noise, then marginalize out Train Test 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 

3 April 27, 2017 

### Today 

- CPU vs GPU 

- Deep Learning Frameworks 

   - Caffe / Caffe2 

   - Theano / TensorFlow 

   - Torch / PyTorch 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -4 4 April 27, 2017 

## CPU vs GPU 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 5 **April 27, 2017** 

# NVIDIA <mark>vs</mark> 

AMD 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -9 9 April 27, 2017 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -1010 April 27, 2017 

#### CPU vs GPU 

||**# Cores**|**Clock Speed**|**Memory**|**Price**|**CPU**: Fewer cores,<br>|
|---|---|---|---|---|---|
|**CPU**<br>(Intel Core<br>i7-7700k)|4<br>(8 threads with<br>hyperthreading<br>)|4.4 GHz|Shared with system|$339|but each core is<br>much faster and<br>much more<br>capable; great at<br>|
|**CPU**<br>(Intel Core<br>i7-6950X)|10<br>(20 threads<br>with<br>hyperthreading<br>|3.5 GHz|Shared with system|$1723|sequential tasks|
|**GPU**<br>(NVIDIA<br>Titan Xp)|)<br>3840|1.6 GHz|12 GB GDDR5X|$1200|**GPU**: More cores,<br>but each core is<br>much slower and<br>“dumber”; great for|
|**GPU**<br>(NVIDIA<br>GTX 1070)|1920|1.68 GHz|8 GB GDDR5|$399|parallel tasks|
|Fei-Fei Li &|Justin Joh|nson & Sere|na Yeung<br>L|ecture 8 -<br>11<br>11|April 27, 2017|

Example: Matrix Multiplication A x B A x C B x C <mark>=</mark> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -1212 April 27, 2017 

#### Programming GPUs 

- CUDA (NVIDIA only) 

- Write C-like code that runs directly on the GPU 

- ○ Higher-level APIs: cuBLAS, cuFFT, cuDNN, etc 

- ● OpenCL 

   - Similar to CUDA, but runs on anything 

   - ○ Usually slower :( 

- Udacity: Intro to Parallel Programming <u>https://www.udacity.com/course/cs344</u> 

   - For deep learning just use existing libraries 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -1313 April 27, 2017 

ME Intel E5-2620 v3. MJ Pascal Titan X (no cuDNN) __‘ Pascal Titan X (cuDNN 5.1) 

24000 

7 Ss co) g g S x % & + & 5 nn = Ty = 

ME Intel E5-2620 v3. MJ Pascal Titan X (no cuDNN) __‘ Pascal Titan X (cuDNN 5.1) 

24000 

7 Ss co) g g S x % & + gS & 5 nn = Ty = 

## Deep Learning Frameworks 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 18 

**April 27, 2017** 

Last year ... 

Caffe (UC Berkeley) 

Torch (NYU / Facebook) Theano (U Montreal) 

TensorFlow (Google) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 19 **April 27, 2017** 

#### This year ... 

Paddle (Baidu) 

Caffe (UC Berkeley) 

Caffe2 (Facebook) 

CNTK (Microsoft) 

Torch (NYU / Facebook) 

Theano (U Montreal) 

PyTorch (Facebook) 

TensorFlow (Google) 

MXNet (Amazon) 

Developed by U Washington, CMU, MIT, Hong Kong U, etc but main framework of choice at AWS 

And others... 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 20 **April 27, 2017** 

#### Today 

##### A bit about these 

Paddle (Baidu) 

Caffe (UC Berkeley) 

Torch (NYU / Facebook) 

Theano (U Montreal) 

Caffe2 (Facebook) PyTorch (Facebook) 

TensorFlow (Google) Mostly these 

CNTK (Microsoft) 

MXNet (Amazon) 

Developed by U Washington, CMU, MIT, Hong Kong U, etc but main framework of choice at AWS 

And others... 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 21 **April 27, 2017** 

hinge loss 1 muti class Classification: boy) - et ymax (0) 1+ Wyx — Wer’) <mark>Li = Dv, MAx(0, Dv, MAx(0, MAx(0,</mark> 6) <mark>— 6, +1) R(W)</mark> ~~<u>e4 5</u>~~ 

### The point of deep learning frameworks 

- (1) Easily build big computational graphs 

(2) Easily compute gradients in computational graphs (3) Run it all efficiently on GPU (wrap cuDNN, cuBLAS, etc) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -2525 April 27, 2017 

import numpy as np np. random. seed(0) N, D= 3, 4 X = np.random.randn(N, D) y = np.random.randn(N, D) zZ = np.random.randn(N, D) 

b=at+z c = np.sum(b) 

import numpy as np np. random. seed(0) 

N, D = 3, 4 

X = np.random.randn(N, D) y = np.random.randn(N, D) zZ = np.random.randn(N, D) 

a=x*y b=at+z c = np.sum(b) grad_c = 1.0 grad_b = grad_c * np.ones((N, D)) grad_a = grad_b.copy() grad_z = grad_b.copy() gradx = grada*y grad_y = grad_a * x 

import numpy as np np. random. seed(0) 

N, D = 3, 4 

X = np.random.randn(N, D) y = np.random.randn(N, D) zZ = np.random.randn(N, D) 

a=x*y b=at+z c = np.sum(b) grad_c = 1.0 grad_b = grad_c * np.ones((N, D)) grad_a = grad_b.copy() grad_z = grad_b.copy() gradx = grada*y grad_y = grad_a * x 

import numpy as np np. random. seed(0) N, D= 3, 4 

X = np.random.randn(N, D) y = np.random.randn(N, D) zZ = np.random.randn(N, D) 

Gir eal Mg b=at+z c = np.sum(b) = grad_c grad_b == grad_c1.0 * np.ones((N, D)) grad_a = grad_b.copy() grad_z = grad_b.copy() grad x = grad a* y grady = grad_a* x 

# Basic computational graph import numpy as np np. random. seed(0) import tensorflow as tf N, D= 3, 4 

x = tf.placeholder(tf£.float32) y = tfi.placeholder(tf.float32) z = tf.placeholder(tf.float32) 

a =a ste!7 c = t£.reduce_sum(b) grad_x, grad_y, grad_z = tf.gradients(c, [x, y, 2Z]) with tf.Session() as sess: values = { X: np.random.randn(N, D), y:zt np.random.randn(N,np.random.randn(N, D),D), } out = sess.run([c, grad_x, grad_y, grad_z], ‘ ? Reh MRE AL ; c_val, grad_x_ val, grad_y val, grad_z val = out 

# Basic computational graph import numpy as np np. random. seed(0) import tensorflow as tf 

N, D = 3, 4 x = tf.placeholder(tf£.float32) y = tf£.placeholder(tf.float32) z = tf£.placeholder(tf.float32) ~~—__——_p~~ > a=x*y b=at+z c = tfi.reduce sum(b) 

grad_x, grad_y, grad_z = tf.gradients(c, with tf.Session() as sess: values = { X: np.random.randn(N, D), y: np.random.randn(N, D), Zz: np.random.randn(N, D), 

[x, y, 2Z]) 

out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values) c_val, grad_x_val, grad_y val, grad_z_val = out 

# Basic computational graph import numpy as np np. random. seed(0) import tensorflow as tf 

N, D= 3, 4 

x = tf.placeholder(tf£.float32) y = tf£.placeholder(tf.float32) z = tf£.placeholder(tf.float32) a=x*y b=at+z c = tf£.reduce_sum(b) ~~——_—_>~~ grad_x, grad_y, grad_z = tf.gradients(c, [x, y, 2Z]) with tf.Session() as sess: values = { X: np.random.randn(N, D), y: np.random.randn(N, D), Zz: np.random.randn(N, D), out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values) c_val, grad_x_val, grad_y val, grad_z_val = out 

import numpy as np np. random. seed(0) import tensorflow as tf 

N, D = 3000, 4000 with tf.device('/cpu:0'): eae o=a 7float32) y = tf.placeholder(tf.float32) z = tf.placeholder(tf.float32) 

a=x*y b=atz c = tf.reduce_sum(b) grad_x, grad_y, grad_z = tf.gradients(c, [x, y, 2]) with tf.Session() as sess: values = { xX: np.random.randn(N, D), y: np.random.randn(N, D), Zz: np.random.randn(N, D), out = sess.run([c, grad_x, grad_y, grad_2z], feed_dict=values) c_val, grad_x_val, grad_y val, grad_z val = out 

import numpy as np np. random. seed(0) import tensorflow as tf 

N, D = 3000, 4000 with tf.device('/gpu:0'): Sa oa ~float32) y = tf.placeholder(tf.float32) z = tf.placeholder(tf.float32) 

a=x*y b=a+z c = tf.reduce_sum(b) grad_x, grad_y, grad_z = tf.gradients(c, [x, y, 2]) with tf.Session() as sess: values = { xX: np.random.randn(N, D), y: np.random.randn(N, D), Zz: np.random.randn(N, D), } out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values) c_val, grad_x_val, grad_y_val, grad_z_val = out 

import torch from torch.autograd import Variable 

N, D= 3, 4 

x = Variable(torch.randn(N, D), requires _grad=True) y = Variable(torch.randn(N, D), requires grad=True) Zz = Variable(torch.randn(N, D}), requires _grad=True) 

a=x*y b=a+t+tz c = torch.sum(b) 

c.backward() print(x.grad.data) print(y.grad.data) print(z.grad.data) 

import torch from torch.autograd import Variable 

N, D= 3, 4 x = Variable(torch.randn(N, D), requires _grad=True) y = Variable(torch.randn(N, D), requires grad=True) Zz = Variable(torch.randn(N, D}), requires _grad=True) a=x*y b=a+t+tz c = torch.sum(b) c.backward() print(x.grad.data) print(y.grad.data) print(z.grad.data) 

import torch from torch.autograd import Variable 

N, D= 3, 4 x = Variable(torch.randn(N, D), requires _grad=True) y = Variable(torch.randn(N, D), requires grad=True) Zz = Variable(torch.randn(N, D}), requires _grad=True) 

a=x*y b=a+t+tz c = torch.sum(b) 

c.backward() 

print(x.grad.data) print(y.grad.data) print(z.grad.data) 

import torch from torch.autograd import Variable 

N, D= 3, 4 

x = Variable(torch.randn(N, D), requires _grad=True) y = Variable(torch.randn(N, D), requires grad=True) Zz = Variable(torch.randn(N, D}), requires _grad=True) 

a=x*y b=a+t+tz c = torch.sum(b) 

c.backward() print(x.grad.data) print(y.grad.data) print(z.grad.data) 

import torch from torch.autograd import Variable 

N, D= 3, 4 

xX = Variable(torch.randn(N, D)j.cuda(), requires_grad=Trupe) | ea Tab = = * = = Vy 1 -cuda(), requires grad=Trupe) z = Variable(torch.randn(N, D)j.cuda(), requires grad=True 

a=x* y b=at+z2z c = torch.sum(b) 

c.backward() 

print(x.grad.data) print(y.grad.data) print(z.grad.data) 

aepesh import numpy as np np. random.Bumpyseed(0)Ss random.Bumpyseed(0)SsBumpyseed(0)Ssseed(0)SsSs oP np.import random. ten **s** orfloweed(0) as tf N, D=== 3, 4 N,; : D= 3, 4 with tf.device('/gpu:0'): x === np.random.randn(N, D) x ee= t£.placeholder(tf.float32) Y np.random.randn(N, D y = tf£.placeholder(tf.float32) _ zZ P sie hes z = tf.placeholder(tf.float32) = np.random.randn(N, D) a=x*y a=x * y b=a+z b=at+z c = tf.reduce_sum(b) c = np.sum(b) grad_x, grad_y, grad_z = 

c = np.sum(b) grad_x, grad_y, grad_z = tf.gradients(c, [x, y, 2Z]) grad_c = 1.0 with tf.Session() as sess: age values = { grad_b = grad_c * np.ones((N, D)) xX: np.random.randn(N, D), grad_a = grad_b.copy() y: np.random.randn(N, D), grad_z = grad_b.copy() : z: np-random.randn(N, D), grad_xgrad_ygrad_y === **grad_a** *** Yxx out = sess.run([c,feed_dict=values)grad_x, grad_y, grad_z], c_val, grad_x_val, grad_y val, grad_z_val = out 

import torch pt torch.autograd import Variable 

N, D= 3, 4 x = Variable(torch.randn(N, D).cuda(), = einkVariable(torch.randnjeobchiesed es Bice.cuda a Y : , ' requires_grad=True) z = Variable(torch.randn(N, D).cuda(), requires_grad=True) 

a=x*y b=a+z c = torch.sum(b) 

c.backward() 

print (x.grad.data) print(y.grad.data) print (z.grad.data) 

TensorFlow (more detail) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -4040 April 27, 2017 

import numpy as np import tensorflow as tf 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad _w2 = tf.gradients(loss, [wl, w2]) 

with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 

x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([loss, grad wl, grad_w2], lossssval, grad—_wlfeed_dict=values)val, gradendiatiooms_w2 val. thewil=actualbe fedoutconcreteinto the GraphicValve which 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) 

with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) 

with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) values = {xX: np.random.ranan(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) with tf.Session as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_ val, grad_wl_val, grad_w2_val = out 

N, D, H = 64, 1000, 100 x = tf£.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) 

h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_pred - y loss = tf.reduce _mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) 

###### with tf.Session() as sess: 

values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} learning rate = le-5 

for t in range(50): out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) loss_val, grad_wl_val, grad_w2_val = out values[wl] -= learning rate * grad_wl_val values[w2] -= learning_rate * grad_w2_val 

N, D, H = 64, 1000, 100 

x = tf£.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.placeholder(tf.float32, shape=(D, H)) w2 = tf.placeholder(tf.float32, shape=(H, D)) 

h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_pred - y loss = tf.reduce _mean(tf.reduce sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) with tf.Session() as sess: values = {x: np.random.randn(N, D), wl: np.random.randn(D, H), w2: np.random.randn(H, D), y: np.random.randn(N, D),} learning rate = le-5 for t in range(50): out = sess.run([{loss, grad_wl, grad_w2], feed_dict=values) wut4oss_val, grad_wl_val, grad_w2_val = out ako <values[(wl] -= learning_rate * grad_wl_val~ aS values[w2] -= learning rate * grad_w2_val 

N, D, H = 64, 1000, 100 x = tfi.placeholder(tf.float32, shape=(N, D)) = Di ace Oe Oa aHe= \) wl = tf ° veriehtaiee . randomunapa deb , B))) posit inthe ong h = t£.maximum(tf.matmul(x, wl), 0) y_pred = tf.matmul(h, w2) diff = ypred - y loss = tf.reduce _mean(tf.reduce sum(diff ** 2\, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) learning rate = le-5 new_wl = wl.assign(wl -— learning rate * grad _wl) new_w2 = w2.assign(w2 —- learning rate * grad _w2) 

with tf.Session() as sess: sess.run(tf.global_variables initializer()) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} for t in range(50): loss_val, = sess.run([{loss], feed_dict=values) 

N, D, H = 64, 1000, 100 x = tfi.placeholder(tf.float32, shape=(N, D)) y = tfi.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

h = t£.maximum(tf.matmul(x, wl), 0) y_pred = tf.matmul(h, w2) diff = ypred - y loss = tf.reduce _mean(tf.reduce sum(diff ** 2\, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) learning rate = le-5 new_wl = wl.assign(wl -— learning rate * grad _wl) new_w2 = w2.assign(w2 —- learning rate * grad _w2) with tf.Session() as sess: sess.run(tf.global_variables initializer()) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} for t in range(50): loss_val, = sess.run([{loss], feed_dict=values) 

N, D, H = 64, 1000, 100 x = tfi.placeholder(tf.float32, shape=(N, D)) y = tfi.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

h = t£.maximum(tf.matmul(x, wl), 0) y_pred = tf.matmul(h, w2) diff = ypred - y loss = tf.reduce _mean(tf.reduce sum(diff ** 2\, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) learning rate = le-5 new_wl = wl.assign(wl -— learning rate * grad _wl) new_w2 = w2.assign(w2 —- learning rate * grad _w2) with tf.Session() as sess: values = {x: np.random.randn(N, D), : np.random.randn(N, D),} for t in range(50): 

with tf.Session() as sess: 

N, D, H = 64, 1000, 100 x = tfi.placeholder(tf.float32, shape=(N, D)) y = tfi.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

1e7 h = tf£.maximum(tf.matmul(x, wl), 0) _ y_pred = tf.matmul(h, w2) 53 diff = ypred - y = loss = tf.reduce _mean(tf.reduce sum(diff ** 2\, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) 2 awww new—W2 is not needed tp comPute 502 learning rate = le-5 the new_wl = wl.assign(wl - learning rate * grad _wl) : 0 10 20 30 40 50 - by A. with tf.Session() as sess: owrpaty So these lines ave SPPps sess.run(tf.global_variables initializer()) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} for t in range(50): loss_val, = sess.run([{loss], feed_dict=values) 

N, D, H = 64, 1000, 100 x = tf£.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

h = t£.maximum(tf.matmul(x, wl), 0) Sf. y_pred = tf.matmul(h, w2) diff = y_ pred - y 4 loss = tf.reduce mean(tf.reduce sum(diff ** 2, axis=1)) 3 grad_wl, grad_w2 = tf.gradients(loss, [wl, w2]) 21 ap learning_rate = le-5 ; new_wl = wl.assign(wl -— learning _rate * grad_wl) + new _w2 = w2.assign(w2 — learning rate * grad _w2) 0 10 20 30 40 50 with tf.Session() as sess: gragh face to von tase io les: sess.run(tf.global_variables _initializer() ) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} losses = [] for t in range(50): a loss val, _= sess.run({loss, <mark>[updates }}</mark> feed_dict=values) 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf£.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = ypred - y loss = tf.reduce mean(tf.reduce _sum(diff * diff, axis=1)) 

optimizer = tf.train.GradientDescentOptimizer(le-5) updates = optimizer.minimize(loss) 

with tf.Session() as sess: sess.run(tf.global_variables initializer() ) values = {x: np.random.randn(N, D), = [] “losses y: np.random.randn(N, D),} for t in range(50): loss val, _ = sess.run({loss,[ <mark>updates ]]</mark> , feed_dict=values) 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf.placeholder(tf.float32, shape=(N, D)) wl = tf.Variable(tf.random_normal((D, H))) w2 = tf.Variable(tf.random_normal((H, D))) 

h = tf.maximum(tf.matmul(x, wl), 0) y_ pred = tf.matmul(h, w2) 

optimizer = tf.train.GradientDescentOptimizer(le-3) updates = optimizer.minimize(loss) 

with tf.Session() as sess: sess.run(tf.globalvariables initializer()) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} for t in range(50): loss val, _ = sess.run([loss, updates], feeddict=values) 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf.placeholder(tf.float32, shape=(N, D)) 

h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=init) y_pred = tf.layers.dense(inputs=h, units=D, 

kernel_initializer=init) 

loss = tf.losses.mean_squared error(y pred, y) 

optimizer = tf.train.GradientDescentOptimizer(le0) updates = optimizer.minimize(loss) 

with tf.Session() as sess: sess.run(tf.globalvariables initializer() ) values = {x: np.random.randn(N, D), y: np.random.randn(N, D),} for t in range(50): loss val, _ = sess.run([loss, updates], | feed_dict=values) 

from keras.models import Sequential from keras.layers.core import Dense, Activation from keras.optimizers import SGD 

N, D, H = 64, 1000, 100 model = Sequential() model.add(Dense(inputdim=D, output _dim=H) ) model.add(Activation('relu') ) model.add(Dense(input_dim=H, output_dim=D)) 

optimizer = SGD(lr=1le0) model.compile(loss='meansquared error', optimizer=optimizer) 

xX = np.random.randn(N, D) y = np.random.randn(N, D) history = model.fit(x, y, nb_epoch=50, batch size=N, verbose=0) 

from keras.models import Sequential from keras.layers.core import Dense, Activation from keras.optimizers import SGD 

N, D, H = 64, 1000, 100 model = Sequential() ~~|~~ model.add(Dense(input_dim=D, output_dim=H)) model.add(Activation('relu') ) model.add(Dense(input_dim=H, output_dim=D)) optimizer = SGD(lr=1le0) model.compile(loss='meansquared error', optimizer=optimizer) xX = np.random.randn(N, D) y = np.random.randn(N, D) history = model.fit(x, y, nb_epoch=50, batch size=N, verbose=0) 

from keras.models import Sequential from keras.layers.core import Dense, Activation from keras.optimizers import SGD 

N, D, H = 64, 1000, 100 model = Sequential() model.add(Dense(inputdim=D, output _dim=H) ) model.add(Activation('relu') ) model.add(Dense(input_dim=H, output_dim=D)) ~~<mark>|</mark>~~ optimizer = SGD(lr=1e0) model.compile(loss= mean squared error’, optimizer=optimizer) 

xX = np.random.randn(N, D) y = np.random.randn(N, D) history = model.fit(x, y, nb_epoch=50, batch size=N, verbose=0) 

from keras.models import Sequential from keras.layers.core import Dense, Activation from keras.optimizers import SGD 

N, D, H = 64, 1000, 100 model = Sequential() model.add(Dense(inputdim=D, output _dim=H) ) model.add(Activation('relu') ) model.add(Dense(input_dim=H, output_dim=D)) 

optimize = D =le(0 ~~—_______»~~ | model.compile(loss='meansquared error’, optimizer=optimizer) xX = np.random.randn(N, D) y = np.random.randn(N, D) history = model.fit(x, y, nb_epoch=50, batch size=N, verbose=0) 

from keras.models import Sequential from keras.layers.core import Dense, Activation from keras.optimizers import SGD 

N, D, H = 64, 1000, 100 model = Sequential() model.add(Dense(inputdim=D, output _dim=H) ) model.add(Activation('relu') ) model.add(Dense(input_dim=H, output_dim=D)) optimizer = SGD(lr=1le0) model.compile(loss='meansquared error', optimizer=optimizer) 

xX = np.random.randn(N, y = np.random.randn(N, history = model.fit(x, 

D) D) y, nb_epoch=50, batch size=N, verbose=0) 

TensorFlow: Other High-Level Wrappers Keras (https://keras.io/) TFLearn (http://tflearn.org/) TensorLayer (http://tensorlayer.readthedocs.io/en/latest/) tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers) TF-Slim <u>(https://github.com/tensorflow/models/tree/master/inception/inception/slim)</u> tf.contrib.learn (https://www.tensorflow.org/get_started/tflearn) Pretty Tensor (https://github.com/google/prettytensor) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -6565 April 27, 2017 

TensorFlow: Other High-Level Wrappers Ships with TensorFlow Keras (https://keras.io/) TFLearn (http://tflearn.org/) TensorLayer (http://tensorlayer.readthedocs.io/en/latest/) tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers) TF-Slim <u>(https://github.com/tensorflow/models/tree/master/inception/inception/slim)</u> tf.contrib.learn (https://www.tensorflow.org/get_started/tflearn) Pretty Tensor (https://github.com/google/prettytensor) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -6666 April 27, 2017 

TensorFlow: Other High-Level Wrappers Ships with TensorFlow Keras (https://keras.io/) TFLearn (http://tflearn.org/) TensorLayer (http://tensorlayer.readthedocs.io/en/latest/) tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers) TF-Slim <u>(https://github.com/tensorflow/models/tree/master/inception/inception/slim)</u> tf.contrib.learn (https://www.tensorflow.org/get_started/tflearn) <mark>Pretty Tensor</mark> (https://github.com/google/prettytensor) 

From Google 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -6767 April 27, 2017 

TensorFlow: Other High-Level Wrappers Ships with TensorFlow Keras (https://keras.io/) TFLearn (http://tflearn.org/) TensorLayer (http://tensorlayer.readthedocs.io/en/latest/) tf.layers (https://www.tensorflow.org/api_docs/python/tf/layers) TF-Slim <u>(https://github.com/tensorflow/models/tree/master/inception/inception/slim)</u> tf.contrib.learn (https://www.tensorflow.org/get_started/tflearn) <mark>Pretty Tensor</mark> (https://github.com/google/prettytensor) <mark>Sonnet</mark> (https://github.com/deepmind/sonnet) 

From Google 

From DeepMind 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -6868 April 27, 2017 

### TensorFlow: Pretrained Models 

###### **Transfer  Learning** 

TF-Slim: (https://github.com/tensorflow/models/tree/master/slim/nets) - - Keras: (https://github.com/fchollet/deep <u>learning models)</u> 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 -6969 April 27, 2017 

import theano import theano.tensor as T 

# Batch size, input dim, hidden dim, num classes N, D, H, C = 64, 1000, 100, 10 

x = T.matrix('x') 

y = T.vector('y', dtype='int64') wl = T.matrix('wl1') w2 = T.matrix('w2') 

- # Forward pass: Compute scores a = x.dot(wl1) arelu = T.nnet.relu(a) scores = arelu.dot(w2) 

- # Forward pass: compute softmax loss probs = T.nnet.softmax(scores) loss = T.nnet.categorical crossentropy(probs, y).mean() 

- # Backward pass: compute gradients dwl, dw2 = T.grad(loss, [wl, w2]) 

f = theano. function ( inputs=[x, y, wl, w2], outputs=[loss, scores, dwl, dw2], ) 

import theano import theano.tensor as T 

# Batch size, input dim, hidden dim, num classes N, D, H, C = 64, 1000, 100, 10 

x = T.matrix('x') y = T.vector('y', dtype='int64') ~~—_———~~ | wl = T.matrix('wl1') w2 = T.matrix('w2') 

- # Forward pass: Compute scores a = x.dot(wl1) arelu = T.nnet.relu(a) scores = a_relu.dot(w2) 

- # Forward pass: compute softmax loss probs = T.nnet.softmax(scores) loss = T.nnet.categorical crossentropy(probs, y).mean() 

# Backward pass: compute gradients dwl, dw2 = T.grad(loss, [wl, w2]) 

f = theano. function( inputs=[x, y, wl, w2], outputs=[loss, scores, dwl, dw2], ) 

import theano import theano.tensor as T 

# Batch size, input dim, hidden dim, num classes N, D, H, C = 64, 1000, 100, 10 x = T.matrix('x') y = T.vector('y', dtype='int64') wl = T.matrix('wl1') w2 = T.matrix('w2') 

# Forward pass: Compute scores a = x.dot(wl1) arelu = T.nnet.relu(a) scores = a_relu.dot(w2) # Forward pass: compute softmax loss probs = T.nnet.softmax(scores) loss = T.nnet.categorical crossentropy(probs, y).mean() # Backward pass: compute gradients ~~—————~~ | dw1, dw2 = T.grad(loss, [wl, w2]) f = theano. function( inputs=[x, y, wl, w2], outputs=[loss, scores, dwl, dw2], ) 

import theano import theano.tensor as T 

# Batch size, input dim, hidden dim, num classes N, D, H, C = 64, 1000, 100, 10 

x = T.matrix('x') y = T.vector('y', dtype='int64') wl = T.matrix('wl1') w2 = T.matrix('w2') 

- # Forward pass: Compute scores a = x.dot(wl1) arelu = T.nnet.relu(a) scores = a_relu.dot(w2) 

# Forward pass: compute softmax loss probs = T.nnet.softmax(scores) loss = T.nnet.categorical crossentropy(probs, y).mean() # Backward pass: compute gradients dwl, dw2 = T.grad(loss, [wl, w2]) ; ~~SEES~~ f = theano. function( inputs=[x, y, wl, w2], outputs=[loss, scores, dwl, dw2], ) 

import theano import theano.tensor as T 

# Batch size, input dim, hidden dim, num classes N, D, H, C = 64, 1000, 100, 10 

# Run the function XX = np.random.randn(N, D) yy = np.random.randint(C, size=N) wwl1 = le-2 * np.random.randn(D, H) ww2 = le-2 * np.random.randn(H, C) fo eee eee loss, scores, dwwl, dww2 = f(xx, yy, wwl, ww2) print loss wwl -= learning rate * dwwl ww2_ -= learning rate * dww2 

= : wee = T.matrix(’x") oe : y = T.vector(‘y', dtype='int64' ) wl = T.matrix('wl1') w2 = T.matrix('w2') # Forward pass: Compute scores a = x.dot(w1) a_relu = T.nnet.relu(a) scores = a_relu.dot(w2) 

# Forward pass: compute softmax loss probs = T.nnet.softmax(scores) loss = T.nnet.categorical crossentropy(probs, y).mean() 

# Backward pass: compute gradients dwl, dw2 = T.grad(loss, [wl, w2]) f = theano. function( inputs=[x, y, wl, w2], outputs=[loss, scores, dwl, dw2], ) 

PyTorch (Facebook) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -7979 April 27, 2017 

### PyTorch: Three Levels of Abstraction 

**Tensor** : Imperative ndarray, but runs on GPU **Variable** : Node in a computational graph; stores data and gradient **Module** : A neural network layer; may store state or learnable weights 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 -8080 April 27, 2017 

### PyTorch: Three Levels of Abstraction 

##### **TensorFlow equivalent** 

**Tensor** : Imperative ndarray, but runs on GPU 

Numpy array 

**Variable** : Node in a computational graph; stores data and gradient **Module** : A neural network layer; may store state or learnable weights 

Tensor, Variable, Placeholder 

tf.layers, or TFSlim, or TFLearn, or Sonnet, or …. 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -8181 April 27, 2017 

import torch 

###### dtype = torch.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) 

learning rate = le-6 for t in range(500): 

h = x.mm(wl) 

h_relu = h.clamp(min=0) 

y_pred = h_relu.mm(w2) loss = (y_pred - y).pow(2).sum() 

grad_y pred = 2.0 * (y_pred =- y) grad_w2 = h_relu.t().mm(grad_ypred) grad_h relu = grad_y pred.mm(w2.t() ) grad_h = grad_h_ relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) 

wl -= learning_rate w2 -= learning_rate 

* grad _wl * grad_w2 

import torch 

###### dtype = torch.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) learning rate = le-6 for t in range(500): h = x.mm(wl) h_relu = h.clamp(min=0) y_pred = h_relu.mm(w2) loss = (y_pred - y).pow(2).sum() grad_y pred = 2.0 * (y_pred =- y) grad_w2 = h_relu.t().mm(grad_ypred) grad_h relu = grad_y pred.mm(w2.t() ) grad_h = grad_h_ relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) wl -= learning_rate * grad _wl w2 -= learning_rate * grad_w2 

import torch 

dtype = torch.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) learning rate = le-6 for t in range(500): h = x.mm(wl) h_relu = h.clamp(min=0) y_pred = h_relu.mm(w2) loss = pred - -pow(2).sum grad_y pred = 2.0 * (y_pred =- y) grad_w2 = h_relu.t().mm(grad_ypred) grad_h relu = grad_y pred.mm(w2.t() ) grad_h = grad_h_ relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) wl -= learning_rate * grad _wl w2 -= learning_rate * grad_w2 

import torch 

###### dtype = torch.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) learning rate = le-6 for t in range(500): h = x.mm(wl) h_relu = h.clamp(min=0) y_pred = h_relu.mm(w2) loss = (y_pred - y).pow(2).sum() grad_y pred = 2.0 * (y_pred =- y) grad_w2 = h_relu.t().mm(grad_ypred grad_h relu = grad_y pred.mm(w2.t() grad_h = grad_h_ relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) wl -= learning_rate * grad _wl w2 -= learning_rate * grad_w2 

import torch 

###### dtype = torch.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) learning rate = le-6 for t in range(500): h = x.mm(wl) h_relu = h.clamp(min=0) y_pred = h_relu.mm(w2) loss = (y_pred - y).pow(2).sum() 

grad_y pred = 2.0 * (y_pred =- y) grad_w2 = h_relu.t().mm(grad_ypred) grad_h relu = grad_y pred.mm(w2.t() ) grad_h = grad_h_ relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) wl -= learning_rate * grad _wl w2 -= learning_rate * grad_w2 

wl -= learning_rate w2 -= learning_rate 

import torch 

dtype = torch.cuda.FloatTensor 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in).type(dtype) y = torch.randn(N, D_out).type(dtype) wl = torch.randn(D_in, H).type(dtype) w2 = torch.randn(H, D_out).type(dtype) 

learning rate = le-6 for t in range(500): h = x.mm(wl) h_relu = h.clamp(min=0) y_pred = h_relu.mm(w2) loss = (y_pred - y).pow(2).sum() grad_y pred = 2.0 * (y_pred - y) grad_w2 = h_relu.t().mm(grad_ypred) grad_h relu = grad_y pred.mm(w2.t() ) grad_h = grad_h relu.clone() grad_h[h < 0] = 0 grad_wl = x.t().mm(grad_h) wl -= learning rate * grad wl w2 -= learning rate * grad_w2 

###### import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires _grad=False) y = Variable(torch.randn(N, D_out), requires _grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires_grad=True) 

learning _rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() 

if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() 

wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

###### import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires _grad=False) y = Variable(torch.randn(N, D_out), requires _grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires_grad=True) 

learning _rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() 

if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() 

wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

###### import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D in), ]|requires grad=False) y = Variable(torch.randn(N, D_out)]| requires grad=False wl = Variable(torch.randn(D_in, H)] requires grad=True) w2 = Variable(torch.randn{#,D_out], requires grad=True) learning _rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires _grad=False) y = Variable(torch.randn(N, D_out), requires _grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires_grad=True) 

learning _rate = le-6 for t in range(500): 

y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() 

if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() 

wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires _grad=False) y = Variable(torch.randn(N, D_out), requires _grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires_grad=True) learning _rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires _grad=False) y = Variable(torch.randn(N, D_out), requires _grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires_grad=True) 

learning _rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() if wl.grad: wl.grad.data.zero() if w2.grad: w2.grad.data.zero() loss.backward() 

wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data 

class ReLU(torch.autograd.Function): def forward(self, x): self.sfor _ b **a** ckward(x)ve_ return x.clamp(min=0) def backward(self, grad y): x, = self.saved_ tensors grad input = grad y.clone() grad_input[x < 0] = 0 return grad input 

import torch 

from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires_grad=False) model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size average=False) 

learning rate = le-4 for t in range(500): y_pred = model(x) loss = loss fn(y_pred, y) model.zero_grad{() loss.backward() 

for param in model.parameters(): param.data -= learning rate 

* param.grad.data 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires_grad=False) 

model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D out)) <mark>loss fn = torch.nn.MSELoss(size_ average=False)</mark> learning rate = le-4 for t in range(500): y_pred = model(x) loss = loss fn(y_pred, y) model.zero_grad{() loss.backward() 

for param in model.parameters(): param.data -= learning rate 

* param.grad.data 

import torch 

from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires_grad=False) model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size average=False) learning rate = le-4 e in ange () ( = y_pred = model(x) loss = loss fn(y_pred, y) model.zero_grad{() loss.backward() 

for param in model.parameters(): param.data -= learning rate 

* param.grad.data 

import torch 

from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires_grad=False) model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size average=False) learning rate = le-4 for t in range(500): y_pred = model(x) loss = loss fn(y_pred, y) 

model.zero_grad{() loss.backward() 

for param in model.parameters(): param.data -= learning rate 

* param.grad.data 

import torch 

from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires_grad=False) model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size average=False) 

learning rate = le-4 for t in range(500): y_pred = model(x) loss = loss fn(y_pred, y) model.zero_grad{() loss.backward() 

for param in model.parameters(): param.data -= learning rate 

* param.grad.data 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires _grad=False) model = torch.nn.Sequential( torch.nn.Linear(Din, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size_average=False) 

optimizer = torch.optim.Adam(model.parameters(), 

lr=learning_rate) 

e a Ce UU e y_pred = model(x) loss = loss fn(y_pred, y) 

optimizer.zero grad() loss .backward( ) 

optimizer.step() 

import torch 

from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires _grad=False) 

###### model = torch.nn.Sequential( 

torch.nn.Linear(Din, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out)) loss_fn = torch.nn.MSELoss(size_average=False) 

learning_rate = le-4 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

for t in range(500): y_pred = model(x) loss = loss fn(y_pred, y) 

optimizer.zero grad() loss .backward( ) 

optimizer.step() 

import torch from torch.autograd import Variable 

class TwoLayerNet(torch.nn.Module): 

def init (self, D_in, H, D_out): super(TwoLayerNet, self). init () self.linearl = torch.nn.Linear(Din, H) self.linear2 = torch.nn.Linear(H, D_out) 

def forward(self, x): h_relu = self.linearl(x).clamp(min=0) y_pred = self.linear2(h_relu) return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires grad=False) 

model = TwoLayerNet(Din, H, D_out) 

criterion = torch.nn.MSELoss(size average=False) optimizer = torch.optim.SGD(model.parameters(), for t in range(500): 

lr=le-4) 

y_pred = model(x) loss = criterion(y pred, y) 

optimizer.zero_grad() loss .backward() optimizer.step() 

import torch from torch.autograd import Variable 

class TwoLayerNet(torch.nn.Module): def init (self, D_in, H, D_out): super(TwoLayerNet, self). init () self.linearl = torch.nn.Linear(Din, H) self.linear2 = torch.nn.Linear(H, D_out) def forward(self, x): h_relu = self.linearl(x).clamp(min=0) y_pred = self.linear2(h_relu) return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires grad=False) model = TwoLayerNet(Din, H, D_out) criterion = torch.nn.MSELoss(size average=False) optimizer = torch.optim.SGD(model.parameters(), lr=le-4) for t in range(500): 

lr=le-4) 

y_pred = model(x) loss = criterion(y pred, y) 

optimizer.zero_grad() loss .backward() optimizer.step() 

import torch from torch.autograd import Variable 

class TwoLayerNet(torch.nn.Module): def init (self, D_in, H, D_out): super(TwoLayerNet, self). init () self.linearl = torch.nn.Linear(Din, H) self.linear2 = torch.nn.Linear(H, D_out) def forward(self, x): h_relu = self.linearl(x).clamp(min=0) y_pred = self.linear2(h_relu) return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires grad=False) model = TwoLayerNet(Din, H, D_out) criterion = torch.nn.MSELoss(size average=False) optimizer = torch.optim.SGD(model.parameters(), lr=le-4) for t in range(500): 

lr=le-4) 

y_pred = model(x) loss = criterion(y pred, y) 

optimizer.zero_grad() loss .backward() optimizer.step() 

import torch from torch.autograd import Variable 

class TwoLayerNet(torch.nn.Module): def init (self, D_in, H, D_out): super(TwoLayerNet, self). init () self.linearl = torch.nn.Linear(Din, H) self.linear2 = torch.nn.Linear(H, D_out) 

def forward(self, x): h_relu = self.linearl(x).clamp(min=0) y_pred = self.linear2(h_relu) return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = Variable(torch.randn(N, D_in)) y = Variable(torch.randn(N, D_out), requires grad=False) model = TwoLayerNet(Din, H, D_out) criterion = torch.nn.MSELoss(size average=False) optimizer = torch.optim.SGD(model.parameters(), lr=le-4) for t in range(500): y_pred = model(x) loss = criterion(y pred, y) 

lr=le-4) 

optimizer.zero_grad() loss .backward() optimizer.step() 

import torch from torch.autograd import Variable 

###### class TwoLayerNet(torch.nn.Module): 

- def init (self, D_in, H, D_out): super(TwoLayerNet, self). init () self.linearl = torch.nn.Linear(Din, H) self.linear2 = torch.nn.Linear(H, D_out) 

def forward(self, x): h_relu = self.linearl(x).clamp(min=0) y_pred = self.linear2(h_relu) return ypred 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = Variable(torch.randn(N, y = Variable(torch.randn(N, 

D_in)) D_out), requires grad=False) 

model = TwoLayerNet(Din, H, D_out) 

criterion = torch.nn.MSELoss(size average=False) optimizer = torch.optim.SGD(model.parameters(), for t in range(500): y_pred = model(x) loss = criterion(y pred, y) 

lr=le-4) 

optimizer.zero_grad() loss .backward() optimizer.step 

import torch from torch.autograd import Variable from torch.utils.data import TensorDataset, DataLoader 

N, D_in, H, D_out = 64, 1000, 100, 10 x = torch.randn(N, D_in) y = torch.randn(N, D_out) 

loader = DataLoader(TensorDataset(x, 

y), batch _size=8) 

model = TwoLayerNet(D_in, H, D_out) 

criterion = torch.nn.MSELoss(size_average=False) optimizer = torch.optim.SGD(model.parameters(), lr=le-4) for epoch in range(10): 

for x_batch, ybatch in loader: X_ var, y_var = Variable(x), Variable(y) y_pred = model(x_var) loss = criterion(y pred, yvar) 

optimizer.zero_ grad() loss. backward( ) optimizer.step() 

import torch from torch.autograd import Variable from torch.utils.data import TensorDataset, DataLoader 

N, D_in, H, D_out = 64, 1000, 100, 10 

x = torch.randn(N, D_in) y = torch.randn(N, D_out) 

loader = DataLoader(TensorDataset(x, y), batch _size=8) 

model = TwoLayerNet(D_in, H, D_out) 

criterion = torch.nn.MSELoss(size_average=False) optimizer = torch.optim.SGD(model.parameters(), for epoch in range(10): 

for x_batch, ybatch in loader: Xvar, y_var = Variable(x), Variable(y) y_pred = model(x_var) loss = criterion(y pred, yvar) 

optimizer.zero_ grad() loss. backward( ) optimizer.step() 

lr=le-4) 

import torch import torchvision alexnet = torchvision.models.alexnet(pretrained=True) vggl6 = torchvision.models.vgg16(pretrained=True) resnetl101 = torchvision.models.resnet101(pretrained=True) 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() 

local function f(w) assert(w == weights) local scores = model: forward(x) local loss = loss_fn:forward(scores, y) 

grad_weights:zero() 

local grad_scores = loss_fn:backward(scores, y) local grad_x = model:backward(x, grad_scores) 

return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() local function f(w) assert(w == weights) local scores = model: forward(x) local loss = loss_fn:forward(scores, y) grad_weights:zero() local grad_scores = loss_fn:backward(scores, y) local grad_x = model:backward(x, grad_scores) return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() local function f(w) assert(w == weights) local scores = model: forward(x) local loss = loss_fn:forward(scores, y) grad_weights:zero() local grad_scores = loss_fn:backward(scorgs, y) local grad_x = model:backward(x, grad_scofes) return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() local function f(w) assert(w == weights local scores = model: forward(x) local loss = loss_fn:forward(scores, y) 

local loss = loss_fn:forward(scores, y) grad_weights:zero() local grad_scores = loss_fn:backward(scores, y) local grad_x = model:backward(x, grad_scores) return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() local function f(w) assert(w == weights) local scores = model: forward(x) local loss = loss_fn:forward(scores, y) grad_weights:zero() local grad_scores = loss_fn:backward(scores, y) local grad_x = model:backward(x, grad_scores) return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

require 'torch' require 'nn' require ‘optim' local N, D, H, C = 64, 256, 512, 10 local model = nn.Sequential() model:add(nn.Linear(D, H)) model: add(nn.ReLU() ) model:add(nn.Linear(H, C)) local loss_fn = nn.CrossEntropyCriterion() local x = torch.randn(N, D) local y = torch.Tensor(N): random(C) local weights, grad_weights = model:getParameters() local function f(w) assert(w == weights) local scores = model: forward(x) local loss = loss_fn:forward(scores, y) 

grad_weights:zero() local grad_scores = loss_fn:backward(scores, y) local grad_x = model:backward(x, grad_scores) 

return loss, grad_weights end local state = {learningRate=1e-3} for t = 1, 100 do optim.adam(f, weights, state) end 

### Torch vs PyTorch 

**Torch PyTorch** (-) Lua (+) Python (-) No autograd (+) Autograd (+) More stable (-) Newer, still changing (+) Lots of existing code (-) Less existing code (0) Fast (0) Fast 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 118 

April 27, 2017 

### Torch vs PyTorch 

**Torch PyTorch** (-) Lua (+) Python (-) No autograd (+) Autograd (+) More stable (-) Newer, still changing (+) Lots of existing code (-) Less existing code (0) Fast (0) Fast Conclusion: Probably use PyTorch for new projects 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 119 April 27, 2017 

N, D, H = 64, 1000, 100 x = tf.placeholder(tf.float32, shape=(N, D)) y = tf.placeholder(tf.float32, shape=(N, D)) wl = tf£.Variable(tf.random_normal((D, H))) w2 = tf£.Variable(tf.random_normal((H, D))) 

h = tf£.maximum(tf.matmul(x, wl), 0) y_pred = tfi.matmul(h, w2) diff = y_ pred - y loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) grad_wl, grad_w2 = tf.gradients(loss, [wl, w2}) learning_rate. new_wl = wl.assign(wl= le-5 — learning; rate * grad_wl) new_w2 = w2.assign(w2 - learning rate * grad_w2) updates = tf.group(new_wl, new_w2) with tf.Session(): as sess: sess.run(tf.global_variables_initializer()) warns = 5h Pee eee ee y: np.random.randn(N, D),} losses = [] for t in range(50): loss_val, _ = sess.run([loss, updates], feed_dict=values) 

import torch from torch.autograd import Variable 

N, D_in, H, D_out = 64, 1000, 100, 10 x = Variable(torch.randn(N, D_in), requires_grad=False) y = Variable(torch.randn(N, D_out), requires grad=False) wl = Variable(torch.randn(D_in, H), requires_grad=True) w2 = Variable(torch.randn(H, D_out), requires grad=True) 

learning rate = le-6 for t in range(500): y_pred = x.mm(wl).clamp(min=0).mm(w2) loss = (y_pred - y).pow(2).sum() if wl.grad: wl.grad.data.zero () if w2.grad: w2.grad.data.zero() loss.backward() wl.data -= learning rate * wl.grad.data w2.data -= learning rate * w2.grad.data _ 

### <u>Static</u> vs Dynamic: Optimization 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 121 April 27, 2017 

### <u>Static</u> vs Dynamic: Serialization 

#### **<mark>Static</mark>** 

#### **<mark>Dynamic</mark>** 

Once graph is built, can **serialize** it and run it without the code that built the graph! 

Graph building and execution are intertwined, so always need to keep code around 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 122 

April 27, 2017 

### Static vs Dynamic: Conditional 

w1 * x   if z > 0 y = w2 * x   otherwise 

12 Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 123 April 27, 2017 3 

N, D, H = 3, 4, 5 x = Variable(torch.randn(N, D)) wl = Variable(torch.randn(D, H)) w2 = Variable(torch.randn(D, H)) 

z= 10 if 2: > Os y = x.mm(wl) else: 

y = x.mm(w2) 

### Static vs Dynamic: LoopsDynamic: Loops: Loops Loops 

12 Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 126 April 27, 2017 6 

### Dynamic Graphs in TensorFlow 

TensorFlow Fold make dynamic graphs easier in TensorFlow through **dynamic batching** 

Looks et al, “Deep Learning with Dynamic Computation Graphs”, ICLR 2017 <u>https://github.com/tensorflow/fold</u> 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 129 April 27, 2017 

### Dynamic Graph Applications 

- Recurrent networks - Recursive networks 

The cat ate a big rat 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 131 April 27, 2017 

### Dynamic Graph Applications 

- Recurrent networks - Recursive networks - Modular Networks 

_What color is the cat?_ 

Andreas et al, “Neural Module Networks”, CVPR 2016 Andreas et al, “Learning to Compose Neural Networks for Question Answering”, NAACL 2016 

<u>This image is in the public domain</u> 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 132 April 27, 2017 

_no_ 

### Dynamic Graph Applications 

compare 

- Recurrent networks - Recursive networks - Modular Networks 

_Are there more cats than dogs?_ 

Andreas et al, “Neural Module Networks”, CVPR 2016 Andreas et al, “Learning to Compose Neural Networks for Question Answering”, NAACL 2016 

<u>This image is in the public domain</u> 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 133 April 27, 2017 

### Dynamic Graph Applications 

- Recurrent networks 

- Recursive networks 

- Modular Networks 

- (Your creative idea here) 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 134 April 27, 2017 

Caffe (UC Berkeley) 

13 Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 135 April 27, 2017 5 

### Caffe Overview 

- ●Core written in C++ 

- ●Has Python and MATLAB bindings 

- ●Good for training or finetuning feedforward classification models 

- ●Often no need to write code! 

- ●Not used as much in research anymore, still popular for deploying models 

13 Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 136 April 27, 2017 6 

### Caffe: Training / Finetuning 

#### No need to write code! 

1. Convert data (run a script) 

2. Define net (edit prototxt) 3. Define solver (edit prototxt) 4. Train (with pretrained weights) (run a script) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 137 **April 27, 2017** 

### Caffe step 1: Convert Data 

- ●DataLayer reading from LMDB is the easiest ●Create LMDB using convert_imageset ●Need text file where each line is ○“[path/to/image.jpeg] [label]” 

- ●Create HDF5 file yourself using h5py 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -** 138 

**April 27, 2017** 

### Caffe step 1: Convert Data 

- ImageDataLayer: Read from image files 

- ● WindowDataLayer: For detection 

- ● HDF5Layer: Read from HDF5 file 

- ● From memory, using Python interface 

- ● All of these are harder to use (except Python) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 139 

**April 27, 2017** 

|||layer|{<br>bottom:<br>"res5Sc"<br>top:<br>"pool5"<br><br>|
|---|---|---|---|
|1<br>name: "ResNet-152"<br>|||name:<br>"pool5"<br><br>|
|pune alae<br>input_dim:<br>1|||type:<br>"Pooling"<br>ype:<br>g|
|input_dim:<br>3|||pooling_param<br>{|
|input_dim: 224<br>|||kernel_size:<br>7<br>|
|input_dim: 224|||eeatioe a<br>stride:|
|layer {|||pool:<br>AVE|
|bottom: "data"<br>top:<br>"convi"<br>name:<br>"convi"||3|te|
|type:<br>"Convolution"||||
|convolution_param {<br>||layer|{<br><br>|
|num_output: 64<br>kernel_size:<br>7|||bottom:<br>"pool5"<br>=<br>Pp|
|pad:<br>3|||top:<br>"fc1000"|
|strides 2<br>pias_term:<br>false|||name: "fc1000"|
|}|||type:<br>"InnerProduct"<br>:|
|}|||inner_product_param<br>{|
|layer<br>{|6765||num_output:1000|
|bottom:<br>"convi"|||<br>i|
|top:<br>"convi"||}||
|name:<br>"bn_convi"||||
|type:<br>"BatchNorm"||||
|batch_norm_param {||layer|{|
|use_global_stats:<br>true|||bottom:<br>"fci1000"|
|i|||top:<br>"prob"|
|‘|||name:<br>"prob"|
|||}|type:<br>"Softmax"|

net: "“models/bvlc_alexnet/train_val.prototxt" test_iter: 1000 test_interval: 1000 base_lr: 0.01 1r_policy: "step" gamma: 0.1 stepsize: 100000 display: 20 max_iter: 450000 momentum: 0.9 weight_decay: 0.0005 snapshot: 10000 snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train" solver_mode: GPU 

### Caffe step 4: Train! 

**./build/tools/caffe train \ -gpu 0 \ -model path/to/trainval.prototxt \ -solver path/to/solver.prototxt \ -weights path/to/pretrained_weights.caffemodel** 

<u>https://github.com/BVLC/caffe/blob/master/tools/caffe.cpp</u> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 143 **April 27, 2017** 

### Caffe step 4: Train! 

**./build/tools/caffe train \** **<mark>-gpu 0 \</mark> -model path/to/trainval.prototxt \ -solver path/to/solver.prototxt \ -weights path/to/pretrained_weights.caffemodel** 

- **-gpu -1** for CPU-only **-gpu all** for multi-gpu 

<u>https://github.com/BVLC/caffe/blob/master/tools/caffe.cpp</u> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 144 **April 27, 2017** 

### Caffe Python Interface 

- Not much documentation… Read the code! Two most important files: ● <u>: caffe/python/caffe/_caffe.cpp</u> ○Exports Blob, Layer, Net, and Solver classes 

- ● <u>caffe/python/caffe/pycaffe.py</u> 

   - ○Adds extra methods to Net class 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -** 146 **April 27, 2017** 

### Caffe Python Interface 

##### Good for: 

- ●Interfacing with numpy 

- ●Extract features: Run net forward 

- ●Compute gradients: Run net backward (DeepDream, etc) ●Define layers in Python with numpy (CPU only) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 8 -** 147 

**April 27, 2017** 

### Caffe Pros / Cons 

- ●(+) Good for feedforward networks 

- ●(+) Good for finetuning existing networks ●(+) Train models without writing any code! ●(+) Python interface is pretty useful! ●(+) Can deploy without Python 

- ●(-) Need to write C++ / CUDA for new GPU layers ●(-) Not good for recurrent networks 

- ●(-) Cumbersome for big networks (GoogLeNet, ResNet) 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 -** 148 **April 27, 2017** 

Caffe2 (Facebook) 

14 Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 149 April 27, 2017 9 

### Caffe2 Overview 

- ●Very new - released a week ago =) 

- ●Static graphs, somewhat similar to TensorFlow 

- ●Core written in C++ 

- ●Nice Python interface 

- ●Can train model in Python, then serialize and deploy without Python 

- ●Works on iOS / Android, etc 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 150 April 27, 2017 

### : **Facebook** : **Google** TensorFlow PyTorch +Caffe2 

_“One framework to rule them all”_ 

Research Production 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 151 April 27, 2017 

### My Advice: 

**TensorFlow** is a safe bet for most projects. Not perfect but has huge community, wide usage. Maybe pair with high-level wrapper (Keras, Sonnet, etc) 

I think **PyTorch** is best for research. However still new, there can be rough patches. 

Use **TensorFlow** for one graph over many machines Consider **Caffe** , **Caffe2,** or **TensorFlow** for production deployment Consider **TensorFlow** or **Caffe2** for mobile 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 8 - 152 April 27, 2017 

### Next Time: CNN Architecture Case Studies 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 153 April 27, 2017 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 154 April 27, 2017 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 8 - 155 April 27, 2017

---

## 源文件

- [cs231n_2017_lecture8.pdf](attachments/documents/AI_CNN-7a2d1dec3d03/cs231n_2017_lecture8.pdf)
