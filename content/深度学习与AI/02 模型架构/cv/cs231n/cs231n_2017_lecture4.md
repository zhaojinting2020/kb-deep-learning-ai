---
title: cs231n_2017_lecture4
source: converted:attachments/documents/AI_CNN-cc3dff81b4c4/cs231n_2017_lecture4.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-cc3dff81b4c4/cs231n_2017_lecture4.pdf
  title: cs231n_2017_lecture4.pdf
---

## Lecture 4: Backpropagation and Neural Networks 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 4 -** 1 

**April 13, 2017** 

##### Administrative 

**Assignment 1** due **Thursday April 20** , 11:59pm on Canvas 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 2 April 13, 2017 

##### Administrative 

**Project:** TA specialities and some project ideas are posted on Piazza 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 3 

April 11, 2017 

##### Administrative 

**Google Cloud:** All registered students will receive an email this week with instructions on how to redeem $100 in credits 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 4 

April 13, 2017 

<mark>s= f(z;W)=We Li = do</mark> j4y, <mark>max(0,</mark> 8; <mark>—</mark> Sy, <mark>+</mark> 1) <mark>L=</mark> 5 <mark>ialit</mark> N dD, <mark>We</mark> 

~~<mark>Se</mark>~~ 

df(z) _, f(e+h)— fle) dx h 0 h 

# ™, | ~~<mark>) Sta</mark>~~ n ~~e~~ 4 ~~<mark>5</mark>~~ 

1 

1 

##### Gradients add at branches 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 51 

April 13, 2017 

##### Vectorized operations 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 53 April 13, 2017 

##### Vectorized operations 

4096-d input vector Q: what is the size of the Jacobian matrix? [4096 x 4096!] 

f(x) = max(0,x) _(elementwise)_ 

4096-d output vector in practice we process an entire minibatch (e.g. 100) of examples at one time: i.e. Jacobian would technically be a [409,600 x 409,600] matrix :\ 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 4 - 

April 13, 2017 

<mark>f(2,W) = ||W-a||° = 0(W</mark> - <mark>2);</mark> 

###### <mark>A vectorized example:</mark> 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 59 

April 13, 2017 

<mark>f(x,W) = ||W- ||? = V1(W V1(W(W</mark> - <mark>2); oo</mark> 

2 

__ 2 __ 2 | <mark>-0.3</mark> 0.1 0.5 <mark>08 l ose a7</mark> oa0.26 0.116 | **0.** 24 | 0.44 1.00 Wi; <mark>3</mark> W121 ++: + Wi ntn Wi ntn ntn <mark>_Of</mark> — <mark>=\- Of</mark> OGK | q=W-2= OW; j k Oqrn OWi,; k 

__ 2 <mark>_ n</mark> 2 <mark>T Vwi =2q:2z</mark> | 0.116 1.00 Wi; <mark>3 _Of</mark> — <mark>=\- Of</mark> OGK | OW; j k Oqrn OWi,; k 

> | —0.30.10.1 **0.** 588 | | <mark>0.088</mark> 0.176 <mark>|</mark> " 099 0.104 0.208 | 0.26 | | **0.** 24 | 0.44 W121 ++: + Wi ntn Wi ntn ntn q=W-r= 

__ 2 <mark>_ n</mark> 2 <mark>T Vwi =2q:2z</mark> | 0.116 1.00 Wi; <mark>3 _Of</mark> — <mark>=\- Of</mark> OGK | OW; j k Oqrn OWi,; k 

> | —0.30.10.1 **0.** 588 | | <mark>0.088</mark> 0.176 <mark>|</mark> " 099 0.104 0.208 | 0.26 | | **0.** 24 | 0.44 W121 ++: + Wi ntn Wi ntn ntn q=W-r= 

<mark>—</mark> 2 4) 9 <mark>0.1 f(z,W)= ||W - all’ = 4,(W - 2);</mark> | 0.088| <mark>0.3</mark> 0.1760.5 <mark>0.8 l</mark> | 9.29 0.104 0.208 | 0.26 | 0.116 | **0.** 24 | 0.44 1.00 X 0.52 Odk W121 <mark>a =</mark> Wii ++: + Wi ntn + Wi ntn Wi ntn ntn <mark>on</mark> of a —-W-r=.. : dk , Of _ Of Ode Writit-:: + Wantn Wantn Ox; <mark>d</mark> k Ogr OX; <mark>f(q) =|lall? =ap+--- +4</mark> = 2244 Wies 

<mark>_ 2</mark> yon 2 0.3 0.8 | 0.1 0.5 | <mark>_ r</mark> 0.088 0.176 1 0.99 <mark>Vil =2W° -q</mark> 0.104 0.208 | 0.26 | » 116 | 0.2 <mark>0.4</mark> | | 0.44 | 1.00 0.112 ] * 0.52 0.636 Odk _ | Wy — Wri —-W-r=** 1t1 +++: + Wintn +++: + Wintn + Wintn Wintn <mark>on afd</mark> dk q Writ rr: rr:: Wantn SLOn; LQ <mark>s</mark> k OgnSS Ox;Mak <mark>f(g) =llall? =a +--+</mark> = <mark>2,246Wr</mark> 

<mark>_ 2</mark> yon 2 0.3 0.8 | 0.1 0.5 | <mark>_ r</mark> 0.088 0.176 1 0.99 <mark>Vil =2W° -q</mark> 0.104 0.208 | 0.26 | » 116 | 0.2 <mark>0.4</mark> | | 0.44 | 1.00 0.112 ] * 0.52 0.636 Odk _ | Wy — Wri —-W-r=** 1t1 +++: + Wintn +++: + Wintn + Wintn Wintn <mark>on afd</mark> dk q Writ rr: rr:: Wantn SLOn; LQ <mark>s</mark> k OgnSS Ox;Mak <mark>f(g) =llall? =a +--+</mark> = <mark>2,246Wr</mark> 

#include <cmath> 

#include <vector> 

#include "caffe/layers/sigmoid_layer.hpp" 

namespace caffe { template <typename Dtype> inline Dtype sigmoid(Dtype x) { return 1. / (1. + exp(-x)); } template <typename Dtype> void SigmoidLayer<Dtype>: :Forward_cpu(const vector<Blob<Dtype>*>& bottom, constconstDtype*vector<Blob<Dtype>*>&bottom_data = bottom[@]->cpu_data();top) { 1 Dtype* top_data = top[0]->mutable_cpu_data(); const int count = bottom[®]->count(); oO PA me for (int i = 0; i < count; ++i) { —ZT top_data[i] = sigmoid(bottom_data[i]); e } } template <typename Dtype> void SigmoidLayer<Dtype>: :Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { if (propagate_down[0]) { const Dtype* top_data = top[®]->cpu_data(); const Dtype* top_diff = top[®]->cpu_diff(); Dtype* bottom_diff = bottom[9]->mutable_cpu_diff(); const int count = bottom[@]->count(); forconst(intconst(int(int Dtypeii = 0;sigmoid_xisigmoid_xii < count;== top_data[i];++i)++i) { CO bi v4 oO ae bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x); t } } #ifdef CPU_ONLY STUB_GPU(SigmoidLayer); #endif INSTANTIATE_CLASS(SigmoidLayer) ; } 

##### Summary so far... 

- neural nets will be very large: impractical to write down gradient formula by hand for all parameters 

- **backpropagation** = recursive application of the chain rule along a computational graph to compute the gradients of all inputs/parameters/intermediates 

- implementations maintain a graph structure, where the nodes implement the **forward** () / **backward** () API 

- **forward** : compute result of an operation and save any intermediates needed for gradient computation in memory 

- **backward** : apply the chain rule to compute the gradient of the loss function with respect to the inputs 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 81 April 13, 2017 

### Next: Neural Networks 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 82 

April 13, 2017 

<mark>ji =We</mark> 

ji <mark>=We f = = W2 max(0, Wiz) max(0, Wiz) Wiz)</mark> 

ji <mark>=We f = = W2 max(0, Wiz) max(0, Wiz) Wiz)</mark> 

<mark>j=Ws f = = W2 max(0, Wiz) max(0, Wiz) Wiz)</mark> 

<mark>j=We f =W?2 max(0, Wa) max(0, Wa) Wa)</mark> 

<mark>f = W3 max(0, Wo max(0, Wiz))</mark> 

- import numpy as np from numpy.random import randn 

7 

- N, D_in, H, D_out = 64, 1000, 100, 10 x, y = randn(N, D_in), randn(N, D_out) wl, w2 = randn(D_in, H), randn(H, D_out) for t in range(2000): h=1/ (1 + np.exp(-x.dot(w1))) y_pred = h.dot(w2) loss = np.square(y_pred - y).sum() print(t, loss) grad_y_pred = 2.9 * (y_pred - y) grad_w2 = h.T.dot(grad_y_pred) grad_h = grad_y_pred.dot(w2.T) grad_wl = x.T.dot(grad_h * h * (1 - h)) 

wl -= 1e-4 * grad_wi w2 -= 1e-4 * grad_w2 

# receive W1,W2,b1,b2 (weights/biases), X (data) # forward pass: hl = #... function of X,W1,b1 scores = #... function of h1,W2,b2 loss = #... (several lines of code to evaluate Softmax loss) # backward pass: dscores = #... dh1,dwW2,db2 = #... dwi,db1 = #... 

<u>This image</u> by Fotis Bobolas is licensed under CC-BY 2.0 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 90 

April 13, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 91 April 13, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 92 

April 13, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - **94** 

April 13, 2017 

###### Be very careful with your brain analogies! 

###### **Biological Neurons:** 

- Many different types 

- Dendrites can perform complex non-linear computations 

- Synapses are not a single weight but a complex non-linear dynamical system 

- Rate code may not be adequate 

[Dendritic Computation. London and Hausser] 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 4 - 95 April 13, 2017 

<mark>o(0) max(0.lx, x = pha</mark> | <mark>Ole)</mark> /<sup><mark>bi, w4x+be)</mark></sup> <mark>tanh(x) Jf. max(wi</mark><sup><mark>x +</mark></sup> <mark>max(0,</mark> >0 <mark>x) _/ Se 1) ; <0 Sf</mark> 

class Neuron: 

def neuron _tick(inputs): """"assume inputs and weights are 1-D numpy arrays and bias is a number """ cell body sum = np.sum(inputs * self.weights) + self.bias firing rate = 1.0 / (1.0 + math.exp(-cell_bodysum)) # sigmoid activation function return firing rate 

f = Ceibde. x: 1.0/(1.6 Sea ie # act tion funct +: gmol X = np.random.randn(3, 1) # random input vectc f three numbers (3x1) hl = f(np.dot(Wl, x) + bl) # calculate first hidden layer activations (4x1 h2 = f(np.dot(W2, hl) + b2) # calculate second hidden layer activation 1x1) out = np.dot(W3, h2) + b3 # output neuron (1x!] 

#### **Summary** 

- We arrange neurons into fully-connected layers 

- The abstraction of a **layer** has the nice property that it allows us to use efficient vectorized code (e.g. matrix multiplies) 

- Neural networks are not really _neural_ 

- Next time: Convolutional Neural Networks 

Fei-Fei Li & Justin Johnson & Serena Yeung 

10 Lecture 4 - 0 

April 13, 2017

---

## 源文件

- [cs231n_2017_lecture4.pdf](attachments/documents/AI_CNN-cc3dff81b4c4/cs231n_2017_lecture4.pdf)
