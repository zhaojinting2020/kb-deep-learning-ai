---
title: cs231n_2017_lecture5
source: converted:attachments/documents/AI_CNN-7f417cdfbdc0/cs231n_2017_lecture5.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-7f417cdfbdc0/cs231n_2017_lecture5.pdf
  title: cs231n_2017_lecture5.pdf
---

## Lecture 5: Convolutional Neural Networks 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 5 - 1 

April 18, 2017 

##### Administrative 

**Assignment 1** due **Thursday April 20** , 11:59pm on Canvas **Assignment 2** will be released Thursday 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 5 - 2 

April 18, 2017 

<mark>j=We f = = W2 max(0, Wiz) max(0, Wiz) Wiz)</mark> 

<mark>“TLL</mark> 

Simple cells: Response to light orientation 

Retinal ganglion cell LGN and and V1 Complex cells: receptive fields fields simple cells Response to light © C) oon orientation and movement CD ©) Hypercomplex cells: . . response to movement PAS @ © with an end point Vo Wie 

No response Response (end point) 

#### A bit of history: 

#### **Neocognitron** _[Fukushima 1980]_ 

<mark>“sandwich” architecture (SCSCSC…)</mark> simple cells: modifiable parameters complex cells: perform pooling 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 5 - 1313 April 18, 2017 

Benign Benign ee Malignant Benign : Fs ie ;me ;meme | “0.08 Bae f 3 

~~<u>i</u>~~ 

## Convolutional Neural Networks 

(First without the brain stuff) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 25 

**April 18, 2017** 

~~s~~ ~~<mark>s ST</mark> S~~ ~~<mark>i</mark>~~ <mark>SSS</mark> 

~~S~~ ~~<mark>ST Se</mark>~~ <mark>SSS</mark> / 

### <mark>Convolution Layer</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 28 

**April 18, 2017** 

### <mark>Convolution Layer</mark> 

32x32x3 image 

<mark>5x5x3 filter</mark> 

**Convolve** the filter with the image i.e. “slide over the image spatially, computing dot products” 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 29 

**April 18, 2017** 

### <mark>Convolution Layer</mark> 

Filters always extend the full depth of the input volume 

32x32x3 image 

<mark>5x5x3 filter</mark> 

**Convolve** the filter with the image i.e. “slide over the image spatially, computing dot products” 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 30 

**April 18, 2017** 

#### Convolution Layer 

###### **activation map** 

<mark>1</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 32 

**April 18, 2017** 

#### Convolution Layer 

###### consider a second, green filter 

32x32x3 image 5x5x3 filter <mark>32</mark> convolve (slide) over all spatial locations <mark>32</mark> 3 

###### **activation maps** 

<mark>28</mark> 

<mark>28</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 33 

**April 18, 2017** 

###### For example, if we had 6 5x5 filters, we’ll get 6 separate activation maps: 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 34 

**April 18, 2017** 

###### **Preview:** ConvNet is a sequence of Convolution Layers, interspersed with activation functions 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 35 

**April 18, 2017** 

###### **Preview:** ConvNet is a sequence of Convolutional Layers, interspersed with activation functions 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 36 

**April 18, 2017** 

# ~~er~~ rr 

RELU RELU RELU RELU RELU CONVIr CONV"| CONVro" 

RELU RELU 

EC 

ecm —= | al) RRS) | | =n™™ BHE--BHe--&SCeSSececcece— = truck oN ‘eS eCOS IES eCSsc Aan—_an—_—_ — | Te 2 r |r = = | ||| Be ae iz lhorse A A Fe) a) | | LL eg LI S| | 

###### A closer look at spatial dimensions: 

###### **activation map** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 41 

**April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 42 

**April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 43 

**April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter 7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 44 **April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter 7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 45 **April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter **=> 5x5 output** 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 46 

**April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7 

7x7 input (spatially) assume 3x3 filter applied **with stride 2** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 47 

**April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter applied **with stride 2** 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 48 **April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter applied **with stride 2 => 3x3 output!** 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 49 **April 18, 2017** 

###### A closer look at spatial dimensions: 

7 

7x7 input (spatially) assume 3x3 filter applied **with stride 3?** 

7 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 50 **April 18, 2017** 

###### A closer look at spatial dimensions: 

7 7x7 input (spatially) assume 3x3 filter applied **with stride 3?** 7 **doesn’t fit!** cannot apply 3x3 filter on 7x7 input with stride 3. 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 51 **April 18, 2017** 

<mark>N</mark> 

<mark>F N F</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

Output size: **(N - F) / stride + 1** e.g. N = 7, F = 3: stride 1 => (7 - 3)/1 + 1 = 5 stride 2 => (7 - 3)/2 + 1 = 3 stride 3 => (7 - 3)/3 + 1 = 2.33 :\ 

**April 18, 2017** 

**Lecture 5 -** 52 

#### In practice: Common to zero pad the border 

0 0 0 0 0 0 e.g. input 7x7 0 **3x3** filter, applied with **stride 1** 0 **pad with 1 pixel** border => what is the output? 0 0 (recall:) (N - F) / stride + 1 **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 5 -** 53 **April 18, 2017** 

#### In practice: Common to zero pad the border 

0 0 0 0 0 0 e.g. input 7x7 0 **3x3** filter, applied with **stride 1** 0 **pad with 1 pixel** border => what is the output? 0 **7x7 output!** 0 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 54 **April 18, 2017** 

#### In practice: Common to zero pad the border 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 55 

**April 18, 2017** 

###### **Remember back to…** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 56 

**April 18, 2017** 

#### Examples time: 

Input volume: **32x32x3** 10 5x5 filters with stride 1, pad 2 Output volume size: ? 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 57 

**April 18, 2017** 

#### Examples time: 

Input volume: **32x32x3** 10 5x5 filters with stride 1, pad 2 

Output volume size: - (32+2*2 5)/1+1 = 32 spatially, so **32x32x10** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 58 

**April 18, 2017** 

#### Examples time: 

Input volume: **32x32x3** 10 5x5 filters with stride 1, pad 2 Number of parameters in this layer? 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 59 

**April 18, 2017** 

#### Examples time: 

Input volume: **32x32x3** 10 5x5 filters with stride 1, pad 2 Number of parameters in this layer? each filter has 5*5*3 + 1 = 76 params      (+1 for bias) * => 76 10 = **760** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 60 

**April 18, 2017** 

Summary. To summarize, the Conv Layer: 

e Accepts a volume of size W, x H,; x D, e Requires four hyperparameters: o Number of filters K, o their spatial extent F, width| height of the filtev the filkew size is Fat xD, o the stride S, © the amount of zero padding P. e Produces a volume of size Wy x Hy x Dz where: °o W,=(W, —F+2P)/S+1 ° Hy = (AM, — F+2P)/S +1 (ie. width and height are computed equally by symmetry) ° Ds, Me ; ¢ With parameter sharing, it introduces F'- F - D, weights per filter, for a total of (F'- F - D,) - K weights and K biases. e Inthe output volume, the d-th depth slice (of size W2 x Hy) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias. 

###### Summary. To summarize, the Conv Layer: 

- e Accepts a volume of sizeW, x H,; x D, e Requires four hyperparameters: o Number of filters K, © their spatial extent F’, o the stride S, o the amount of zero padding P. 

e Produces a volume of size Wy x Hy x Dz where: 

°o W,=(W, —F+2P)/S+1 ° Hy = (AM, — F+2P)/S +1 (ie. width and height are computed equally by symmetry) ° Ds, Me ; ¢ With parameter sharing, it introduces F'- F - D, weights per filter, for a total of (F'- F - D,) - K weights and K biases. 

e Inthe output volume, the d-th depth slice (of size W2 x Hy) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias. 

###### (btw, 1x1 convolution layers make perfect sense) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 63 

**April 18, 2017** 

||SpatialConvolution|
|---|---|
||module = nn.SpatialConvolution(nInputPlane,<br>nOutputPlane,<br>kW,<br>kH,<br>[dW],<br>[dH],<br>[padW],<br>[padH])|
||Applies a 2D convolution over an input image composed of several input planes. The input tensorin forward(input)<br>is<br>expected to be a 3D tensor(nInputPlane x height x width ).<br>The parameters are the following:|
||¢<br>nInputPlane<br>: The number ofexpected inputplanes<br>intheimagegiven into forward()<br>. input depth<br>* nOutputPlane :The numberofoutputplanestheconvolution layerwillproduce. owkputdepth<br>e kw:Thekernelwidthofthesional fitter size<br>¢<br>kH:<br>The kernel height of the convolution<br>e dw:Thestepofthe convolution inthewidthdimension. Defaultis 1.4 stn<br>|<br>e<br>dH:<br>The step of the convolution in the height dimension. Default is 1<br>.<br>¢<br>padw<br>: The additional zeros added per width to the input planes. Default is<br>© ,a good numberis<br>(kw-1)/2<br>.<br>¢<br>padH<br>: The additional zeros added per height to the input planes. Default is padw ,a good numberis<br>(kH-1)/2.|
|:<br>Summary. To summarize, the Conv Layer:|Notethatdependingofthesize ofyourkernel, several (ofthe last)columns orrowsofthe inputimage mightbe lost. It isup<br>See<br>to the user to add proper padding in images.|
|e« Accepts avolume ofsizeW, x Ay, x D;<br>e<br>Requires four hyperparameters:<br>© Number of filters K,<br>© their spatial extent F'<br>;<br>‘<br>©<br>the stride S,|Ifthe input image is a3Dtensor nInputPlane x height x width<br>, theoutputimage size will be nOutputPlane x oheight x<br>owidth where<br>owidth<br>= floor((width<br>+ 2*padW<br>- kW) / dW + 1)<br>oheight<br>= floor((height<br>+ 2*padH<br>-<br>kH)<br>/ dH<br>+<br>1)|
|© theamount ofzero padding P.||

layer { name: “convl" type: "Convolution" bottom: “data" top: "“convl" # learning rate and decay multipliers for the filters param { lr_mult: 1 decay_mult: 1 } # learning rate and decay multipliers for the biases param { lr_mult: 2 decay_mult: 0 } convolution_param { num_output: 96 # learn 96 filters kernel_size: 11 # each filter is 11xl1l stride: 4 # step 4 pixels between each filter application weight_filler { type: “gaussian” # initialize the filters from a Gaussian std: 0.01 # distribution with stdev 0.01 (default mean: 0) } Summary. To summarize, the Conv Conv Layer: bias_filler f{ type: “constant" # initialize——— 8 8 the biases to zero (0) ¢ Accepts a volume of size W, size W, W, x Hy x D, } —— —— —— e Requires four hyperparameters: } © Number of filters K, K, } © their spatial extent F’, © the stride S, © the amount amount of zero zero padding P. 

###### The brain/neuron view of CONV Layer 

32x32x3 image 5x5x3 filter <mark>32</mark> **1 number:** <mark>32</mark> 3 

the result of taking a dot product between the filter and this part of the image (i.e. 5*5*3 = 75-dimensional dot product) 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 66 

**April 18, 2017** 

# ~~er~~ rr 

RELU RELU RELU RELU RELU CONVIr CONV"| CONVro" 

RELU RELU 

EC 

ecm —= | al) RRS) | | =n™™ BHE--BHe--&SCeSSececcece— = truck oN ‘eS eCOS IES eCSsc Aan—_an—_—_ — | Te 2 r |r = = | ||| Be ae iz lhorse A A Fe) a) | | LL eg LI S| | 

_ clown samp ing 

(| da 112x112x64 pp . Oy downsampling oe are =112 112 224 

#### MAX POOLING 

###### Sin le de th slice <u>g p</u> 

<mark>y</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 73 

**April 18, 2017** 

e Accepts a volume of size W, x H,; x D, 

e Requires three hyperparameters: © their spatial extent) Size of pritins fitter © the stride S, e Produces a volume of size Wz x Hy x D2 where: o W,=(W, —F)/S+1 ° Hy =(H, —F)/S+1 ° Ds = D; e Introduces zero parameters since it computesa fixed function of the input ¢ Note that it is not common to use zero-padding for Pooling layers 

e Accepts a volume of size W, x H,; x D, 

e Requires three hyperparameters: © their spatial extent F’, © the stride S, e Produces a volume of size Wz x Hy x D2 where: o W,=(W, —F)/S+1 ° Hy =(H, —F)/S+1 ° Ds = D; e Introduces zero parameters since it computesa fixed function of the input ¢ Note that it is not common to use zero-padding for Pooling layers 

### Summary 

- ConvNets stack CONV,POOL,FC layers 

- Trend towards smaller filters and deeper architectures - Trend towards getting rid of POOL/FC layers (just CONV) - Typical architectures look like **[(CONV-RELU)*N-POOL?]*M-(FC-RELU)*K,SOFTMAX** 

- where N is usually up to ~5, M is large, 0 <= K <= 2. - but recent advances such as ResNet/GoogLeNet challenge this paradigm 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 5 -** 78 

**April 18, 2017**

---

## 源文件

- [cs231n_2017_lecture5.pdf](attachments/documents/AI_CNN-7f417cdfbdc0/cs231n_2017_lecture5.pdf)
