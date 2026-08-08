---
title: convolutional-networks
source: converted:attachments/documents/AI_CNN-6db844467f5f/convolutional-networks.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-6db844467f5f/convolutional-networks.pdf
  title: convolutional-networks.pdf
---

# convolutional-networks

CS231n Convolutional Neural Networks for Visual Recognition 

#### Table of Contents: 

Architecture Overview ConvNet Layers Convolutional Layer Pooling Layer Normalization Layer Fully-Connected Layer Converting Fully-Connected Layers to Convolutional Layers ConvNet Architectures Layer Patterns Layer Sizing Patterns Case Studies (LeNet / AlexNet / ZFNet / GoogLeNet / VGGNet) Computational Considerations Additional References 

# Convolutional Neural Networks (CNNs / ConvNets) 

Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply. 

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more eQcient to implement and vastly reduce the amount of parameters in the network. 

## Architecture Overview 

Recall: Regular Neural Nets. As we saw in the previous chapter, Neural Networks receive an input (a single vector), and transform it through a series of hidden layers. Each hidden layer is made up of a set of neurons, where each 

## Layers used to build ConvNets 

As we described above, a simple ConvNet is a sequence of layers, and every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: **Convolutional Layer** , **Pooling Layer** , and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet **architecture** . 

Example Architecture: Overview. We will go into more details below, but a simple ConvNet for CIFAR-10 classiWcation could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail: 

- INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B. 

- CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 Wlters. 

- RELU layer will apply an elementwise activation function, such as the _max_ (0, _x_ ) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]). 

- POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12]. 

- FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume. 

In this way, ConvNets transform the original image layer by layer from the original pixel values to the Wnal class scores. Note that some layers contain parameters and other don’t. In particular, the CONV/FC layers perform transformations that are a function of not only the activations in the input volume, but also of the parameters (the weights and biases of the neurons). On the other hand, the RELU/POOL layers will implement a Wxed function. The parameters in the CONV/FC layers will be trained with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image. 

#### In summary: 

- A ConvNet architecture is in the simplest case a list of Layers that transform the image volume into an output volume (e.g. holding the class scores) 

- There are a few distinct types of Layers (e.g. CONV/FC/RELU/POOL are by far the most popular) 

- Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function 

Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t) 

e@ 

**Left:** An example input volume in red (e.g. a 32x32x3 CIFAR-10 image), and an example volume of neurons in the Trst Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input - see discussion of depth columns in text below. **Right:** The neurons from the Neural Network chapter remain unchanged: They still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially. 

**Spatial arrangement** . We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven’t yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the **depth, stride** and **zero-padding** . We discuss these next: 

1. First, the **depth** of the output volume is a hyperparameter: it corresponds to the number of Wlters we would like to use, each learning to look for something different in the input. For example, if the Wrst Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edges, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a **depth column** (some people also prefer the term Wbre). 

2. Second, we must specify the **stride** with which we slide the Wlter. When the stride is 1 then we move the Wlters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the Wlters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially. 

3. As we will soon see, sometimes it will be convenient to pad the input volume with zeros around the border. The size of this **zero-padding** is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we’ll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same). 

We can compute the spatial size of the output volume as a function of the input volume size ( _W_ ), the receptive Weld size of the Conv Layer neurons ( _F_ ), the stride with which they are applied ( _S_ ), and the amount of zero padding used ( _P_ ) on the border. You can convince yourself that the correct formula for calculating how many neurons “Wt” is given by ( _W_ − _F_ + 2 _P_ )/ _S_ + 1. For example for a 7x7 input and a 3x3 Wlter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output. Lets also see one more graphical example: 

claims that the input images were 224x224, which is surely incorrect because (224 - 11)/4 + 1 is quite clearly not an integer. This has confused many people in the history of ConvNets and little is known about what happened. My own best guess is that Alex used zero-padding of 3 extra pixels that he does not mention in the paper. 

**Parameter Sharing.** Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are 55*55*96 = 290,400 neurons in the Wrst Conv Layer, and each has 11*11*3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 parameters on the Wrst layer of the ConvNet alone. Clearly, this number is very high. 

It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a **depth slice** (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constrain the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the Wrst Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96*11*11*3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55*55 neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice. 

Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the CONV layer can in each depth slice be computed as a **convolution** of the neuron’s weights with the input volume (Hence the name: Convolutional Layer). This is why it is common to refer to the sets of weights as a **Tlter** (or a **kernel** ), that is convolved with the input. 

Example Tlters learned by Krizhevsky et al. Each of the 96 Tlters shown here is of size [11x11x3], and each one is shared by the 55*55 neurons in one depth slice. Notice that the parameter sharing assumption is relatively reasonable: If detecting a horizontal edge is important at some location in the image, it should intuitively be useful at some other location as well due to 

the translationally-invariant structure of images. There is therefore no need to relearn to detect a horizontal edge at every one of the 55*55 distinct locations in the Conv layer output volume. 

Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some speciWc centered structure, where we should expect, for example, that completely different features should be learned on one side of the image than another. One practical example is when the input are faces that have been centered in the image. You might expect that different eye-speciWc or hairspeciWc features could (and should) be learned in different spatial locations. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a **Locally-Connected Layer** . 

**Numpy examples.** To make the discussion above more concrete, lets express the same ideas but in code and with a speciWc example. Suppose that the input volume is a numpy array `X` . Then: 

- A depth column (or a Wbre) at position `(x,y)` would be the activations `X[x,y,:]` . 

- A depth slice, or equivalently an activation map at depth `d` would be the activations `X[:,:,d]` . 

Conv Layer Example. Suppose that the input volume `X` has shape `X.shape: (11,11,4)` . Suppose further that we use no zero padding ( _P_ = 0), that the Wlter size is _F_ = 5, and that the stride is _S_ = 2. The output volume would therefore have spatial size (11-5)/2+1 = 4, giving a volume with width and height of 4. The activation map in the output volume (call it `V` ), would then look as follows (only some of the elements are computed in this example): 

```
V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0
V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0
V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0
V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0
```

Remember that in numpy, the operation `*` above denotes elementwise multiplication between the arrays. Notice also that the weight vector `W0` is the weight vector of that neuron and `b0` is the bias. Here, `W0` is assumed to be of shape `W0.shape: (5,5,4)` , since the Wlter size is 5 and the depth of the input volume is 4. Notice that at each point, we are computing the dot product as seen before in ordinary neural networks. Also, we see that we are using the same weight and bias (due to parameter sharing), and where the dimensions along the width are increasing in steps of 2 (i.e. the stride). To construct a second activation map in the output volume, we would have: 

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1 V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1 V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1 V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1` 

- `V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1` (example of going along y) `V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1` (or along both) 

where we see that we are indexing into the second depth dimension in `V` (at index 1) because we are computing the second activation map, and that a different set of parameters ( `W1` ) is now used. In the example above, we are for brevity leaving out some of the other operations the Conv Layer would perform to Wll the other parts of the output array `V` . Additionally, recall that these activation maps are often followed elementwise through an activation function such as ReLU, but this is not shown here. 

**Summary** . To summarize, the Conv Layer: 

Accepts a volume of size _W_ 1 × _H_ 1 × _D_ 1 Requires four hyperparameters: 

   - Number of Wlters _K_ , 

   - their spatial extent _F_ , the stride _S_ , 

   - the amount of zero padding _P_ . 

- Produces a volume of size _W H D_ 2 × 2 × 2 

- where: 

   - _H_ 2 = ( _H_ 1 − _F_ + 2 _P_ )/ _S_ + 1 (i.e. width and height are computed equally by symmetry) _D_ = _K_ 2 

- With parameter sharing, it introduces _F_ ⋅ _F_ ⋅ _D_ 1 weights per Wlter, for a total of ( _F_ ⋅ _F_ ⋅ _D_ 1) ⋅ _K_ weights and _K_ biases. 

- In the output volume, the _d_ -th depth slice (of size _W_ 2 × _H_ 2) is the result of performing a valid convolution of the _d_ -th Wlter over the input volume with a stride of _S_ , and then offset by _d_ -th bias. 

A common setting of the hyperparameters is _F_ = 3, _S_ = 1, _P_ = 1. However, there are common conventions and rules of thumb that motivate these hyperparameters. See the ConvNet architectures section below. 

**Convolution Demo** . Below is a running demo of a CONV layer. Since 3D volumes are hard to visualize, all the volumes (the input volume (in blue), the weight volumes (in red), the output volume (in green)) are visualized with each depth slice stacked in rows. The input volume is of size _W_ 1 = 5, _H_ 1 = 5, _D_ 1 = 3, and the CONV layer parameters are _K_ = 2, _F_ = 3, _S_ = 2, _P_ = 1. That is, we have two Wlters of size 3 × 3, and they are applied with a stride of 2. Therefore, the output volume size has spatial size (5 - 3 + 2)/2 + 1 = 3. Moreover, notice that a padding of _P_ = 1 is applied to the input volume, making the outer border of the input volume zero. The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the Wlter (red), summing it up, and then offsetting the result by the bias. 

|Inp|ut V|olume (+pad 1) (7x7x3)<br>Filter W0 (3x3x3)|Filter W1 (3x3x3)|Output Volume (3x3x2)|
|---|---|---|---|---|
|`x[:`|`,:`|`,0]`<br>`w0[:,:,0]`|`w1[:,:,0]`|`o[:,:,0]`|
|0|0|0<br>0<br>0<br>0<br>0<br>-1<br>0<br>1|-1<br>0<br>-1|1<br>1<br>4|
|0|2|0<br>2<br>2<br>2<br>0<br>-1<br>1<br>0|1<br>1<br>-1|3<br>7<br>2|
|0|0|2<br>1<br>1<br>0<br>0<br>1<br>-1<br>-1|1<br>0<br>-1|6<br>0<br>1|
|0|0|1<br>1<br>0<br>2<br>0<br>`w0[:,:,1]`|`w1[:,:,1]`|`o[:,:,1]`|
|0|1|1<br>1<br>1<br>1<br>0<br>0<br>1<br>0|1<br>-1<br>-1|-5<br>-4<br>7|
|0|2|1<br>0<br>2<br>1<br>0<br>-1<br>1<br>0|-1<br>-1<br>-1|-11<br>-11<br>4|
|0|0|0<br>0<br>0<br>0<br>0<br>1<br>0<br>-1|1<br>-1<br>-1|-4<br>-3<br>4|
|`x[:`<br>0|`,:`<br>0|`,1]`<br>0<br>0<br>0<br>0<br>0<br>`w0[:,:,2]`<br>0<br>0<br>1|`w1[:,:,2]`<br>1<br>0<br>-1||
|0|0|0<br>0<br>2<br>1<br>0<br>1<br>0<br>0|-1<br>1<br>0||
|0|1|1<br>1<br>2<br>0<br>0<br>0<br>1<br>-1|1<br>-1<br>-1||
|0|2|0<br>2<br>2<br>1<br>0<br>Bias b0 (1x1x1)|Bias b1 (1x1x1)||
|0<br>0<br>0<br>`x[:`<br>0<br>0|2<br>0<br>0<br>`,:`<br>0<br>0|2<br>2<br>0<br>0<br>0<br>0<br>0<br>0<br>0<br>1<br>0<br>0<br>0<br>0<br>0<br>`,2]`<br>0<br>1<br>0<br>1<br>0<br>0<br>0<br>2<br>0<br>0<br> <br>`b0[:,:,0]`<br>1|<br>`b1[:,:,0]`<br>0<br>toggle m|ovement|
|0|2|1<br>0<br>2<br>1<br>0|||
|0|2|1<br>0<br>1<br>1<br>0|||
|0|0|0<br>2<br>2<br>0<br>0|||
|0|2|0<br>2<br>1<br>2<br>0|||
|0|0|0<br>0<br>0<br>0<br>0|||

**Implementation as Matrix Multiplication** . Note that the convolution operation essentially performs dot products between the Wlters and local regions of the input. A common implementation pattern of the CONV layer is to take advantage of this fact and formulate the forward pass of a convolutional layer as one big matrix multiply as follows: 

1. The local regions in the input image are stretched out into columns in an operation commonly called **im2col** . For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 Wlters at stride 4, then we 

would take [11x11x3] blocks of pixels in the input and stretch each block into a column vector of size 11*11*3 = 363. Iterating this process in the input at stride of 4 gives (227-11)/4+1 = 55 locations along both width and height, leading to an output matrix `X_col` of im2col of size [363 x 3025], where every column is a stretched out receptive Weld and there are 55*55 = 3025 of them in total. Note that since the receptive Welds overlap, every number in the input volume may be duplicated in multiple distinct columns. 

2. The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 Wlters of size [11x11x3] this would give a matrix `W_row` of size [96 x 363]. 

3. The result of a convolution is now equivalent to performing one large matrix multiply <mark>`np.dot(W_row, X_col)`</mark> , which evaluates the dot product between every Wlter and every receptive Weld location. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each Wlter at each location. 

4. The result must Wnally be reshaped back to its proper output dimension [55x55x96]. 

This approach has the downside that it can use a lot of memory, since some values in the input volume are replicated multiple times in `X_col` . However, the beneWt is that there are many very eQcient implementations of Matrix Multiplication that we can take advantage of (for example, in the commonly used BLAS API). Moreover, the same im2col idea can be reused to perform the pooling operation, which we discuss next. 

**Backpropagation.** The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-pipped Wlters). This is easy to derive in the 1-dimensional case with a toy example (not expanded on for now). 

**1x1 convolution** . As an aside, several papers use 1x1 convolutions, as Wrst investigated by Network in Network. Some people are at Wrst confused to see 1x1 convolutions especially when they come from signal processing background. Normally signals are 2-dimensional so 1x1 convolutions do not make sense (it’s just pointwise scaling). However, in ConvNets this is not the case because one must remember that we operate over 3- dimensional volumes, and that the Wlters always extend through the full depth of the input volume. For example, if the input is [32x32x3] then doing 1x1 convolutions would effectively be doing 3-dimensional dot products (since the input depth is 3 channels). 

**Dilated convolutions.** A recent development (e.g. see paper by Fisher Yu and Vladlen Koltun) is to introduce one more hyperparameter to the CONV layer called the dilation. So far we’ve only discussed CONV Wlters that are contiguous. However, it’s possible to have Wlters that have spaces between each cell, called dilation. As an example, in one dimension a Wlter `w` of size 3 would compute over input `x` the following: <mark>`w[0]*x[0] + w[1]*x[1] + w[2]*x[2]`</mark> . This is dilation of 0. For dilation 1 the Wlter would instead compute <mark>`w[0]*x[0] + w[1]*x[2] + w[2]*x[4]`</mark> ; In other words there is a gap of 1 between the applications. This can be very useful in some settings to use in conjunction with 0-dilated Wlters because it allows you to merge spatial information across the inputs much more agressively with fewer layers. For example, if you stack two 3x3 CONV layers on top of each other then you can convince yourself that the neurons on the 2nd layer are a function of a 5x5 patch of the input 

(we would say that the effective receptive Weld of these neurons is 5x5). If we use dilated convolutions then this effective receptive Weld would grow much quicker. 

### Pooling Layer 

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overWtting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with Wlters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer: 

Accepts a volume of size _W_ 1 × _H_ 1 × _D_ 1 Requires two hyperparameters: their spatial extent _F_ , the stride _S_ , 

Produces a volume of size _W H D_ 2 × 2 × 2 where: 

Introduces zero parameters since it computes a Wxed function of the input For Pooling layers, it is not common to pad the input using zero-padding. 

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with _F_ = 3, _S_ = 2 (also called overlapping pooling), and more commonly _F_ = 2, _S_ = 2. Pooling sizes with larger receptive Welds are too destructive. 

**General pooling** . In addition to max pooling, the pooling units can also perform other functions, such as average pooling or even L2-norm pooling. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice. 

Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. See the Neural Network section of the notes for more information. 

### Converting FC layers to CONV layers 

It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. However, the neurons in both layers still compute dot products, so their functional form is identical. Therefore, it turns out that it’s possible to convert between FC and CONV layers: 

- For any CONV layer there is an FC layer that implements the same forward function. The weight matrix would be a large matrix that is mostly zero except for at certain blocks (due to local connectivity) where the weights in many of the blocks are equal (due to parameter sharing). 

- Conversely, any FC layer can be converted to a CONV layer. For example, an FC layer with _K_ = 4096 that is looking at some input volume of size 7 × 7 × 512 can be equivalently expressed as a CONV layer with _F_ = 7, _P_ = 0, _S_ = 1, _K_ = 4096. In other words, we are setting the Wlter size to be exactly the size of the input volume, and hence the output will simply be 1 × 1 × 4096 since only a single depth column “Wts” across the input volume, giving identical result as the initial FC layer. 

**FC->CONV conversion** . Of these two conversions, the ability to convert an FC layer to a CONV layer is particularly useful in practice. Consider a ConvNet architecture that takes a 224x224x3 image, and then uses a series of CONV layers and POOL layers to reduce the image to an activations volume of size 7x7x512 (in an AlexNet architecture that we’ll see later, this is done by use of 5 pooling layers that downsample the input spatially by a factor of two each time, making the Wnal spatial size 224/2/2/2/2/2 = 7). From there, an AlexNet uses two FC layers of size 4096 and Wnally the last FC layers with 1000 neurons that compute the class scores. We can convert each of these three FC layers to CONV layers as described above: 

- Replace the Wrst FC layer that looks at [7x7x512] volume with a CONV layer that uses Wlter size _F_ = 7, giving output volume [1x1x4096]. 

- Replace the second FC layer with a CONV layer that uses Wlter size _F_ = 1, giving output volume [1x1x4096] Replace the last FC layer similarly, with _F_ = 1, giving Wnal output [1x1x1000] 

Each of these conversions could in practice involve manipulating (e.g. reshaping) the weight matrix _W_ in each FC layer into CONV layer Wlters. It turns out that this conversion allows us to “slide” the original ConvNet very eQciently across many spatial positions in a larger image, in a single forward pass. 

For example, if 224x224 image gives a volume of size [7x7x512] - i.e. a reduction by 32, then forwarding an image of size 384x384 through the converted architecture would give the equivalent volume in size [12x12x512], since 384/32 = 12. Following through with the next 3 CONV layers that we just converted from FC layers would now give 

the Wnal volume of size [6x6x1000], since (12 - 7)/1 + 1 = 6. Note that instead of a single vector of class scores of size [1x1x1000], we’re now getting an entire 6x6 array of class scores across the 384x384 image. 

Evaluating the original ConvNet (with FC layers) independently across 224x224 crops of the 384x384 image in strides of 32 pixels gives an identical result to forwarding the converted ConvNet one time. 

Naturally, forwarding the converted ConvNet a single time is much more eQcient than iterating the original ConvNet over all those 36 locations, since the 36 evaluations share computation. This trick is often used in practice to get better performance, where for example, it is common to resize an image to make it bigger, use a converted ConvNet to evaluate the class scores at many spatial positions and then average the class scores. 

Lastly, what if we wanted to eQciently apply the original ConvNet over the image but at a stride smaller than 32 pixels? We could achieve this with multiple forward passes. For example, note that if we wanted to use a stride of 16 pixels we could do so by combining the volumes received by forwarding the converted ConvNet twice: First over the original image and second over the image but with the image shifted spatially by 16 pixels along both width and height. 

An IPython Notebook on Net Surgery shows how to perform the conversion in practice, in code (using Caffe) 

## ConvNet Architectures 

We have seen that Convolutional Networks are commonly made up of only three layer types: CONV, POOL (we assume Max pool unless stated otherwise) and FC (short for fully-connected). We will also explicitly write the RELU activation function as a layer, which applies elementwise non-linearity. In this section we discuss how these are commonly stacked together to form entire ConvNets. 

### Layer Patterns 

The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern: 

##### `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC` 

where the `*` indicates repetition, and the `POOL?` indicates an optional pooling layer. Moreover, `N >= 0` (and usually `N <= 3` ), `M >= 0` , `K >= 0` (and usually `K < 3` ). For example, here are some common ConvNet 

architectures you may see that follow this pattern: 

- `INPUT -> FC` , implements a linear classiWer. Here `N = M = K = 0` . 

- `INPUT -> CONV -> RELU -> FC` 

- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC` . Here we see that there is a single 

- CONV layer between every POOL layer. 

- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` Here we 

- see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation. 

Prefer a stack of small Wlter CONV to one large receptive Weld CONV layer. Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the Wrst CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the Wrst CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive Welds. These neurons would have a receptive Weld size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the = volumes have _C_ channels, then it can be seen that the single 7x7 CONV layer would contain _C_ × (7 × 7 × _C_ ) 49 _C_<sup>2</sup> = parameters, while the three 3x3 CONV layers would only contain 3 × ( _C_ × (3 × 3 × _C_ )) 27 _C_<sup>2</sup> parameters. Intuitively, stacking CONV layers with tiny Wlters as opposed to having one CONV layer with big Wlters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation. 

**Recent departures.** It should be noted that the conventional paradigm of a linear list of layers has recently been challenged, in Google’s Inception architectures and also in current (state of the art) Residual Networks from Microsoft Research Asia. Both of these (see details below in case studies section) feature more intricate and different connectivity structures. 

**In practice: use whatever works best on ImageNet** . If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “don’t be a hero”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and Wnetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch. I also made this point at the Deep Learning school. 

### Layer Sizing Patterns 

Until now we’ve omitted mentions of common hyperparameters used in each of the layers in a ConvNet. We will Wrst state the common rules of thumb for sizing the architectures and then follow the rules with a discussion of the notation: 

The **input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512. 

The **conv layers** should be using small Wlters (e.g. 3x3 or at most 5x5), using a stride of _S_ = 1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when _F_ = 3, then using _P_ = 1 will retain the original size of the input. When _F_ = 5, _P_ = 2. For a general _F_ , it can be seen that _P_ = ( _F_ −1)/2 preserves the input size. If you must use bigger Wlter sizes (such as 7x7 or so), it is only common to see this on the very Wrst conv layer that is looking at the input image. 

The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive Welds (i.e. _F_ = 2), and with a stride of 2 (i.e. _S_ = 2). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another slightly less common setting is to use 3x3 receptive Welds with a stride of 2, but this makes. It is very uncommon to see receptive Weld sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance. 

Reducing sizing headaches. The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don’t zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and Wlters “work out”, and that the ConvNet architecture is nicely and symmetrically wired. 

Why use stride of 1 in CONV? Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise. 

Why use padding? In addition to the aforementioned beneWt of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly. 

Compromising based on memory constraints. In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, Wltering a 224x224x3 image with three 3x3 CONV layers with 64 Wlters each and padding 1 would create three activation 

volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the Wrst CONV layer of the network. For example, one compromise might be to use a Wrst CONV layer with Wlter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses Wlter sizes of 11x11 and stride of 4. 

### Case studies 

There are several architectures in the Weld of Convolutional Networks that have a name. The most common are: 

- **LeNet** . The Wrst successful applications of Convolutional Networks were developed by Yann LeCun in 1990’s. Of these, the best known is the LeNet architecture that was used to read zip codes, digits, etc. **AlexNet** . The Wrst work that popularized Convolutional Networks in Computer Vision was the AlexNet, developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the ImageNet ILSVRC challenge in 2012 and signiWcantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer). 

- **ZF Net** . The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the ZFNet (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers and making the stride and Wlter size on the Wrst layer smaller. 

- **GoogLeNet** . The ILSVRC 2014 winner was a Convolutional Network from Szegedy et al. from Google. Its main contribution was the development of an Inception Module that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently Inception-v4. 

- **VGGNet** . The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the VGGNet. Its main contribution was in showing that the depth of the network is a critical component for good performance. Their Wnal best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. Their pretrained model is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M). Most of these parameters are in the Wrst fully connected layer, and it was since found that these FC layers can be removed with no performance downgrade, signiWcantly reducing the number of necessary parameters. 

- **ResNet** . Residual Network developed by Kaiming He et al. was the winner of ILSVRC 2015. It features 

special skip connections and a heavy use of batch normalization. The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming’s presentation (video, slides), and some recent experiments that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from Kaiming He et al. Identity Mappings in Deep Residual Networks (published March 2016). 

**VGGNet in detail** . Lets break down the VGGNet in more detail as a case study. The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding). We can write out the size of the representation at each step of the processing and keep track of both the representation size and the total number of weights: 

```
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
```

```
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
```

```
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
```

```
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
```

```
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
```

```
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000
```

```
TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters
```

As is common with Convolutional Networks, notice that most of the memory (and also compute time) is used in the early CONV layers, and that most of the parameters are in the last FC layers. In this particular case, the Wrst FC 

layer contains 100M weights, out of a total of 140M. 

### Computational Considerations 

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of: 

- From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. Wrst Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below. From the parameter sizes: These are the numbers that hold the network **parameters** , their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so. 

- Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc. 

Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every poating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and Wnally GB. If your network doesn’t Wt, a common heuristic to “make it Wt” is to decrease the batch size, since most of the memory is usually consumed by the activations. 

## Additional Resources 

Additional resources related to implementation: 

Soumith benchmarks for CONV performance 

- ConvNetJS CIFAR-10 demo allows you to play with ConvNet architectures and see the results and computations in real time, in the browser. 

Caffe, one of the popular ConvNet libraries. 

State of the art ResNets in Torch7 

cs231n cs231n karpathy@cs.stanford.edu

---

## 源文件

- [convolutional-networks.pdf](attachments/documents/AI_CNN-6db844467f5f/convolutional-networks.pdf)
