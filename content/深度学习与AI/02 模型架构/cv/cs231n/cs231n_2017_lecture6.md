---
title: cs231n_2017_lecture6
source: converted:attachments/documents/AI_CNN-43c4c9904da9/cs231n_2017_lecture6.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-43c4c9904da9/cs231n_2017_lecture6.pdf
  title: cs231n_2017_lecture6.pdf
---

### Lecture 6: Training Neural Networks, Part I 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 1 

**April 20, 2017** 

##### Administrative 

**Assignment 1** due **Thursday (today)** , 11:59pm on Canvas **Assignment 2** out today **Project proposal** due Tuesday April 25 Notes on backprop for a linear layer and vector/tensor derivatives linked to Lecture 4 on syllabus 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 2 

**April 20, 2017** 

# ~~ | ~~<mark>) Sta</mark>~~ n ~~e~~ 4 ~~<mark>5</mark>~~ 

<mark>j=We f = = W2 max(0, Wiz) max(0, Wiz) Wiz)</mark> 

<mark>Sthbho LAT</mark> 

Image Maps 

###### Where we are now... 

###### **Convolutional Layer** 

###### **activation map** 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 6 - 6 April 20, 2017 

###### Where we are now... 

For example, if we had 6 5x5 filters, we’ll get 6 separate activation maps: 

###### **Convolutional Layer** 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 - 7 

April 20, 2017 

while True: 

~~<mark>OO</mark>~~ 

###### Where we are now... 

##### **Mini-batch SGD** 

Loop: 

1. **Sample** a batch of data 

2. **Forward** prop it through the graph (network), get loss 

3. **Backprop** to calculate the gradients 4. **Update** the parameters using the gradient 

> Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 - 9 April 20, 2017 

#### Next: Training Neural Networks 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 10 

**April 20, 2017** 

### Overview 

**1. One time setup** _activation functions, preprocessing, weight initialization, regularization, gradient checking_ **2. Training dynamics** _babysitting the learning process, parameter updates, hyperparameter optimization_ **3. Evaluation** _model ensembles_ 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 11 

**April 20, 2017** 

### Part 1 

- Activation Functions 

- Data Preprocessing 

- Weight Initialization 

- Batch Normalization 

- Babysitting the Learning Process 

- Hyperparameter Optimization 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 12 

**April 20, 2017** 

### Activation Functions 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 13 

**April 20, 2017** 

<mark>o(0) = pha pha</mark> | <mark>tanh(x) Jf. max(0, x) _/</mark> 

<mark>max(0.lx, x Ole)</mark> /<sup><mark>bi, w4x+be)</mark></sup> <mark>max(wi</mark><sup><mark>x +</mark></sup> >0 <mark>Se 1) ; <0 Sf</mark> 

o(x) <mark>=1/(1+e7”)</mark> 

o(x) <mark>=1/(1+e7”)</mark> 

M23if H% 2 x > pews 2 6-9RY 9 Le (yale ———— 08 x so SR Oo <u><mark>o(x) _— 1/(1 “tt e*)</mark></u> 0.6 / — ~~——_____~~ ea Ox Ox Oo da B otOW” _ ok5a fa). owa2 almost 2evo if 2 Is ‘v from 0 ° oe is almost 2ev0, So we avegy to mt Py a almost 2evo Thing to the chain 28 is fine . a fp) nepain 2w0 

o(x) <mark>=1/(1+e7”)</mark> 

oo x has positive ¢ negative valves 

o(x) <mark>=1/(1+e7”)</mark> 

Vv 

Grachient = 0 radienct = o Cin prati su) Gradiew| = 4 

active ReLU **DATA CLOUD** dead ReLU will never activate => never update 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 27 **April 20, 2017** 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 28 **April 20, 2017** 

###### f(z) <mark>=</mark> max(0.01z2, z) 

## <mark>f(z) <max(0.01e,2)| 7) =a</mark> 

fii = <mark>1</mark> a(exp(z)—1) **i** fxzfa< > **0** 

#### <mark>max(w)</mark> T xz <mark>+b), wx +</mark> bo) 

###### **TLDR: In practice:** 

- Use ReLU. Be careful with your learning rates - Try out Leaky ReLU / Maxout / ELU - Try out tanh but don’t expect much - Don’t use sigmoid 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 33 

**April 20, 2017** 

### Data Preprocessing 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 34 

**April 20, 2017** 

###### **TLDR: In practice for Images:** center only 

- e.g. consider CIFAR-10 example with [32,32,3] images - Subtract the mean image (e.g. AlexNet) (mean image = [32,32,3] array) 

- - Subtract per-channel mean (e.g. VGGNet) (mean along each channel = 3 numbers) 

Not common to normalize variance, to do PCA or whitening 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 39 

**April 20, 2017** 

### Weight Initialization 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 40 

**April 20, 2017** 

###### W = 0.01* np.random.randn(D,H)|| 

###### W = 0.01* np.random.randn(D,H)|| 

# assume some unit gaussian 10-D input data D = np.random.randn(1000, 500) hidden layer sizes = [500]*10 nonlinearities = ['tanh']*len(hidden_layer sizes) act = {'relu': lambda x:np.maximum(0,x), 'tanh':lambda x:np.tanh(x)} Hsfor= {}i in xrange(len(hidden_layer sizes)): X = D if i == © else Hs[i-1] # input at this layer fan_in = X.shape[1] fan_out = hidden layer sizes[i] W = np.random.randn(fan_in, fan_out) * 0.01 # layer initialization H = np.dot(X, W) # matrix multiply H = act[nonlinearities[i]](H) # nonlinearity Hs[i] = H # cache result on this layer # look at distributions at each layer print ‘input layer had mean %f and std %f' % (np.mean(D), np.std(D)) layer_means = [np.mean(H) for i,H in Hs.iteritems()] layer stds = [np.std(H) for i,H in Hs.iteritems()] for i,H in Hs.iteritems(): print ‘hidden layer %“d had mean %f and std %f' % (i+1, layer_means[i], layer _stds[i]) # plot the means and standard deviations plt.figure() plt.subplot(121) plt.plot(Hs.keys(), layer_means, ‘ob-') plt.title('layer mean') plt.subplot(122) plt.plot(Hs.keys(), layer_stds, ‘or-') plt.title('layer std‘) # plot the raw distributions plt.figure() for i,H in Hs.iteritems(): plt.subplot(1,len(Hs),i+1) plt.hist(H.ravel(), 30, range=(-1,1)) 

input layer had mean 0.000927 and std 0.998388 hidden layer 1 had mean -0.000117 and std 0.213081 hidden layer 2 had mean -0.000001 and std 0.047551 hidden layer 3 had mean -0.000002 and std 0.010630 hidden layer 4 had mean 0.000001 and std 0.002378 hidden layer 5 had mean 0.000002 and std 0.000532 hidden layer 6 had mean -0.000000 and std 0.000119 hidden layer 7 had mean 0.000000 and std 0.000026 hidden layer 8 had mean -0.000000 and std 0.000006 hidden layer 9 had mean 0.000000 and std 0.000001 hidden layer 10 had mean -0.000000 and std 0.000000 0.00002 layer mean — layer std i / ~0. **0** 010}= a AnGOOLEE I ? 3 3 5 r 7 F 9 0005 I 2 3 3 5 6 7 E 9 30000 150q00 150q00 150g00 150900 15000 150400 150400 150400 7 20000 100q00 100400 1wogoo 100900 wooo 100400 190490 100900 bi 10000 Sogo 50900 504900 504900 50400 s0900 50900 50400 a 0-05 00 05 1010-05 00 05 1010-08 00 05 7041 0-05 00 05 701 0-65 00 05 10-10-05 60 05 TO 0-0.5 00 05 104 06-0500 05 Tost 0-05 00 05 Tost 60-0500 05 10 

input layer had mean 0.000927 and std 0.998388 hidden layer 1 had mean -0.000117 and std 0.213081 hidden layer 2 had mean -0.000001 and std 0.047551 hidden layer 3 had mean -0.000002 and std 0.010630 hidden layer 4 had mean 0.000001 and std 0.002378 hidden layer 5 had mean 0.000002 and std 0.000532 hidden layer 6 had mean -0.000000 and std 0.000119 hidden layer 7 had mean 0.000000 and std 0.000026 hidden layer 8 had mean -0.000000 and std 0.000006 hidden layer 9 had mean 0.000000 and std 0.000001 hidden layer 10 had mean -0.000000 and std 0.000000 0.00002 layer mean — layer std i / ~0. **0** 010}= a AnGOOLEE I ? 3 3 5 r 7 F 9 0005 I 2 3 3 5 6 7 E 9 30000 150q00 150q00 150g00 150900 15000 150400 150400 150400 7 20000 100q00 100400 1wogoo 100900 wooo 100400 190490 100900 bi 10000 Sogo 50900 504900 504900 50400 s0900 50900 50400 a 0-05 00 05 1010-05 00 05 1010-08 00 05 7041 0-05 00 05 701 0-65 00 05 10-10-05 60 05 TO 0-0.5 00 05 104 06-0500 05 Tost 0-05 00 05 Tost 60-0500 05 10 

= = 2 = — <mark>W = np.random.randn(fan</mark> in, fan out) * 1.0 # <mark>layer initialization</mark> 

# layer initialization 

input layer had mean 0.000501 and std 0.999444 |W = np.random.randn(fan in, fan out) / np.sqrt(fan in/2) hidden layer 1 had mean 0.562488 and std 0.825232 = = = hidden layer 2 had mean 0.553614 and std 0.827835 hidden layer 3 had mean 0.545867 and std 0.813855 hidden layer 4 had mean 0.565396 and std 0.826902 hidden layer 5 had mean 0.547678 and std 0.834092 hidden layer 6 had mean 0.587103 and std 0.860035 hidden layer 7 had mean 0.596867 and std 0.870610 hidden layer 8 had mean 0.623214 and std 0.889348 hidden layer 9 had mean 0.567498 and std 0.845357 hidden layer 10 had mean 0.552531 and std 0.844523 

input layer had mean 0.000501 and std 0.999444 |W = np.random.randn(fan in, fan out) / np.sqrt(fan in/2) # layer initialization hidden layer 1 had mean 0.562488 and std 0.825232 ie ra — hidden layer 2 had mean 0.553614 and std 0.827835 hidden layer 3 had mean 0.545867 and std 0.813855 hidden layer 4 had mean 0.565396 and std 0.826902 hidden layer 5 had mean 0.547678 and std 0.834092 hidden layer 6 had mean 0.587103 and std 0.860035 hidden layer 7 had mean 0.596867 and std 0.870610 hidden layer 8 had mean 0.623214 and std 0.889348 hidden layer 9 had mean 0.567498 and std 0.845357 hidden layer 10 had mean 0.552531 and std 0.844523 

###### <mark>Proper initialization is an active area of research…</mark> 

**_Understanding the difficulty of training deep feedforward neural networks_** by Glorot and Bengio, 2010 

**_Exact solutions to the nonlinear dynamics of learning in deep linear neural networks_** by Saxe et al, 2013 

**_Random walk initialization for training very deep feedforward networks_** by Sussillo and Abbott, 2014 

**_Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification_** by He et al., 2015 

**_Data-dependent Initializations of Convolutional Neural Networks_** by Krähenbühl et al., 2015 **_All you need is a good init_** , Mishkin and Matas, 2015 

… 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 52 

**April 20, 2017** 

### Batch Normalization 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 53 

**April 20, 2017** 

(hk) alk) — E[x(*)) J Var|x(*))] Var|x(*))] ty recov the data befire beth novma lization, Let Ny to lawn the Y and P 

ysethe mean and sted fiom training coca, olovt calentate new mean std -fom “test data, Same in eoch activation map 

#### Babysitting the Learning Process 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 61 

**April 20, 2017** 

###### **<mark>Step 2: Choose the architecture:</mark>** say we start with one hidden layer of 50 neurons: 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 63 

**April 20, 2017** 

def init two layermodel(inputmodel(input size, hidden size, output size): model = {} model['Wi'] = 0.0001 * np.random.randn(inputsize,size, hidden size) model['bl'] = np.zeros(hiddensize)size) model[‘W2'] = 0.0001 * np.random.randn(hiddensize,size, output size) model['b2'] = np.zeros(outputsize)size) mn model model = init two layermodel(32*32*3,model(32*32*3, 50, 10) # input size, hidden size, number of classes loss, grad = two layer _net(Xtrain, _net(Xtrain,train, model, y_train <mark>[ 0.0)]</mark> 2.print3026121tosprint3026121tos3026121tostos **6** 167 <mark>e</mark> ~~<mark>S</mark>~~ or +t 

def init two layermodel(inputmodel(input size, hidden size, output size): model = {} model['Wi'] = 0.0001 * np.random.randn(inputsize,size, hidden size) model['bl'] = np.zeros(hiddensize)size) model[‘W2'] = 0.0001 * np.random.randn(hiddensize,size, output size) model['b2'] = np.zeros(outputsize)size) mn model model = init two layermodel(32*32*3,model(32*32*3, 50, 10) # ingut size, hidden size, number of classes loss, grad = twolayerlayer print loss net(Xtrain,train, model, y train] train] <mark>1e3) |</mark> 3. 06859716482 06859716482 ____ 

model = init_twolayer model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() Xtiny = X_train[:20] # take 20 examples ytiny = y_train[:20] best model, stats = trainer.train(Xtiny, ytiny, Xtiny, ytiny, model, two layer net, numepochs=200, reg=0.0, update='sgd', learningrate decay=1, sample batches = False, learning rate=le-3, verbose=True) 

modeltrainer= =init_twoClassifierTrainer()layer _model(32*32*3, 50, 10) # input size, hidden size, number of classes X_tiny = X_train[:20] # take 20 examples y_tiny = ytrain[:20] best model, stats = trainer.train(Xtiny, ytiny, Xtiny, ytiny, model, two layer net, numepochs=200, reg=0.0, update='sgd', learningrate decay=1, sample batches = False, learning rate=le-3, verbose=True) Finished epoch 1 / 200: cost 2.302603, train: 0.400000, val 0.400000, Lr 1.9000000e-03 Finished epoch 2 / 200: cost 2.302258, train: 0.450000, val 0.450000, lr 1.000000e-03 Finished epoch 3 / 200: cost 2.301849, train: 0.600000, val 0.600000, Lr 1.000000e-03 Finished epoch 4 / 200: cost 2.301196, train: 6.650000, val 0.650000, lr 1.000000e-03 Finished epoch 5 / 200: cost 2.300044, train: 0.650000, val 0.650000, lr 1.000000e-03 Finished epoch 6 / 200: cost 2.297864, train: 6.550000, val 0.550000, lr 1.000000e-03 Finished epoch 7 / 200: cost 2.293595, train: 0.600000, val 0.600000, lr 1.000000e-03 Finished epoch 8 / 200: cost 2.285096, train: 0.550000, val 0.550000, lr 1.000000e-03 Finished epoch 9 / 200: cost 2.268094, train: 0.550000, val 0.550000, lr 1.000000e-03 Finished epoch 10 / 200: cost 2.234787, train: 0.500000, val 0.500000, Ilr 1.000000e-03 Finished epoch 11 / 200: cost 2.173187, train: 0.500000, val 0.500000, Ilr 1.000000e-03 Finished epoch 12 / 200: cost 2.076862, train: 0.500000, val 0.500000, lr 1.000000e-03 Finished epoch 13 / 200: cost 1.974090, train: 0.400000, val 0.400000, lr 1.000000e-03 Finished epoch 14 / 200: cost 1.895885, train: 0.400000, val 0.400000, lr 1.000000e-03 Finished epoch 15 / 200: cost 1.820876, train: 0.450000, val 0.450000, lr 1.000000e-03 Finished epoch 16 / 200: cost 1.737430, train: 0.450000, val 0.450000, lr 1.000000e-03 Finished epoch 17 / 200: cost 1.642356, train: 0.500000, val 0.500000, lr 1.000000e-03 Finished epoch 18 / 200: cost 1.535239, train: 0.600000, val 0.600000, Ilr 1.000000e-03 Finished epoch 19 / 200: cost 1.421527, train: 0.600000, val 0.600000, lr 1.000000e-03 Pint -bowd ...-k An s: mAN aaa a marten aoe te na erannn =-—_1 a erannn 1. 2 Annnnn~ Ar Finished epoch 195 / 200: cost 0.002694, train: 1.000000, val 1.000000, lr 1.000000e-03 Finished epoch 196 / 200: cost 0.002674, train: 1.000000, val 1.000000, lr 1.000000e-03 Finished epoch 197 / 200: cost 0.002655, train: 1.000000, val 1.000000, lr 1.000000e-03 Finished epoch 198 / 200: cost 0.002635, train: 1.000000, val 1.000000, lr 1.000000e-03 Finished epoch 199 / 200: cost 0.002617, train: 1.000000, val 1.000000, lr 1.000000e-03 Finished epoch 200 / 200: cost 0.002597, train: 1.000000, val 1.000000, lr 1.000000e-03 finished optimization. best validation accuracy: 1.900000 

= 

uo 

model = init two layermodel(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best model, stats = trainer.train(X_ train, ytrain, X_ val, y val, model, two layer net, num_epochs=10, reg=0.000001, update='sgd', learningrate decay=1, sample batches = True, learning rate=le-6, verbose=True) 

|model = <br>trainer|init _tw<br>= Clas|o_layer<br>sifierTra|_model<br>iner(|(32*32*3,<br>)|50,<br>10|) # input|size,|hidden s|ize, number of classes|
|---|---|---|---|---|---|---|---|---|---|
|best mod|el,<br>st|ats = tr|ainer.|train(X_ t<br>mode<br>num<br>upd<br>arp|rain,<br><br>l, two<br>_epochs<br>ate='sg<br><br>CFA|ytrain,<br>X_ <br> layer net<br>=10,<br>reg=0<br>d', le|val,<br>,<br>.0000<br>ar<br>r<br>ec;|yval,<br>01,<br>nin<br>ate decay=|g<br>1,|
|Finished|epoch|1 /<br>10:]||cost|2.302576,||train:|0.080000,|yal|0.103000,|Lr 1.000000e-06|
|Finished|epoch|2 /<br>10:]||cost|2.302582,||train:|0.121000,|wal|0.124000,|Ilr 1.000000e-06|
|Finished|epoch|3 /<br>10:]||cost|2.302558,||train:|0.119000,|yal|0.138000,|Lr 1.000000e-06|
|Finished|epoch|4 /<br>10:]||cost|2.302519,||train:|0.127000,|yal|0.151000,|Lr 1.000000e-06|
|Finished|epoch|5 /<br>10:]||cost|2.302517,||train:|0.158000,|yal|0.171000,|Lr 1.000000e-06|
|Finished|epoch|6 /<br>10:]||cost|2.302518,||train:|0.179000,|yal|0.172000,|Lr 1.000000e-06|
|Finished|epoch|7 / 10:||cost|2.302466,||train:|0.180000,|yal|0.176000,|Lr 1.000000e-06|
|Finished|epoch|8 / 10:||cost|2.302452,||train:|0.175000,|yal|0.185000,|Lr 1.000000e-06|
|Finished|epoch|9 /<br>10:]||cost|2.302459,||train:|0.206000,|yal|0.192000,|Lr 1.000000e-06|
|Finished|epoch|10<br>/<br>10}|cost|2.302420]|traijn|:<br>0.190000|,<br>jva|l 0.192000,|Lr 1.000000e-06|
|finished|optimi|zation.|Lhe|alidatiow|accur|acy:<br>0.192|000|||

|model = <br>trainer|init _tw<br>= Clas|o_layer<br>sifierTra|_model<br>iner(|(32*32*3,<br>)|50,<br>10|) # input|size,|hidden s|ize, number of classes|
|---|---|---|---|---|---|---|---|---|---|
|best mod|el,<br>st|ats = tr|ainer.|train(X_ t<br>mode<br>num<br>upd<br>arp|rain,<br><br>l, two<br>_epochs<br>ate='sg<br><br>CFA|ytrain,<br>X_ <br> layer net<br>=10,<br>reg=0<br>d', le|val,<br>,<br>.0000<br>ar<br>r<br>ec;|yval,<br>01,<br>nin<br>ate decay=|g<br>1,|
|Finished|epoch|1 /<br>10:]||cost|2.302576,||train:|0.080000,|yal|0.103000,|Lr 1.000000e-06|
|Finished|epoch|2 /<br>10:]||cost|2.302582,||train:|0.121000,|wal|0.124000,|Ilr 1.000000e-06|
|Finished|epoch|3 /<br>10:]||cost|2.302558,||train:|0.119000,|yal|0.138000,|Lr 1.000000e-06|
|Finished|epoch|4 /<br>10:]||cost|2.302519,||train:|0.127000,|yal|0.151000,|Lr 1.000000e-06|
|Finished|epoch|5 /<br>10:]||cost|2.302517,||train:|0.158000,|yal|0.171000,|Lr 1.000000e-06|
|Finished|epoch|6 /<br>10:]||cost|2.302518,||train:|0.179000,|yal|0.172000,|Lr 1.000000e-06|
|Finished|epoch|7 / 10:||cost|2.302466,||train:|0.180000,|yal|0.176000,|Lr 1.000000e-06|
|Finished|epoch|8 / 10:||cost|2.302452,||train:|0.175000,|yal|0.185000,|Lr 1.000000e-06|
|Finished|epoch|9 /<br>10:]||cost|2.302459,||train:|0.206000,|yal|0.192000,|Lr 1.000000e-06|
|Finished|epoch|10<br>/<br>10}|cost|2.302420]|traijn|:<br>0.190000|,<br>jva|l 0.192000,|Lr 1.000000e-06|
|finished|optimi|zation.|Lhe|alidatiow|accur|acy:<br>0.192|000|||

model = init _two_ layer_model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best model, stats = trainer.train(X_ train, ytrain, X_ val, y val, model, two layer net, num_epochs=10, reg=0.000001, update='sgd', learningrate decay=1, ethp a e, learning rate=le-6,] verbose=True) Finished epoch 1 / 10:]|cost 2.302576, |train: 0.080000, yal 0.103000, Lr 1.000000e-06 Finished epoch 2 / 10:|cost 2.302582, |train: 0.121000, yal 0.124000, Lr 1.000000e-06 Finished epoch 3 / 10:]|cost 2.302558, |train: 0.119000, yal 0.138000, Lr 1.000000e-06 Finished epoch 4 / 10:]|cost 2.302519, |train: 0.127000, yal 0.151000, Lr 1.000000e-06 Finished epoch 5 / 10:]|cost 2.302517, |train: 0.158000, yal 0.171000, Lr 1.000000e-06 Finished epoch 6 / 10:]|cost 2.302518, |train: 0.179000, yal 0.172000, Lr 1.000000e-06 Finished epoch 7 / 10:|cost 2.302466, |train: 0.180000, yal 0.176000, Lr 1.000000e-06 Finished epoch 8 / 10:|cost 2.302452, |train: 0.175000, yal 0.185000, Lr 1.000000e-06 Finished epoch 9 / 10:]|cost 2.302459, |train: 0.206000, yal 0.192000, Lr 1.000000e-06 Finished epoch 10 / 10} cost 2.302420] traijn: 0.190000, jval 0.192000, Lr 1.000000e-06 finished optimization. Lhe alidatiow accuracy: 0.192000 

probability of ctu stil dliffise, but thoy ave shifted Slight & Wit, in this srtouctioy Ther (ass dacsnt change so much, but wo make oleoision but choy the max. a 

model = init _two_layer_model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best model, stats = trainer.train(X train, ytrain, X_ val, y val, model, two layer net, num_epochs=10, reg=0.000001, update='sgd',sample batcheslearning= True, rate decay=1, 

model = init _two_layer_model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best model, stats = trainer.train(X train, ytrain, X_ val, y val, model, two layer net, num_epochs=10, reg=0.000001, update='sgd',sample batcheslearning= True, rate decay=1, learning rate=le6, verbose=True) 

/home/karpathy/cs231n/code/cs231n/classifiers/neuralnet.py:50: RuntimeWarning: divide by zero en countered in log data loss = -np.sum(np.log(probs[range(N), y])) / N /home/karpathy/cs231n/code/cs231n/classifiers/neural_net.py:48: RuntimeWarning: invalid value enc ountered in subtract probs = np.exp(scores - np.max(scores, axis=1, keepdims=True)) Finished epoch 1 / 10: cost nan, train: 0.091000, val 0.087000, Lr 1.000000e+06 Finished epoch 2 / 10: cost nan, train: 0.095000, val 0.087000, Lr 1.000000e+06 Finished epoch 3 / 10: cost nan, train: 0.100000, val 0.087000, Lr 1.000000e+06 

###### cast exploded 

er————--> 

model = init two layer_model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best_model, stats = trainer.train(X train, ytrain, X_ val, y_val, model, two layer net, numepochs=10, reg=0.000001, update='sgd', learningrate decay=1, sample batches = True, learning rate=3e-3, verbose=True) Finished epoch 1 / 10: cost 2.186654, train: 0.308000, val 0.306000, Ilr 3.000000e-03 Finished epoch 2 / 10: cost 2.176230, train: 0.330000, val 0.350000, lr 3.000000e-03 Finished epoch 3 / 10: cost 1.942257, train: 0.376000, val 0.352000, Ilr 3.000000e-03 Finished epoch 4 / 10: cost 1.827868, train: 0.329000, val 0.310000, Ilr 3.000000e-03 Finished epoch 5 / 10: cost inf, train: 0.128000, val 0.128000, lr 3.000000e-03 Finished epoch 6 / 10: cost inf, train: 0.144000, val 0.147000, Ilr 3.000000e-03 

#### Hyperparameter Optimization 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 75 

**April 20, 2017** 

**Cross-validation strategy coarse -> fine** cross-validation in stages 

**First stage** : only a few epochs to get rough idea of what params work **Second stage** : longer running time, finer search … (repeat as necessary) 

Tip for detecting explosions in the solver: If the cost is ever > 3 * original cost, break out early 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 76 

**April 20, 2017** 

max count = 100 for count in xrange(max count): * not uniformly Sample in reg = 10**uniform(-5, 5) 107 10° lr = 10**uniform(-3, -6)_ “<—— * instied© g'**)© g'**) g'**) trainer = ClassifierTrainer() model = init two layer_model(32*32*3,_model(32*32*3, 50, 10) # input size, hidden size, number of classes trainer = ClassifierTrainer() best model local, stats = trainer.train(X train, ytrain,train, Xval,val, y val, val, model, two layer net, numepochs=5,epochs=5, reg=reg, update='momentum', lea **r** ateningdecay=0.9,ningdecay=0.9,decay=0.9, sample batches = True, batch size = 100, learning rate=lr, verbose=False) 

|val_acc:|0.214000,|Lr:|7.231888e-06,|reg:|2.32128le-04,|(2|/ 100)|
|---|---|---|---|---|---|---|---|
|val_acc:|0.208000,|Lr:|2.11957le-06,|reg:|8.011857e+01,|(3|/ 100)|
|val_acc:|0.196000,|Ir:|1.551131le-05,|reg:|4.374936e-05,|(4|/ 100)|
|val_acc:|0.079000,|Lr:|1.753300e-05,|reg:|1.200424e+03,|(5|/ 100)|
|val acc:|0.223000,|Lr:|4.215128e-05,|reg:|4.196174e+01,|(6|/<br>100)|
|val<br>acc:|0.241000,|lr:|6.74923le-05,|reg:|4.226413e+01,|(8|/<br>100|
|~~——_+»~~||||||||
|val_ acc:<br>|0.079000,<br>|Lr:<br>|5.401602e-06,<br>|reg:<br>|1.599828e+04,<br>|(10 <br>|/ 100<br>|
|val_ acc:|0.154000,|Lr:|1.618508e-06,|reg:|4.925252e-01,|(11|/100)|

_Random Search for Hyper-Parameter Optimization_ Bergstra and Bengio, 2012 

###### Random Search vs. Grid Search 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 80 

**April 20, 2017** 

###### **Hyperparameters to play with:** - network architecture 

- learning rate, its decay schedule, update type - regularization (L2/Dropout strength) 

neural networks practitioner music = loss function 

<u>This image</u> by Paolo Guereta is licensed under CC-BY 2.0-BY 2.0BY 2.0 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 81 **April 20, 2017** 

Loss 

<mark>time</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 84 

**April 20, 2017** 

Loss 

Bad initialization a prime suspect 

<mark>time</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 6 -** 85 

**April 20, 2017** 

# assume parameter vector W and its gradient vector dW param scale = np.Llinalg.norm(W.ravel()) update = -learning rate*dW # simple SGD update update scale = np.linalg.norm(update.ravel()) W += update # the actual update print update scale / param scale # want ~le-3 

clovt wart fp update the weight te quick or 19 Slow 

Summary We looked in detail at: 

###### TLDRs 

- Activation Functions (use ReLU) 

- Data Preprocessing (images: subtract mean) 

- - Weight Initialization (use Xavier init) 

- - Batch Normalization (use) 

- Babysitting the Learning process - Hyperparameter Optimization (random sample hyperparams, in log space when appropriate) **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 88 **April 20, 2017** 

##### Next time: Training Neural Networks, Part 2 

- Parameter update schemes 

- Learning rate schedules 

- - Gradient checking 

- - Regularization (Dropout etc.) 

- - Evaluation (Ensembles etc.) 

- - Transfer learning / fine-tuning 

> **Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 6 -** 89 

**April 20, 2017**

---

## 源文件

- [cs231n_2017_lecture6.pdf](attachments/documents/AI_CNN-43c4c9904da9/cs231n_2017_lecture6.pdf)
