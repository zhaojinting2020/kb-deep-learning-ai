---
title: mock_exam
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-3d0dbea77618/mock_exam.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-3d0dbea77618/mock_exam.pdf
  title: mock_exam.pdf
---

# Machine Learning for Communications Mock Exam 

January 01, 2018 

Institute for Communications Engineering Technical University of Munich Prof. Dr. sc. techn. Gerhard Kramer 

_1 Neural Networks I_ 

## **1 Neural Networks I** 

### **1.1 Linear Regression and Cost Functions** 

Table 1 contains data points form the training set _D_ . We model the relation between the input variables _x_ and output variables _y_ as 

where _U ⇠U_ [ _−_ 1 _,_ 1] is a random variable uniformly distributed on the [ _−_ 1 _,_ 1] interval, i.e., the probability density function of _U_ is 

|_i_<br>_x_<sup>(</sup><sup>_i_)</sup><br>_y_<sup>(</sup><sup>_i_)</sup>|
|---|
|1<br>0<br>2|
|2<br>3<br>2|
|3<br>4<br>3|
|4<br>5<br>4|

Table 1: Training set _D_ . 

- (a) Derive the cost function for the model as in (1) using the maximum likelihood (ML) principle, i.e., the miniming parameters of the cost function should be the ML-parameters for the model. (Hint: you can use log likelihoods) 

- (b) Assume the following parameters _w_ = 1 _, b_ = _−_ 1. Compute the cost for these parameters. (Hint: you may find it useful to plot the model and the data points) 

- (c) Assume the following parameters _w_ =<sup><u>1</u></sup> 2<sup>_, b_= 1.Computethecostfortheseparam-</sup> eters. 

- (d) State ML-parameters for the model (1) and the training set _D_ . Justify the choice. Are the ML-parameters unique? Justify. 

Now we change the model. The relation between the input variables _x_ and output variables _y_ is now modeled as 

where _N ⇠N_ (0 _,_ 1) is a Gaussian random variable with zero mean and unit variance, i.e., the probability density function of _N_ is 

2 

~~<u>!) (ev ML -1tw6rb, atwx+b] Plyl % wb) = 0-5 Ye [A +wxtb 4+wxtb] eedant</u>~~ <u>sample</u> ~~<u>PUY» Yan Xr Xn Wb) f Pel Xi.Wrb) + PlYn |X, wb) a max POY. Yn Xi.</u>~~ <u>Xap b)</u> ~~<u>w</u> Wib~~ ~~<u>N (= 4 Vi, YeL+ Wxitb, 1+ Wxitb]</u> “que Z~~ ~~<u>~ bog, PCY Nin Wb)</u> +a~~ ~~<u>( MN</u>~~ <u>=</u> ~~<u>2 ~My</u>~~ <u>L</u> ~~<u>aly wae)</u>~~ <u>- W(y-1-wx-b)] 2)</u> ~~<u>( =~ log. -3x laos) = +9</u>~~ x 

~~+)~~ <u>oO</u> ~~Cost~~ ~~<u>W</u> ow~~ ~~<u>£ ~ (YW —b) xi = 0 =nJ</u> bot~~ ~~<u>= &</u> LoyLM~~ ~~<u>— Wi —h)=</u>~~ <u>ye</u> ~~<u>-3</u> e)~~ <u>no 1</u> ~~they ome wot. because ivaddianks awe rot~~ ~~<u>ZW</u>~~ 

_1.2 A simple Neural Network_ 

- (e) Derive the cost function for the model as in (3) using the maximum likelihood (ML) principle, i.e., the minimizing parameters of the cost function should be the MLparameters for the model. (Hint: you can use log likelihoods). 

- (f) Assume the following parameters _w_ = 1 _, b_ = _−_ 1. Compute the gradient of the cost function for these parameters. 

- (g) Are the parameters from the previous question the ML-parameters? Justify. 

### **1.2 A simple Neural Network** 

Consider neural network as in Figure 1 with the following notation: 

- _w_<sup>[</sup><sup>_k_]</sup> _ij_<sup>denotes a weight associated with a signal going from</sup><sup>_j_-th neuron in the (</sup><sup>_k−_1)-</sup> 

- th layer to the _i_ -th neuron in the _k_ -th layer. 

- _b_<sup>[</sup> _i_<sup>_k_]</sup> denotes the bias term for the _i_ -th neuron in the _k_ -th layer. 

- _a_<sup>[</sup> _i_<sup>_k_]</sup> denotes the output of the _i_ -th neuron in the _k_ -th layer. 

- _zi_<sup>[</sup><sup>_k_]</sup> is the (total) input to the _i_ -th neuron in the _k_ -th layer. 

Figure 1: Neural Network for Exercise 2. 

We use the network for a regression problem with the training set _D_ from Table 1 (from the previous exercise). We model _y_ as being an output of the network with input _x_ . We set the neuron in the output layer to use the identity activation function, i.e., _g_ ( _z_ ) = _z_ . Other neurons use the rectified linear unit (ReLU) activation function, i.e., 

3 

_1.2 A simple Neural Network_ 

_g_ ( _z_ ) = max( _z,_ 0). Assume the following parameters: 

To measure the performance of the network we use the quadratic cost function 

where _C_<sup>(</sup><sup>_i_)</sup> = _c_ ( _x_<sup>(</sup><sup>_i_)</sup> _, y_<sup>(</sup><sup>_i_)</sup> ) is the cost for the _i_ -th training sample. 

- (a) Compute the cost for the 3rd training sample, i.e., the cost _C_<sup>(3)</sup> for the pair ( _x_<sup>(3)</sup> _, y_<sup>(3)</sup> ). 

- (b) Compute the gradients of the cost for the 3rd training sample with respect to the network parameters, i.e., compute _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(3)</sup> [1]<sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> **W**<sup>(3)[2]</sup><sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(3)</sup> [3]<sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_ [1]<sup>(3)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_ [2]<sup>(3)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_<sup>[3](3).(Useback</sup> propagation) 

- (c) Assume that the gradients for other training samples are zero, i.e., all the gradients from the precious point are zero for _i_ = 1 _,_ 2 _,_ 4. Are the network parameters optimal in this case? Justify. 

- (d) Assume that the cost for other training samples is zero, i.e, _C_<sup>(</sup><sup>_i_)</sup> = 0 for _i_ = 1 _,_ 2 _,_ 4. Are the parameters optimal in this case? Justify. 

_2 Universal Source Coding_ 

## **2 Universal Source Coding** 

1. Can _S_ = _{_ 0 _,_ 10 _,_ 11 _}_ be a set of valid contexts for a source model? Justify your answer. 

2. We want to find the codeword for a block of 9 symbols from a binary source with _X_ = _{_ 0 _,_ 1 _}_ using Context Tree Weighting algorithm. The context tree with the quadruples ( _as, bs, P_<sup>_e_</sup> ( _x_ [ _s_ ]) _, Ps_<sup>_w_(</sup><sup>_x_</sup> [ _s_ ]<sup>)afterparsingfirst5sourcesymbols</sup> is shown below. If the next 4 symbols are _x_ 6 _· · · x_ 9 = 0110 with the past 3 symbols _x_ 3 _· · · x_ 5 = 101 then calculate the corresponding quadruples after encoding the 9 bit source block. Mention the final probability of the source block fed to the arithmetic encoder and calculate the codeword length for this block when using arithmetic coding. 

Figure 2: Context tree after encoding 5 bits. 

3. If the actual source model is _S_ = _{_ 0 _,_ 01 _,_ 011 _,_ 111 _}_ then what is the loss of not knowing the actual source model when using the context tree in part ( _b_ ). 

5 

_3 Inference in Graphical Models_ 

## **3 Inference in Graphical Models** 

Consider random variables _Xi 2 {_ 0 _,_ 1 _}, i_ = 1 _,_ 2 _,_ 3 _,_ 4 with constraints 

where **xor** and **or** are defined by 

Let _Yi_ denote the noisy version of _Xi_ . From _Yi, i_ = 1 _,_ 2 _,_ 3 _,_ 4 we have 

||_pXi|Yi_(0_|yi_)|_pXi|Yi_(1_|yi_)|
|---|---|---|
|_i_= 1|0_._2|0_._8|
|_i_= 2|0_._9|0_._1|
|_i_= 3|0_._3|0_._7|
|_i_= 4|0_._5|0_._5|

1. Compute the marginal distribution of _Xi, i_ = 1 _,_ 2 _,_ 3 _,_ 4 given **_Y_** = _Y_ 1 _Y_ 2 _Y_ 3 _Y_ 4: _pXi|_ **_Y_** ( _xi|_ **_y_** ) by using brute forcing (compute the joint distribution of all possibilities). 

2. Draw a factor graph of the given problem. 

3. Describe all factor nodes with (conditional) distributions or functions. 

4. Is this factor graph cycle-free or not? 

5. Compute the marginal distribution of _Xi, i_ = 1 _,_ 2 _,_ 3 _,_ 4 after one iteration: _p_ ˜ _Xi|_ **_Y_** ( _xi|_ **_y_** ). (every node should be updated once) 

_4 Expectation Maximization_ 

## **4 Expectation Maximization** 

Consider a transceiver setup with on-o↵keying (OOK) where the sent symbol experiences noise that is exponentially distributed with parameter _λ_ . The constellation is depicted in the following picture. 

Both points are used with di↵erent probability, i.e., 

The resulting channel model is 

where the PDF of the noise _N_ is 

The receiver wants to estimate the parameter vector **_✓_** = ( _x_ 1 _, p_ 1 _, λ_ )<sup>T</sup> based on a number of _N_ received samples _y_ 1 _, . . . , yN_ . 

1. Formulate the log-likelihood function for the estimation problem. 

2. What is the latent variable in this example? Why? 

3. Formulate the E-step of the EM algorithm, i.e., provide an expression for the posterior of the latent variable for the _t_ -th iteration based on the previous parameter estimate **_✓_**<sup>(</sup><sup>_t−_1)</sup> . 

4. Derive expressions for the update of the parameter values in the M-step. 

5. For a similar problem, the M-step requires to calculate an interim expression of the form: 

where _fj_ and _gj_ are the components of two vectors **_f_** and **_g_** of length _K_ . Assume that you are given _y_ 1 _, . . . , yN_ , _f_ 1 _, . . . , fK_ and _g_ 1 _, . . . , gK_ as column vectors ( `y` , `f` , `g` ) of the `numpy` type `numpy.ndarray` Formulate the required single line of Python code to calculate the vector `z` . What `numpy` feature do you use for this? 

7 

_5 Dimensionality Reduction_ _<u>(PCA)</u>_ 

## **5 Dimensionality Reduction (PCA)** 

1. You are given the following data sets with realizations of the random vector _<u>X</u>_ = ( _<u>X</u>_ <u>(1)</u> _, X_ <u>(2)):</u> 

Determine a reasonable estimate of the sample covariance matrix for each case. 

2. Show that the sample covariance matrix **_C_** _X 2_ R<sup>_M⇥M_</sup> of a data set is symmetric T 

and positive semi-definite, i.e., **_C_** _X_<sup>T=</sup><sup>**_C_**</sup><sup>_X_and</sup><sup>_<u>z</u>_</sup> **_C_** _X_ _<u>z</u> ≥_ 0 for all _<u>z</u> 2_ R<sup>_M_</sup> . 

3. For probabilistic PCA, we consider a Gaussian model 

where _<u>Z</u> ⇠N_ (0 _, IK_ ), _<u>m</u> 2_ R<sup>_M_</sup> , **Q** _2_ R<sup>_M⇥K_</sup> and _<u>V</u> ⇠N_ (0 _, σ_<sup>2</sup> _IM_ ). 

In the following, we will consider a mixture of two probabilistic PCA models with parameters _<u>✓</u>_ 1<sup>and</sup><sup>_<u>✓</u>_</sup> ~~2~~<sup>where</sup><sup>_<u>✓</u>_</sup> _~~i~~_<sup>= (</sup><sup>_Qi, mi, σ_</sup> _i_<sup>2)and</sup><sup>_P_(</sup><sup>_<u>✓</u>_</sup> 1<sup>) = 1</sup><sup>_/_3and</sup><sup>_P_(</sup><sup>_<u>✓</u>_</sup> 2<sup>) = 2</sup><sup>_/_3.</sup> 

- a) Compute the mean _µi_ and covariance matrix **C** _X,i_ of the two models and sketch the contour lines of the probability density function of both models. 

- b) Use the Woodbury identity to calculate the inverse of **C** _X,_ 1 and **C** _X,_ 2. 

- c) Consider a data point _<u>x</u>_ = [0 _,_ 0]<sup>T</sup> . Is _<u>x</u>_ assigned to the first or second model according to a maximum likelihood criterion? 

- d) Use the model chosen in 3. to reduce the dimension of _<u>x</u>_ to one dimension. 

8 

~~<u>A Cov [ (*)] 5 hci boy</u>~~ <u>(xi,</u> ~~<u>2) |</u> Cov (%)m) = Cov [X>,X2)~~ ~~<u>- ieEL</u> (xiMY) -m)e-alEL~~ ~~<u>bn)EL GoMod]my! | im\@.[*o +4| mgd (2) E3 ql4 Img"d ?@): Lyoo 50</u> x=pe~~ ~~<u>(8 web wT 70, Te AEE AAME E1118 unm</u> T~~ <u>;</u> ~~distri but i~~ <u>></u> ~~AB Ho~~ ~~<u>Vav 0x = Var Oe) >$ 2T (, 2 =-/—Ny 2 ZzT(x[0 -Mi) (x MVR = 9 (te 472) ((w-mre )</u> =~~ ~~<u>[</u> (ways~~ ~~<u>|</u>~~ 2 ~~70~~ ~~<u>fr</u> a~~ ~~<u>z Semi — positive definit</u>~~

---

## 源文件

- [mock_exam.pdf](attachments/documents/AI_Machine-Learning-in-Communication-3d0dbea77618/mock_exam.pdf)
