---
title: script
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-6e716d50d0b9/script.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-6e716d50d0b9/script.pdf
  title: script.pdf
---

# Machine Learning for Communications 

Prof. Gerhard Kramer Lars Palzer, M.Sc. Marcin Pikus, M.Sc. Tobias Prinz, M.Sc. Fabian Steiner, M.Sc. Peihong Yuan, M.Sc. 

December 4, 2019 

Institute for Communications Engineering Technical University of Munich Prof. Gerhard Kramer 

The following lecture notes are part of the course “Machine Learning for Communications” o↵ered by the Institute for Communications Engineering at the Technical University of Munich. All content is subject to copyright restrictions. If you are planning to use any of the material, please contact Prof. Gerhard Kramer ( `gerhard.kramer@tum.de` ). 

2 

## **Contents** 

|**1**<br>**Sup**|**ervised **|**Learning: Neural Networks**<br>**5**|
|---|---|---|
|1.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>5|
|1.2|Single|Neuron . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>5|
||1.2.1|Linear Regression with a Single Neuron . . . . . . . . . . . . . . .<br>6|
||1.2.2|Binary Classifcation with a Single Neuron<br>. . . . . . . . . . . . .<br>12|
|1.3|Neura|l Networks<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>15|
|1.4|Backp|ropagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>18|
|**2**<br>**Pro**|**babilisti**|**c Graphical Models**<br>**23**|
|2.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>23|
|2.2|Factor|Graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>23|
|2.3|Sum-P|roduct Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>24|
||2.3.1|Separation and Marginalization . . . . . . . . . . . . . . . . . . . .<br>25|
||2.3.2|Factor Nodes to Variable Nodes . . . . . . . . . . . . . . . . . . . .<br>26|
||2.3.3|Variable Nodes to Factor Nodes . . . . . . . . . . . . . . . . . . . .<br>26|
||2.3.4|Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>27|
||2.3.5|Max-Product Rules<br>. . . . . . . . . . . . . . . . . . . . . . . . . .<br>28|
|2.4|Belief|Propagation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>29|
||2.4.1|Example: Single Parity Check Code of Length Three . . . . . . . .<br>29|
||2.4.2|Example: Single Parity Check Code of Length Four<br>. . . . . . . .<br>32|
|**3**<br>**App**|**roxima**|**te Message Passing**<br>**37**|
|3.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>37|
|3.2|Messa|ge Passing Derivation . . . . . . . . . . . . . . . . . . . . . . . . . .<br>38|
||3.2.1|Simplifying the Messages<br>. . . . . . . . . . . . . . . . . . . . . . .<br>40|
||3.2.2|Reducing the Number of Messages . . . . . . . . . . . . . . . . . .<br>43|
||3.2.3|The AMP Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . .<br>45|
|3.3|From|AMP to GAMP . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>45|
|3.4|Applic|ation: Massive MIMO<br>. . . . . . . . . . . . . . . . . . . . . . . . .<br>47|
||3.4.1|System Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>47|
|**4**<br>**Uns**|**upervis**|**ed Learning: Expectation Maximization**<br>**51**|
|4.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>51|
|4.2|Prelim|inaries<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>51|
|4.3|Maxim|um-Likelihood Estimation . . . . . . . . . . . . . . . . . . . . . . .<br>51|
|4.4|Expec|tation Maximization (EM) Algorithm . . . . . . . . . . . . . . . . .<br>53|
||4.4.1|Evidence Lower Bound (ELBO)<br>. . . . . . . . . . . . . . . . . . .<br>53|

_Contents_ 

||4.4.2|Algorithmic Formulation . . . . . . . . . . . . . . . . . . . . . .|. .<br>55|
|---|---|---|---|
||4.4.3|Convergence Analysis<br>. . . . . . . . . . . . . . . . . . . . . . .|. .<br>56|
|4.5|K-Me|ans . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>56|
|4.6|Blind|Parameter Estimation for Probabilistic Amplitude Shaping . . .|. .<br>57|
||4.6.1|Decoder Soft-Information . . . . . . . . . . . . . . . . . . . . .|. .<br>58|
||4.6.2|Imposed Model . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>58|
||4.6.3|EM Formulation . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>58|
|**5**<br>**Sam**|**pling T**|**echniques**|**61**|
|5.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>61|
|5.2|Impor|tance Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>63|
|5.3|Reject|ion Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>66|
|5.4|Marko|v Chain Monte Carlo (MCMC) Sampling . . . . . . . . . . . . .|. .<br>67|
||5.4.1|Markov Chains . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>67|
||5.4.2|Metropolis-Hastings Sampling . . . . . . . . . . . . . . . . . . .|. .<br>69|
||5.4.3|Gibbs Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>69|
||5.4.4|Example: Conditional Mean . . . . . . . . . . . . . . . . . . . .|. .<br>70|
|**6**<br>**Dim**|**ension**|**ality Reduction**|**74**|
|6.1|Introd|uction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>74|
|6.2|Princi|pal Component Analysis<br>. . . . . . . . . . . . . . . . . . . . . .|. .<br>75|
||6.2.1|Derivation of PCA . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>76|
||6.2.2|Singular Value Decomposition . . . . . . . . . . . . . . . . . . .|. .<br>77|
||6.2.3|Pre-Processing . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>78|
||6.2.4|Choosing the Dimension _K_<br>. . . . . . . . . . . . . . . . . . . .|. .<br>78|
||6.2.5|Algorithm Description . . . . . . . . . . . . . . . . . . . . . . .|. .<br>79|
|6.3|Proba|bilistic PCA<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>81|
||6.3.1|Maximum Likelihood Estimation . . . . . . . . . . . . . . . . .|. .<br>81|
||6.3.2|Dimensionality Reduction with PPCA . . . . . . . . . . . . . .|. .<br>84|
||6.3.3|Relationship Between PCA and PPCA . . . . . . . . . . . . . .|. .<br>84|
|6.4|Expec|tation Maximization . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>85|
||6.4.1|EM for Probabilistic PCA . . . . . . . . . . . . . . . . . . . . .|. .<br>85|
||6.4.2|EM for PCA<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|. .<br>88|

4 

## **1 Supervised Learning: Neural Networks** 

### **1.1 Introduction** 

Neural networks (NNs) and deep learning are ideas that were trendy several times over the past 70 years. In the 1940s-1960s they were part of a larger area called cybernetics; in the 1980s-1990s they were part of an area called connectionism; and since 2006 they appeared as neural networks and deep learning. The third wave of neural networks research began with a breakthrough in 2006: Geo↵rey Hinton developed an efficient method to train _deep-belief_ networks, and this improved the performance of neural networks and resulted in many impressive results. 

In this part of the lecture, we will study _supervised learning_ . We are given a set of labeled training data, i.e., a set of pairs (input data, desired output data), and we would like the NN to approximate a high-dimensional function. The NN is optimized by using the training data, and then the network is used to form predictions based on new data. 

### **1.2 Single Neuron** 

We begin by studying a single neuron that is capable of learning. A single neuron is a processing unit with _M_ real-valued inputs _x_ 1 _, x_ 2 _, . . . , xM_ and one real-valued output _a_ . Each input is associated with a real-valued weight _wk_ , _k_ = 1 _, . . . , M_ . See Figure 1.1. 

The neuron first computes an _activation z_ by weighting the inputs and adding a bias term _b_ : 

where _<u>x</u>_ = [ _x_ 1 _, . . . , xM_ ]<sup>_T_</sup> and _<u>w</u>_ = [ _w_ 1 _, . . . , wM_ ]<sup>_T_</sup> are column vectors (the superscript _T_ denotes transpose). Next, an _activation function g_ : _R ! R_ , where _R_ denotes the set of real numbers, is applied to obtain the neuron’s output 

The neuron’s behavior is thus defined by _<u><mark>w</mark></u>_ <u><mark>,</mark></u> _<mark>b</mark>_ <mark>, and</mark> _<mark>g</mark>_ <mark>(</mark> _<mark>·</mark>_ <mark>).</mark> The values of _<u>w</u>_ and _b_ are usually found by optimization (or training, or learning) and _g_ ( _·_ ) is chosen based on the application. The most common activation functions are the following, see Figure 1.2. 

- Identity: _g_ ( _z_ ) = _z_ 

_1.2.1 Linear Regression with a Single Neuron_ 

Figure 1.1: A single neuron with four inputs. 

- Tanh: _g_ ( _z_ ) = tanh( _z_ ) 

- Rectified linear unit (ReLU): _g_ ( _z_ ) = max( _z,_ 0) _._ 

#### **1.2.1 Linear Regression with a Single Neuron** 

One task a neuron can perform is linear regression. Regression is a task of predicting a scalar real-valued output _y_ given an input vector _<u>x</u>_ <u>.</u> One can think of regression as discovering a model, or a relation that maps _<u>x</u>_ to _y_ . 

To discover the relation, we are given a training set _D_ that consists of _N inputs {x_ ~~1~~<sup>_, . . . , x_</sup> _~~N~~_<sup>_}_and</sup><sup>_Noutputs{y_1</sup><sup>_, . . . , yN}_,wheretheinput</sup><sup>_<u>x</u>_</sup> _i_<sup>gavetheoutput</sup><sup>_yi_for</sup> all _i_ . Suppose each _<u>x</u>_ _~~i~~_<sup>has</sup><sup>_M_entriesandeach</sup><sup>_yi_isascalar,real-valuedoutputfor</sup> _i_ = 1 _,_ 2 _, . . . , N_ . 

In linear regression, we use a linear model (or, more precisely, an _affine_ model), i.e., we model _<u>x</u>_ and _y_ as being related via 

Note that the “true” relation between _<u>x</u>_ and _y_ may be non-linear. However, linear models are useful because they are simple and often provide good results. The <mark>linear model</mark> (1.3) corresponds to <mark>a single neuron</mark> with the identity activation function _g_ ( _z_ ) = _z_ . In this case, the neuron output for the _i_ -th input from _D_ is 

We are interested in finding _<u><mark>w</mark></u>_ <mark>= [</mark> _<mark>w</mark>_ 1 _<mark>, . . . , w</mark> M_ <mark>]</mark><sup>_T_</sup> and _b_ such that the outputs _ai_ are close to the outputs _yi_ from the training set for _i_ = 1 _, . . . , N_ . If we are successful, then we say that the neuron has _learned_ to generate outputs according to the data from _D_ . 

6 

_1.2.1 Linear Regression with a Single Neuron_ 

Figure 1.2: Activation functions. 

7 

_1.2.1 Linear Regression with a Single Neuron_ 

To assess performance, we introduce a cost function _C_ ( _<u>x, y, w, b</u>_ ). The choice of cost function depends on the application. Preferably, the cost should be continuous in _<u>w</u>_ and _b_ , and it should separate into a sum of per-sample costs, so that 

where _Ci_ = _c_ ( _<u>x</u>_ _~~i~~_<sup>_, yi, w_</sup> _<u>, b</u>_ ) is the cost for the _i_ -th training sample. Here, we consider a quadratic cost function where 

Observe that _C ⇡_ 0 implies _yi ⇡ ai_ for each _i_ = 1 _, . . . , N_ . This means that if _C_ is small, our model explains the data from _D_ correctly. <mark>The goal of learning is to accurately predict data outside of</mark> _<mark>D</mark>_ <mark>.</mark> In other words, the goal of our optimization procedure is as follows: given the training set _D_ , <mark>fnd</mark> _<u><mark>w</mark></u>_ <mark>and</mark> _<mark>b</mark>_ such that _C_ is small for most, or perhaps even all, possible data pairs. The minimization is often done by _gradient descent_ algorithms.<sup>1</sup> 

##### **Gradient Descent** 

Gradient descent is an optimization algorithm that aims to minimize a cost function. Pictorially, we <mark>start at</mark> a random point in the parameter space, i.e., at <mark>randomly chosen</mark> _<u><mark>w</mark></u>_ <mark>and</mark> _<mark>b</mark>_ <mark>.</mark> At this point, <u>we look for a direction where the cost function decreases the most.</u> Next, <u>we take a small step in this direction</u><sup>2</sup> , moving to a new point in the parameter space. At this point, we again look for a direction where the cost function decreases the most. We take another small step, and we continue to iterate. 

The gradient is a row vector that points in the direction where the cost function grows the most. The gradient of _C_ with respect to _<u>w</u>_ is the vector of partial derivatives 

The reason that we define the gradient <u>as a</u> _<u>row</u>_ <u>vector rather</u> than a column vector is to use a particular _Jacobian formulation_ for matrix calculus, as explained below. The gradient is a function of _<u>w</u>_ in general. We denote the function at point _<u>w</u>_ ~~0~~<sup>by</sup> 

> 1The optimal _<u>w</u>_ and _b_ can be found analytically for <u>the</u> linear regression problem. We will describe the gradient descent algorithm because it is also useful when an analytical solution is not available (when an <u>activation function other than the identity</u> is used or <u>when the cost function is not quadratic</u> as in 1.6.), e.g., for <u>logistic regression</u> described in Section 1.2.2. 

> 2 E↵ectively, we linearize the function and take the step where this linearization decreases the most. 

8 

_1.2.1 Linear Regression with a Single Neuron_ 

Now consider the gradient with respect to _<u>w</u>_ <u>.</u> If we take a small step _<u>✏</u>_ = [ _✏_ 1 _, . . . , ✏M_ ]<sup>_T_</sup> from _<u>w</u>_ to _<u>w</u>_ + _<u>✏</u>_ <u>,</u> then the cost changes as<sup>3</sup> 

If we choose _<u>✏T</u>_ = _−β_<sup>_@C_</sup> _@w_<sup><u>(</u></sup><sup>_<u>w</u>_</sup> <u>)</u> with _β >_ 0 (i.e., we take a step in the direction opposite to the gradient) then we have 

which is negative. A similar approach can be used for the gradient with respect to _b_ . For a fixed step size _k✏k_ , choosing the step as above results in the maximal decrease of the right-hand side of (1.10). This can be seen by applying the Cauchy-Schwarz inequality to (1.9). The constant _β >_ 0 is called the _<mark>learning rate</mark>_ <mark>,</mark> and it specifies the step size of each iteration of the gradient descent algorithm. There is a trade-o↵in choosing _β_ : small _β_ results in small steps and slow convergence (gradient descent must perform many steps to find the optimum). For large _β_ , the approximation (1.9) may not be accurate because the derivative is valid only locally. The cost may even increase, i.e., ∆ _C >_ 0, or the algorithm may oscillate around the solution. In practice, _β_ is often chosen by trial and error. 

Summarizing, the gradient descent algorithm can be represented as 

1. Choose _<u>w</u>_ and _b_ , perhaps randomly. 

2. Repeat until convergence: 

It remains to compute the gradients. We have (see (1.5)) 

and 

> 3In this step we actually linearize _C_ ( _<u>w</u>_ ) in _<u>w</u>_ as _C_ ( _<u>w</u>_ + _<u>✏</u>_ <u>)</u> _⇡ C_ ( _<u>w</u>_ <u>) +</u><sup>_@C_</sup> _@_<sup><u>(</u></sup> _<u>w</u>_<sup>_<u>w</u>_</sup> <u>)</u> _<u>✏</u>_ <u>.</u> 

_1.2.1 Linear Regression with a Single Neuron_ 

Figure 1.3: Gradient descent steps with an appropriately chosen learning rate(blue) and with a too large learning rate (red). 

Finally, we stack the<sup>_@C_</sup> _@w_<sup><u>(</u></sup><sup>_w_</sup> _k_<sup>_<u>k</u>_</sup><sup><u>)</u></sup><sup>_, k_= 1</sup><sup>_, . . . , M_,togetthecolumnvector</sup> 

Similar computations for _b_ (left as an exercise) give 

_Prediction:_ Suppose gradient descent resulted in the parameters _<u>w⇤</u>_ and _b⇤_ . How can we predict _y_ for a new _<u>x</u>_ ~~n~~ ew<sup>notcontainedin</sup><sup>_D_?Ourmodelclassis</sup><sup>_y_=</sup><sup>_a_=</sup><sup>_<u>w</u>_</sup> _T_ _<u>x</u>_ + _b_ , so the prediction is _y_ new = _<u>w⇤T x</u>_ ~~n~~ ew<sup>+</sup><sup>_b⇤_.</sup> 

##### **Probabilistic Interpretation** 

One motivation for choosing the quadratic cost function is based on _Maximum Likelihood_ (ML) estimation of the neuron parameters _<u>w</u>_ and _b_ . The ML principle can be described as follows. 

Recall the training set 

We model the training set _D_ to be generated by some density with parameters _<u>✓</u>_ <u>.</u> The ML estimator then puts out the _<u>✓</u>_ which gives the highest probability of our data, i.e., <u>we model the data as being generated according to the parameters</u> 

10 

_1.2.1 Linear Regression with a Single Neuron_ 

Now suppose we choose the class of models with _<u>✓</u>_ = [ _<u>w, b</u>_ ] and 

where _Ei_ is an error. The error could include noise and modeling mismatch, e.g., <u>there might be hidden variables that we are not considering.</u> We further model _✏i_ as a <mark>Gaussian random variable with zero mean and variance</mark> _<mark>σ</mark>_<sup><mark>2</mark></sup> , i.e., we have the probability density function 

and we write this as _Ei ⇠N_ (0 _, σ_<sup>2</sup> ). We thus have _Yi ⇠N_ ( _<u>wT x</u>_ _~~i~~_<sup>+</sup><sup>_b, σ_2)andthe</sup> conditional probability density function 

If the _<u>E</u>_ <u>,</u> _<u>i</u>_ = 1 _<u>,</u>_ <u>2</u> _<u>, . . . , N</u>_ <u>, are statistically independent, then we have the joint conditional</u> probability density function 

If we assume that all _<u>x</u>_ _~~<u>i</u>~~_<sup>havethesame</sup><sup><u>probability</u>densityor</sup><sup><u>probability</u>mass,thenthe</sup> ML estimator chooses _<u>w, b</u>_ so that the training data likelihood is maximized. This can be done by maximizing (1.15): 

where the last step follows by (1.5)–(1.6). Thus, <u>minimizing the quadratic cost (1.6) is equivalent to fnding the ML estimate of</u> _<u>w, b</u>_ <u>for the assumed model.</u> 

The model can be extended to non-uniform _<u>xi</u>_<sup>and to nonlinear functions.For instance,</sup> we may wish to extend the model by non-linear terms. For the <mark>one-dimensional</mark> case, we could introduce the model 

instead of using (1.13). 

11 

~~Linewy regression Logistic repre sion + Gaussian model of Y givenX~~ <u>‘</u> ~~Bormuli model. of Given PUYLx, wb) eee Ply lx, b= orgy?~~ <u>= sig (w4b)” [I-</u> ~~<u>sews) |</u> + dpunctratic cast fortion~~ 5 ~~t=~~ ~~<u>¢ Cros emtroPy wst function = (yi - ai) ust =S-4 bay (ai) — C-ys) bg (1-a:)</u> if Yeh! poyle) =~~ 

_1.2.2 Binary Classification with a Single Neuron_ 

##### **Cost Function and Gradient Descent** 

According to the ML principle, we would like to find parameters _<u>w, b</u>_ that maximize the likelihood of _yi_ given _<u>x</u>_ _~~i~~_<sup>for</sup><sup>_i_= 1</sup><sup>_, . . . , N_.Recallthatthetrainingset</sup> 

For each input-output pair from _D_ we can write (see (1.16)) 

where for brevity we introduced _ai_ = _<u>g</u>_ <u>(</u> _<u>wT x</u>_ _~~<u>i</u>~~_<sup><u>+</u></sup><sup>_b_</sup><sup><u>)</u>todenoteaneuronoutput.</sup> The formula can be presented in the compact form as 

The probability of the data _D_ is 

According to the ML principle we wish to compute 

In the third line we applied the logarithm. Note that the dependence of the cost on _<u>w, b</u>_ is via _ai_ = _<u>g</u>_ <u>(</u> _<u>wT x</u>_ _~~<u>i</u>~~_<sup><u>+</u></sup><sup>_b_</sup><sup><u>)</u>for</sup><sup>_i_=1</sup><sup>_, . . . , N_.Equation(1.24)definestheper-samplecost</sup> and (1.25) defines the total cost used in logistic regression. This loss function is often called a _<u>cross-entropy cost function</u>_ <u>.</u> For two binary probability distributions _P_ and _Q_ , the cross-entropy is defined as 

The summand in (1.24) is the cross entropy between a distribution of the correct output (given in the training data) _P_ such that _P_ (0) = _yi_ and a distribution of the output 

13 

_1.2.2 Binary Classification with a Single Neuron_ 

generated by our binary classifier _Q_ such that _Q_ (0) = _ai_ . Note that _yi 2 {_ 0 _,_ 1 _}_ and the distribution _P_ has all probability mass in one point. The task of minimizing the crossentropy cost function corresponds to minimizing the di↵erence between the outputs from the training set and the output generated by a neuron. Note that the cross-entropy is minimized when _P_ = _Q_ . 

The values _<u>w</u>_ ~~M~~ L<sup>_, b_MLcanbefoundbyminimizingthecostfunction(1.25).Wewill</sup> use the gradient descent algorithm, and so we need to compute the gradient of the cost function with respect to _<u>w</u>_ and _b_ . Recall that _zi_ = _<u>wT x</u>_ _~~i~~_<sup>+</sup><sup>_b_.Usingthesameapproach</sup> as before, and assuming natural logarithms, we compute 

The partial derivatives are easy to compute and left as an exercise to the reader. The second partial derivative shows an interesting property of the sigmoid function (1.19), namely that _g_<sup>_0_</sup> ( _z_ ) = _g_ ( _z_ )(1 _− g_ ( _z_ )). As in the case of linear regression, we sum the derivatives over all training pairs _{_ ( _<u>xi</u>_<sup>_, yi_)</sup><sup>_}N_</sup> _i_ =1<sup>toobtainthepartialderivatives</sup> _T @C_ <u>(</u> _wk_ <u>)</u> _@C_ <u>(</u> _<u>w</u>_ <u>)</u> _@wk_<sup>_, k_= 1</sup><sup>_, . . . , M_and then stack the partial derivatives to obtain a vector</sup> ⇣ _@w_ ⌘ . This results in a vector equation 

Similar computations for _b_ (left as an exercise) give 

Surprisingly, the update rules are the same as in the linear regression. In fact, <u>both the linear regression and logistic regression models belong to a class of generalized linear models (GLMs) that have the same update rules,</u> and that share many other properties. <u>GLMs are convex problems, i.e., gradient descent can fnd the global optimum if the step size is chosen appropriately.</u> <mark>A major limitation of GLMs is the linear dependence of the parameter of interest (or its function) and the input variable</mark> _<u><mark>x</mark></u>_ <mark>.</mark> <u>One can incorporate nonlinear functions by mapping</u> _<u>x ! φ</u>_ <u>(</u> _<u>x</u>_ <u>) and then using a linear model</u> _<u>wT φ</u>_ <u>(</u> _<u>x</u>_ <u>). However this approach requires to fnd an apporpriate function</u> _<u>φ</u>_ <u>manually for each application.</u> This limitation can be overcome to some extent by NNs. 

_Prediction:_ Suppose we have found optimal _<u>w⇤</u>_ and _b⇤_ for binary classification. We are interested in classifying a sample _<u>x</u>_ new<sup>not included in</sup><sup>_D_.The neuron output returns</sup> _a_ ( _<u>x</u>_ ~~n~~ ew<sup>) =</sup><sup>_P_</sup> _Y |X_ (1 _|x_ ~~n~~ ew<sup>).Thereforeweshouldpredict</sup> 

~~how +o avoid to complex woth when~~ <u>P is</u> ~~setto be layge? 0 Wai ght olecay~~ ~~<u>wost =F t- Yi lag (ar) - (Fo) lag (1a) + 2</u>~~ <u>||</u> ~~<u>will”</u> not penalize Wo~~ <u>-</u> ~~dst| — dast~~ . <u>Oa oe</u> ~~<u>OW le 3 Ai V2 Ow.</u> 0~~ ~~<u>= 5 (ai- 4)</u>~~ <u>Xi</u> ~~<u>+ law</u> DA Ur~~ 

_1.3 Neural Networks_ 

<u>This means that the plane</u> _<u>w⇤T x</u>_ <u>+</u> _<u>b</u>_<sup>_⇤_</sup> = 0 <u>specifes a decision boundary.</u> 

### **1.3 Neural Networks** 

Figure 1.4: Two-layer feed-forward NN with 2 input nodes in the input layer, 3 hidden nodes in the hidden layer, and 1 output node in the output layer. 

NNs are computing structures built from several neurons. We will consider only _feed-forward_ NNs that are built in layers. The message flow in a feed-forward NN is one-directional, i.e., from “lower” layers to “higher” layers. A neuron in the _l_ -th layer receives messages from neurons in the ( _l −_ 1)-th layer only, as opposed to _recurrent_ NNs where messages can be fed back to lower layers. An example of a feed-forward NN is shown in Fig. 1.4. Feed-forward NNs are used in many commercial applications since many years, e.g., by the United States Postal Service for handwritten postal addresses interpretation since 1997, and by the self-driving car ALVINN from 1989. 

As before, each neuron has input weights _w_ 1 _, . . . , wM_ , a bias term _b_ , and the activation function _g_ ( _·_ ). The notation is as follows. 

- _w_<sup>[</sup><sup>_k_]</sup> _ij_<sup>denotes a weight associated with a signal going from</sup><sup>_j_-th neuron in the (</sup><sup>_k−_1)-</sup> 

- th layer to the _i_ -th neuron in the _k_ -th layer. 

- _b_<sup>[</sup> _i_<sup>_k_]</sup> denotes the bias term for the _i_ -th neuron in the _k_ -th layer. 

- _a_<sup>[</sup> _i_<sup>_k_]</sup> denotes the output of the _i_ -th neuron in the _k_ -th layer. 

- _zi_<sup>[</sup><sup>_k_]</sup> is the (total) input to the _i_ -th neuron in the _k_ -th layer. 

- _gi_<sup>[</sup><sup>_k_]</sup> is the activation function of the _i_ -th neuron in the _k_ -th layer. 

15 

_1.3 Neural Networks_ 

Using this notation, the output of the network in Fig. 1.4 is 

This notation might be confusing and is not easy to work with. Therefore we will introduce simplified notation using vectors and matrices. First, the input layer (also called layer 0) is special, with inputs that are entries of the input vector _<u>x</u>_ . Neurons in the input layer use the identity activation function, have no bias term, and they only forward the received inputs to the neurons in the next layer. We write 

We stack all the outputs/inputs in a vector. Then, the function _g_ is applied element-wise to each entry in the vector. We obtain the equation 

The input layer is introduced artificially to allow a consistent notation. Therefore it is also not considered in the _depth_ of the network. <u>The depth of the network is the number of NN layers, excluding the input layer.</u> For instance, the NN in Fig. 1.4 has two layers. Now consider the _hidden_ layer, which refers to a layer whose output cannot be observed directly. The NN on Fig. 1.4 has one hidden layer. We first compute the activations of each neuron in the layer as 

The three equations can be brought into matrix-vector notation to read as 

<u>We choose all activation functions in the hidden layer to be the</u> _<u>same</u>_ <u>function</u> _<u>g</u>_ <u>(</u> _·_ <u>).</u> Then, we can write the output from the hidden layer as 

16 

_1.3 Neural Networks_ 

The last layer is called the output layer. Applying the same notation to the output layer gives us the output of the NN 

This equation shows that the NN is a composition of vector functions. <u>In practice, the activation function used in the last layer may be di↵erent from the activation functions used in the hidden layers. The hidden layers usually use non-linear functions, e.g., sigmoid, tanh, and ReLU. The activation function in the output layer depends on the application, e.g.,</u> we may use the identity activation function for a regression problem or <u>the sigmoid activation function for binary classifcation problem.</u> 

Summarizing, we state the _forward propagation_ algorithm for computing the output of a feed-forward NN with _L_ layers. The 1<sup>st</sup> line is for initialization. The 2<sup>nd</sup> and 3<sup>rd</sup> lines are executed iteratively for subsequent layers to obtain the NN’s output _<u>a</u>_ [ _L_ ]. 

##### **Cost Functions** 

The cost functions for NNs can be derived using the Maximum Likelihood (ML) principle just like in the previous case with single neurons. Instead of modeling the relation between _<u>x</u>_ and _y_ as linear, we model _y_ as being an output of a NN for the input _<u>x</u>_ . Assume a regression problem with the training data set 

Suppose we choose the class of model 

where _Ei ⇠N_ (0 _, σ_<sup>2</sup> ) is an error just as in the case of linear regression (1.13) and _a_<sup>[</sup><sup>_L_]</sup> ( _<u>x</u>_ _~~i~~_<sup>)</sup> is the output of a NN with _L_ layers <u>and a single neuron in the output layer.</u> If the _✏i_ , _i_ = 1 _,_ 2 _, . . . , N_ , are statistically independent, then we have the joint probability density function 

The NN’s output _a_<sup>[</sup><sup>_L_]</sup> ( _._ ) depends on the NN’s parameters **W**<sup>[</sup><sup>_l_]</sup> _, b_ [ _l_ ] _, l_ = 1 _, . . . , L_ . Following the same steps as in the case of linear regression, the ML parameters are obtained 

17 

_1.4 Backpropagation_ 

by minimizing the quadratic cost function: 

Thus, minimizing the <u>quadratic cost</u> corresponds to finding the ML parameters for the model (1.34). 

Similarly, for the binary classification problem we have the training data set 

and we model the probability of _<u>xi</u>_<sup>belongingthetheclasswhere</sup><sup>_y_=1asbeingan</sup> output of the NN with _L_ layers and a single neuron in the output layer 

Applying the ML principle, we again minimize the <u>cross-entropy cost</u> function 

### **1.4 Backpropagation** 

Backpropagation is an efficient method to compute the gradient of the cost function _C_ with respect to the the NN parameters **W**<sup>[</sup><sup>_l_]</sup> _, b_ [ _l_ ] _, l_ = 1 _, . . . , L_ . The algorithm applies the chain rule for derivatives in a specific order that is efficient. Computing the gradients is essential for using the gradient descent algorithm and improving the NN performance. Backpropagation became widely appreciated after a 1986 paper by David Rumelhart, Geo↵rey Hinton, and Ronald Williams. The paper describes several NNs where backpropagation improves upon earlier approaches significantly, allowing the NNs to solve problems that had previously been insolvable. Today, backpropagation is the standard approach for learning in NNs. 

To begin, consider a NN with _L_ layers where the <u>output layer has only one neuron producing the scalar output</u> _<u>a</u>_<sup>[</sup><sup>_L_]</sup> . Consider our training set 

18 

_1.4 Backpropagation_ 

and the <mark>quadratic cost</mark> function 

We have introduced _a_<sup>[</sup><sup>_L_]</sup> ( _<u>x</u>_ _~~i~~_<sup>)toemphasizethedependenceofthenetworkoutput</sup><sup>_a_[</sup><sup>_L_]</sup> on the input variable _<u>x</u>_ <u>.</u> In what follows, we focus on the cost for a single sample pair ( _<u>x</u>_ _~~i~~_<sup>_, yi_)</sup><sup>_2D_.Oncewecomputethegradientof</sup><sup>_C_(</sup><sup>_<u>x</u>_</sup> _i_<sup>_, yi_)for</sup><sup>_i_=1</sup><sup>_, . . . , N_,wesumthese</sup> gradients to get the gradient of the total cost _C_ . For brevity, we introduce _a_<sup>[</sup><sup>_L_]</sup> = _a_<sup>[</sup><sup>_L_]</sup> ( _<u>x</u>_ _~~i~~_<sup>),</sup> _y_ = _yi_ , and _<u>x</u>_ = _<u>x</u>_ _~~i~~_<sup>.</sup> 

So consider the <u>per-sample cost</u> in the short-hand notation 

We are interested in finding the per-sample partial derivatives of the NN parameters 

First, we introduce notation for derivatives. Consider a function _f_ : _R_<sup>_N_</sup> _! R_<sup>_M_</sup> and the vectors _<u>x</u> 2 R_<sup>_N_</sup> and _<u>y</u>_ = _f_ ( _<u>x</u>_ <u>)</u> _2 R_<sup>_M_</sup> . We write 

This matrix is called the _Jacobian_ and has the same meaning as a normal (univariate) derivative in the sense that a small change ∆ _<u>x</u>_ to _<u>x</u>_ results in a change _@@xy_ ∆ _<u>x</u>_ to _<u>y</u>_ . Note that for _y 2 R_ and _<u>x</u> 2 R_<sup>_N_</sup> the Jacobian reduces to a gradient vector, which is a row vector according to the notation above. 

As for univariate derivatives, there is a chain rule for Jacobians. Consider the functions _f_ : _R_<sup>_N_</sup> _! R_<sup>_M_</sup> , _g_ : _R_<sup>_M_</sup> _! R_<sup>_K_</sup> , and the composite function _h_ = _f ◦ g_ : _R_<sup>_N_</sup> _! R_<sup>_K_</sup> . Then _<u>x</u> 2 R_<sup>_N_</sup> , _<u>y</u>_ = _f_ ( _<u>x</u>_ <u>)</u> _2 R_<sup>_M_</sup> , and _<u>z</u>_ = _g_ ( _<u>y</u>_ <u>) =</u> _h_ ( _<u>x</u>_ <u>)</u> _2 R_<sup>_K_</sup> . The Jacobian chain rule reads 

The intuition behind the chain rule is that a small change ∆ _<u>x</u>_ to _<u>x</u>_ results in a change _@@xy_ ∆ _<u>x</u>_ to _<u>y</u>_ <u>,</u> which results in a change _@_<sup>_@_</sup> _<u>y</u>_<sup>_<u>z</u>_</sup> _@@xy_ ∆ _<u>x</u>_ to _<u>z</u>_ <u>.</u> The Jacobian chain rule can be proved by looking at the single entry from _@_<sup>_@_</sup> _<u>x</u>_<sup>_<u>z</u>_</sup> and noticing that it is an inner product of a row from<sup>_@z_</sup> and a column from _@y_ . _@y @x_ 

We now return to the problem of computing the derivatives of the parameters of a NN. We first compute the derivatives<sup>_@C_</sup> _@z_<sup><u>(</u></sup><sup>_<u>z</u>_</sup> [ _l_ [] _l_ ])<sup>_, l_= 1</sup><sup>_, . . . , L_.The derivatives can be interpreted</sup> 

19 

_1.4 Backpropagation_ 

as an indication how each layer’s input contributes to a change in the cost. Once they are computed, we claim that obtaining the derivatives (1.37) is straightforward. We start with the derivative of the cost for the last layer: 

Now, we derive an equation for the next layer (moving backwards) 

where the computation of the second and third partial derivatives is left as an exercise. For the last step, the derivative<sup>_@g_</sup> _@_<sup><u>(</u></sup> _<u>z</u>_<sup>_<u>z</u>_</sup> [ _l_ [ _l_ ]]) is a diagonal matrix with a diagonal equal to _g_<sup>_0_</sup> ( _<u>z</u>_ [ _l_ ]). Multiplying by such a matrix <u>is equivalent to an element-wise multiplication</u> [ _l_ ]) of vectors, i.e., the Hadamard product, which we denote by _⊙_ . Note that<sup>_@C_</sup> _@z_<sup><u>(</u></sup><sup>_<u>z</u>_</sup> [ _l_ ] can [ _l_ +1]) be computed easily if we know<sup>_@C_</sup> _@z_<sup><u>(</u></sup><sup>_<u>z</u>_</sup> [ _l_ +1]<sup>.Using(1.40)and(1.44)wecancomputethe</sup> values for<sup>_@C_</sup> _@z_<sup><u>(</u></sup><sup>_<u>z</u>_</sup> [ _l_ [] _l_ ])<sup>_, l_=</sup><sup>_L, L −_1</sup><sup>_, . . . ,_1.</sup> We next compute the derivatives (1.37): 

For the derivative with respect to **W**<sup>[</sup><sup>_l_]</sup> , we consider first the derivative <u>with respect to</u> [ _l_ ] the _i_ -th row of **W**<sup>[</sup><sup>_l_]</sup> , denoted by _<u>W</u> i_ :<sup>.Wecannotconsiderthederivativewithrespect</sup> 

20 

~~<u>{we consiolor ACB) 9 _ 4 alae</u> of 3°)~~ ~~<u>0 d ¢(3™)</u> gard~~ ~~<u>= hy)</u> (2)~~ ~~<u>2) 7 dc ( gt!) | of c( 3%") q in</u>~~<sup>~~<u>g(2"))</u>~~</sup> ~~<u>ae as wer ag</u>~~<sup>~~<u>(</u>~~</sup> ~~<u>.</u> I~~ ~~<u>d 3" - i i | yyoertd" A clzr)</u> dp f~~ 

_1.4 Backpropagation_ 

to the matrix **W**<sup>[</sup><sup>_l_]</sup> because our Jacobian definition is only for vectors 

[ _l_ ] Lets assume the matrix **W**<sup>[</sup><sup>_l_]</sup> has _M_ rows. Stacking the rows vectors<sup>_@C_</sup> _@W_<sup><u>(</u></sup><sup>_<u>W</u>_</sup> [ _~~i~~ l_ : _~~<u>i</u>~~_ ]:<sup><u>)</u></sup> for _i_ = 1 _, . . . , M_ , we get a matrix of the same size as **W**<sup>[</sup><sup>_l_]</sup> , <u>where each entry is the derivative</u> _@C_ ( _Wi_<sup>[</sup> _<u>j</u>_<sup>_l_])</sup> . This matrix describes how **W**<sup>[</sup><sup>_l_]</sup> should be changed to decrease the cost. We _@W_<sup>[</sup><sup>_l_]</sup> _ij_ denote the matrix by (with slight abuse of notation) 

Summarizing, we can write the four equations specifying the backward propagation algorithm for a NN with _L_ -layers: 

Note that the gradient for the output layer (1.51) <u>depends on the cost function that</u> we used and <u>the fact that the output layer consists of a single neuron.</u> For other cost functions, this expression should be derived as in steps (1.38)–(1.40). In the backpropagation algorithm, equation (1.51) is used for initialization. Then, equation (1.52) is applied iteratively for the subsequent layers _l_ = _L −_ 1 _, . . . ,_ 1. Finally, using (1.53) and (1.54), the gradients of the NN’s parameters can be computed from the previously computed gradients<sup>_@C_</sup> _@z_<sup><u>(</u></sup><sup>_<u>z</u>_</sup> [ _l_ [] _l_ ]) for _l_ = _L, . . . ,_ 1. The backpropagation algorithm <u>(1.51)– (1.54) requires knowledge of</u> _<u>z</u>_ [ _l_ ] <u>and</u> _<u>a</u>_ [ _l_ ] <u>for</u> _<u>l</u>_ = <u>0</u> _<u>, . . . , L</u>_ <u>. Recall that these vectors can be computed by the forward-propagation algorithm (1.31)–(1.33).</u> 

21 

_1.4 Backpropagation_ 

##### **Gradient Descent** 

The backpropagation algorithm from the previous section regards the derivatives of the cost function for a single input-output sample from the training set with respect to the network parameters. The gradient descent algorithm requires summing up the per-sample results and could be performed as follows. 

1. Choose **W**<sup>[</sup><sup>_l_]</sup> _, b_ [ _l_ ] for _l_ = 1 _, . . . , L_ , perhaps randomly 

2. For each sample ( _<u>x</u>_ _~~i~~_<sup>_, yi_),</sup><sup>_i_= 1</sup><sup>_, . . . , N_</sup> 

   - [ _l_ ] [ _l_ ] 

   - 2.1. For the input _<u>x</u>_ _~~i~~_<sup>,computethevalues</sup><sup>_<u>a</u>_</sup> _i_<sup>_, z_</sup> _~~i~~_<sup>,</sup><sup>_l_=1</sup><sup>_, . . . , L_,usingforward</sup> propagation (1.31)-(1.33) 

   - [ _l_ ]) 

   - 2.2. For the output _yi_ , compute the gradients<sup>_@C_</sup> _@_<sup>_<u>i</u>_</sup> **W**<sup><u>(</u></sup><sup>**W**[</sup><sup>_l_][</sup><sup>_l_])</sup> _,_<sup>_@C_</sup> _@_<sup>_<u>i</u>_</sup> _<u>b</u>_<sup><u>(</u></sup> [<sup>_<u>b</u>_</sup> _l_ ] , _l_ = 1 _, . . . , L_ , using backpropagation (1.51)-(1.54). Save the gradients. 

3. <u>Sum the</u> saved gradients to obtain the total gradients for _l_ = 1 _, . . . , L_ 

We remark that using this update is called <u>a</u> _<u>batch</u>_ <u>or</u> _<u>o✏ine</u>_ <u>gradient</u> calculation. One can also perform the gradient calculation _<u>sequentially</u>_ <u>, or</u> _<u>online</u>_ <u>, by updating</u> the <u>gradient</u> sequentially for each <u>pair (</u> _<u>x</u>_ _~~<u>,</u>~~_<sup>_<u>yi</u>_</sup><sup><u>),</u></sup><sup>_i_= 1</sup><sup>_<u>, . . . , L</u>_.</sup> 

4. Update for _l_ = 1 _, . . . , L_ 

5. Repeat steps 2–4 until convergence 

In contrast to linear or logistic regression, the problem of minimizing the cost function for NNs is <u>in general</u> _<u>non-convex</u>_ <u>.</u> This can be seen in (1.30). Even if we assume that the cost function is convex, the output of the NN may be a non-convex function as it is a composition of many functions (even if the activation functions are convex, this does not mean that the composition function is convex). Applying gradient descent to a nonconvex problem will usually result in convergence to a local minimum. <u>NNs trained with the gradient descent algorithm often achieve better performance than linear models.</u> 

22 

~~algorithm fv binary classr-freation~~ ~~<u>o 4c)dati de,dat dotaz Hh. BOLD cat-fnchon4A :_ A f-ylogat—Cey)lagCia") Ff gt</u> oat d 2°~~ <u>-</u> ~~. » GAY~~ <u>pot</u> ~~<u>[-y -op ek | q (@)</u>~~ <u>=</u> ~~<u>Wd</u> i~~ <u>; (4-a*) ) Scalay Stel</u> ~~<u>é) dclz) 4 d c(b™) AL</u>~~ <u>df</u> ~~<u>34</u>~~ <u>ol</u> ~~<u>bt</u> Q~~ ~~<u>aclwe)wl lay of yer)</u>~~ 

## ~~<u>Regn larization of NNs</u>~~ 

<u>=</u> ~~<u>Hp48</u> C - 14) |]?~~ i= ~~é "Bl All wil~~ - ~~<u>= 2</u>~~ <u>4</u> ~~<u>yi</u>~~ <u>boy ( ai”)) ey)</u> ~~<u>cry dog(oq (4(4—- ai)Qe +2 all W thi ||]</u>~~ <u>:</u> ~~<u>ac</u>~~ <u>cei</u> ~~<u>acl")</u>~~ <u>|</u> ~~<u>= 1] al Q: 9 zit + ie rv</u>~~<sup>~~W~~</sup> 

## **2 Probabilistic Graphical Models** 

### **2.1 Introduction** 

Probabilistic graphical models (PGMs) [1, 2, 3, 4] connect <u>probability theory</u> and <u>graph theory,</u> and provide a <u>graph representation of the conditional dependence structure in (possibly large) collections of random variables with complex interactions.</u> PGMs are used in many machine learning applications, such as image and speech recognition, natural language processing, medical diagnosis, and channel coding. 

The inference problem we will consider here is computing marginal distributions of hidden variables _X_<sup>_n_</sup> = _X_ 1 _, X_ 2 _, . . . , Xn_ given observed variables _Y_<sup>_m_</sup> = _Y_ 1 _, Y_ 2 _, . . . , Ym_ and a statistical model _P_ ( _x_<sup>_n_</sup> 1<sup>_, ym_).Inotherwords,wewishtoefficientlycompute</sup> 

where the notation<sup>P</sup> _⇠xi_<sup>denotesthesumoverall</sup><sup>_xj_,</sup><sup>_j_= 1</sup><sup>_, . . . , n_,exceptfor</sup><sup>_xi_:</sup> 

### **2.2 Factor Graphs** 

Consider the following _global_ function of many variables 

and suppose _f_ ( _·_ ) factors as <u>a product of</u> _<u>local</u>_ <u>functions,</u> each of which depends on a subset of the variables, i.e., we have 

where _J_ is a discrete index set, _Xj_ is a subset of _{x_ 1 _, x_ 2 _, . . . , xn}_ , and _fj_ ( _Xj_ ) is a local function having the elements of _Xj_ as arguments. Such factorizations can be visualized by _factor graphs_ [7, 9] with _<u>variable nodes</u>_ that represent variables and _<u>factor nodes</u>_ that represent the local functions _fj_ ( _·_ ). 

For example, consider the global function 

_2.3 Sum-Product Algorithm_ 

Figure 2.1: Factor graph for the function in (2.5). 

whose factor graph is shown in Fig. 2.1. In this case, the graph is a _tree_ , i.e., a connected <u>graph without cycles.</u> Factor graphs generalize PGMs, but we will focus mainly on global functions _f_ ( _·_ ) that are probability distributions [8, 10]. 

A basic problem of factor graphs is _<u><mark>marginalization</mark></u>_ <u><mark>,</mark></u> i.e., compute 

For example, the marginal of _X_ 2 in Fig. 2.1 is 

_·_ where we have abused notation by using the same notation _f_ ( ) as for the global function. We will continue this convention, if the context makes clear which function we are ˜ considering. We further use the notation _f_ (˜ _xi_ ) to emphasize that _f_ is evaluated for _xi_ . This choice will become clearer in Sec. 2.3, where we distinguish between the <u>variable node</u> _<u>xi</u>_ and the <u>value</u> _<u>x</u>_ ˜ _<u>i</u>_ <u>for which the message is evaluated.</u> 

<mark>If the alphabet of each variable has cardinality</mark> _<mark>d</mark>_ <mark>, then it seems that we must perform 3</mark> _<mark>d</mark>_<sup>4</sup> <mark>multiplications and</mark> _<mark>d</mark>_<sup>4</sup> <mark>additions for each value of</mark> _<mark>X</mark>_ 2 <mark>.</mark> 

### **2.3 Sum-Product Algorithm** 

The sum-product algorithm [9] performs inference on graphical models. For tree-structured (cycle-free) factor graphs, the sum-product algorithm efficiently computes the exact marginals. To show this, the basic tool we use is the _distributive law_ of multiplication [12]: 

Observe that the the right hand side needs one multiplication rather than two. More generally, we can formulate for two arbitrary functions _f_ and _g_ the _Cartesian-product distributive law_ [11, Lemma 12.1] as 

24 

_2.3.1 Separation and Marginalization_ 

Figure 2.2: Separation. 

where _X_ and _Y_ are two sets and _X ⇥Y_ is their Cartesian product. The computational savings are even more obvious here. Instead of having _|X| |Y|_ multiplications and _|X| |Y| −_ 1 additions, we only have one multiplication and _|X|_ + _|Y| −_ 2 additions. Applying the distributive law to (2.7), we can reorganize the sums as 

In the following, we interpret this reorganization graphically, and we consider its impact on the computational complexity. 

#### **2.3.1 Separation and Marginalization** 

The tree-structured factor graph can be separated into two parts that are connected to _x_ 2, see Fig. 2.2. We interpret the graphical separation by using the distributive law of multiplication to write 

Now define the _messages_ from factor node _fj_ to variable node _xi_ as 

<mark>where</mark> _<mark>mf</mark>_ 1 _<mark>!x</mark>_ 2 <mark>(</mark> _<mark>x</mark>_ ˜ <mark>2) is function of the realization</mark> _<mark>x</mark>_ ˜ <mark>2 only.</mark> From (2.12)-(2.13) we see that computing the marginal for the variable _x_ 2 requires computing the _product_ of the incoming messages of the variable node _x_ 2. More generally, we have 

We will use a second kind of message _mxi!fj_ (˜ _xi_ ) from variable node _xi_ to factor node _fj_ . In Fig. 2.5 and 2.4 the respective messages are depicted as vectors, e.g., _<u>m</u>_ _~~x~~_ 3 _!f_ 2<sup>with</sup> 

25 

_2.3.2 Factor Nodes to Variable Nodes_ 

Figure 2.3: Compute the marginal. 

Figure 2.4: Message from factor nodes to variable nodes. 

the components being _mx_ 3 _!f_ 2(˜ _x_ 3) _, 8x_ ˜3 _2 X_ . Both types of messages are defined more precisely next. 

#### **2.3.2 Factor Nodes to Variable Nodes** 

Consider the messages from factor nodes to variable nodes, e.g., _mf_ 2 _!x_ 2(˜ _x_ 2) in Fig. 2.4. We can write (2.13) as 

where _mxi!fj_ (˜ _xi_ ) is the message from variable node _xi_ to factor node _fj_ , which is a ˜ function of _xi_ . From (2.15), we see that computing the message from _f_ 2 to _x_ 2 requires computing the _sum_ of incoming messages of _f_ 2 _except_ the message from _x_ 2. More generally, we have 

#### **2.3.3 Variable Nodes to Factor Nodes** 

Consider the message from variable nodes to factor nodes, e.g., _mx_ 3 _!f_ 2(˜ _x_ 3) in Fig. 2.5. We can write the term in square brackets in (2.15) as 

26 

_2.3.5 Max-Product Rules_ 

2. update the messages with sum-product rules; 

3. send one message to each neighbor. 

The sum-product message passing rules are given by (2.14), (2.16), (2.18): 

- Marginals: Multiply all incoming messages 

- Messages from factor nodes: add incoming messages except for the received message 

- Messages from variable nodes: multiply incoming messages except for the received message 

#### **2.3.5 Max-Product Rules** 

Suppose we wish to calculate the value 

Fortunately, there is a generalized distributive law for maximization: 

max( _ab, ac_ ) = _a_ (max( _b, c_ )) _, a >_ 0 _._ (2.24) 

The corresponding message passing algorithm for performing marginalization is called _max-product_ because the sums in (2.20)-(2.22) are replaced by maximizations. The update rules are as follows. 

- Max-marginal: _Mi_ 

- _•_ Messages from factor nodes: 

28 

_2.4 Belief Propagation_ 

Figure 2.6: Factor graph with a cycle. 

### **2.4 Belief Propagation** 

Belief propagation is a sum-product algorithm that performs inference on graphs. If the graph is cycle-free then belief propagation computes the exact marginals. However, if the graph has cycles (see the red lines in Fig. 2.6), then inference is usually only approximate, and the algorithm might not converge. 

When performing _loopy_ belief propagation, i.e., belief propagation on graphs with cycles, the messages along one edge are usually updated often based on di↵erent incoming messages once the messages cycle around. One usually updates the messages as in (2.20)(2.22). The messages are not necessarily probabilities, but they are usually non-negative, and appropriate normalization gives a probability distribution [13]. Observe that the messages can be passed endlessly around the graph. However, for some graph structures the message values can converge after some iterations. In case of convergence, the result does not necessarily correspond to exact marginalization, but studies have shown that the approximation is often good [14, 15, 16]. 

#### **2.4.1 Example: Single Parity Check Code of Length Three** 

Consider a single-parity check code that has the codewords _x_<sup>3</sup> _2 {_ 000 _,_ 011 _,_ 101 _,_ 110 _}_ . We can describe this code by using one _parity check_ constraint 

where _⊕_ denotes the XOR operation. 

Let _Yi_ be a _continuous_ random variable that is a noisy version of _Xi_ . For example, we might have transmitted uniformly-distributed _X_<sup>3</sup> = _X_ 1 _, X_ 2 _, X_ 3 via on-o↵keying (OOK) over an additive white Gaussian noise (AWGN) channel 

where _E_ is the average energy of the signal and _Z_<sup>3</sup> = _Z_ 1 _, Z_ 2 _, Z_ 3 is a string of independent and identically distributed (i.i.d.) Gaussian random variables. 

29 

_2.4.1 Example: Single Parity Check Code of Length Three_ 

Figure 2.7: Factor graph for a single parity check code of length three. 

To minimize the bit error rate, we would like to compute _P_ ( _xi|y_<sup>3</sup> ), _i_ = 1 _,_ 2 _,_ 3, by using belief propagation to marginalize _P_ ( _x_<sup>3</sup> _|y_<sup>3</sup> ). As a fist step, observe that 

and that _p_ ( _y_<sup>3</sup> ) is simply a constant for the marginalization. We may thus alternatively marginalize the numerator in (2.30) and subsequently normalize the results to obtain the desired probabilities. 

Suppose each codeword has the same probability _P_ ( _x_<sup>3</sup> ) = 1 _/_ 4, and consider the factorization 

where the indicator function I( _·_ ) takes on the value 1 if its argument is true, and 0 otherwise. The resulting factor graph is shown in Fig. 2.7 and it has the factor functions 

where we have removed the factor 1/4 for convenience. 

As an example of belief propagation, suppose we have the following normalized statistics at the leaves _fi_ ( _xi_ ), _i_ = 1 _,_ 2 _,_ 3, of the factor graph: 

The constants _ci_ = _P_ ( _xi_ ) _/p_ ( _yi_ ) are chosen to give the probabilities _P_ ( _xi|yi_ ) in the table. We now use (2.21) and (2.22) to compute 

30 

_2.4.1 Example: Single Parity Check Code of Length Three_ 

Figure 2.8: Belief propagation for a single parity check code of length three. 

Observe that each edge may as well carry one number only, as we are dealing with probabilities for binary random variables. Next, we use (2.21) to compute 

and similarly for _mf_ 4 _!x_ 2(˜ _x_ 2) and _mf_ 4 _!x_ 3(˜ _x_ 3). We have 

The marginal distribution of _X_ 1 is computed by using (2.20): 

Finally, we normalize to obtain 

We similarly compute 

31 

_2.4.2 Example: Single Parity Check Code of Length Four_ 

Figure 2.9: Factor graph for a single parity check code of length four. 

Observe that the average bit-error rate is minimized by choosing _x_ ˆ _i_ = 1 for all _i_ = 1 _,_ 2 _,_ 3, but this does not give a valid codeword. This will not happen if we instead try to minimize the block error probability. 

#### **2.4.2 Example: Single Parity Check Code of Length Four** 

Consider next a single parity check code of length four with the constraint 

We can repeat the above analysis with the factor graph shown in Fig. 2.9. However, observe that the number of multiplications in computing the messages in (2.20)-(2.22) is proportional to the number of neighbors minus one. Thus, if a node’s degree is large, this node must perform many multiplications. For example, factor node _f_ 5 in Fig. 2.9 has four neighbors, and must thus perform three multiplications for every term in the brackets in (2.21). 

To reduce complexity, one can try to find alternative graphs for the same code. Consider, for example, the _trellis_ of the length four single-parity check code shown in Fig. 2.10. The trellis is a _state_ diagram where the state is given by (see [17]) 

The eight paths on the trellis represent the eight codewords _x_<sup>4</sup> that satisfy the constraint 2.36. 

We can draw the trellis as a factor graph with three state variables _s_ 1, _s_ 2, _s_ 3, as shown in Fig. 2.11. The factorization is now 

where _s_ 0 = 0. The new factor graph has three additional factor and variable nodes as compared to Fig. 2.9, but now each variable node has degree two, and each factor 

_2.4.2 Example: Single Parity Check Code of Length Four_ 

Figure 2.10: Trellis for a single parity check code of length four. The edge labels are the codeword symbols _xi_ , _i_ = 1 _,_ 2 _,_ 3 _,_ 4. 

Figure 2.11: Alternative factor graph for a single parity check code of length four. 

node has degree at most three. Thus, at most one multiplication is needed in (2.20) and (2.21), and the variaable-to-factor node messages (2.22) are simply passed on. 

The above ideas extend naturally to any code that can be described by a state diagram, e.g., convolutional or trellis or linear block codes. The marginalization turns out to be especially useful for _soft-in, soft-out_ processing for turbo codes, low-density paritycheck (LDPC) codes, and other codes designed for iterative decoding. It is also useful for _turbo equalization_ [18]. For example, a soft-in, soft-out equalizer for the channel _h_ ( _D_ ) = 1+ _D_ + _D_<sup>2</sup> and an OOK signal with three symbols can be implemented by using belief propagation on the trellis shown in Fig. 2.12. 

33 

_2.4.2 Example: Single Parity Check Code of Length Four_ 

Figure 2.12: Trellis for the channel _h_ ( _D_ ) = 1 + _D_ + _D_<sup>2</sup> and three OOK symbols with _E_ = 1 _/_ 2. The state labels are the last two input symbols. The edge labels are the noise-free channel output symbols. 

34 

~~Al* Fuckea Pearl (UA) » 1982~~ ~~<u>leallager wv 14b0</u> Crmp le SiclohuanithezenmnbersmAnz72) 1 Vomiables Yi~Xy 6 Constraint frunetions~~ ~~<u>Rw 4 AlCx%w)eA)</u> 6~~ ~~<u>an</u>~~ <u>Constraints</u> ~~<u>A = > (423) (4) 3,2) » (2,43), (203, DY, G,42) 0,21) ] colums: A(X Yu. ¥e)eL)</u>~~<sup>~~35a}~~</sup> ~~6 tomstrains Xi = ara max Pll yi ya) we 24)~~<sup>~~2,~~</sup> ~~<u>Consider a Plat yt) = Po) Plyilx) = AL») Ale [| Plyil xi )</u> CER EERCECE CEE~~ ~~<u>EP</u>~~ <u>cya</u> 

~~<u>P(4 1X)</u>~~ <u>»</u> ~~<u>ere idk hts iic J y) 3 v</u>~~ <u>oO</u> ~~PUYerx=1)=4q~~ ~~<u>PY</u> piy=al x=i=4 ply221 Xl) = 0 Ply x= )=0~~ ~~<u>Pyeeled PHL</u>~~ <u>|</u> ~~<u>Plgeala-2) Phe) | Plysolacadplies)_ Pye) ey</u> on plx=1)> Pld) = PIK=3)~~ ~~<u>» Pl ysafxet) = PlY=al x2) = Ply=a)r=3)</u> “%~~ 

## **Bibliography** 

- [1] M. J. Wainwright, “Graphical models and message-passing algorithms: Some introductory lectures,” in _Mathematical Foundations of Complex Networked Information Systems_ . Springer, 2015, pp. 51–108. 

- [2] M. J. Wainwright, M. I. Jordan _et al._ , “Graphical models, exponential families, and variational inference,” _Foundations and Trends in Machine Learning_ , vol. 1, no. 1–2, pp. 1–305, 2008. 

- [3] D. Koller, N. Friedman, and F. Bach, _Probabilistic graphical models: principles and techniques_ . MIT press, 2009. 

- [4] D. Barber, _Bayesian reasoning and machine learning_ . Cambridge Univ. Press, 2012. 

- [5] W. Ryan and S. Lin, _Channel codes: classical and modern_ . Cambridge Univ. Press, 2009. 

- [6] B. Matuz and G. Liva, _Lecture notes for Channel Codes for Iterative Decoding (SS2018)_ . LNT, Technische Universit¨at M¨unchen, 2018. 

- [7] S. M. Aji and R. J. McEliece, “A general algorithm for distributing information in a graph,” _IEEE Int. Symp. Inf. Theory_ , 1997, p. 6. 

- [8] R.G. Gallager, _Low-density parity-check codes_ . M.I.T. Press, 1963. 

- [9] F. R. Kschischang, B. J. Frey, and H.-A. Loeliger, “Factor graphs and the sumproduct algorithm,” _IEEE Trans. Inf. Theory_ , vol. 47, no. 2, pp. 498–519, 2001. 

- [10] J. Pearl, _Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference_ . Elsevier, 2014. 

- [11] D. Forney, “The Sum-Product Algorithm,” in _6.451 Principles of Digital Communication II_ , Cambridge MA, 2005, MIT OpenCourseWare. 

- [12] S. M. Aji and R. J. McEliece, “The generalized distributive law,” _IEEE Trans. Inf. Theory_ , vol. 46, no. 2, pp. 325–343, 2000. 

- [13] J. Mooij and H. Kappen, “Sufficient conditions for convergence of loopy belief propagation,” _arXiv preprint arXiv:1207.1405_ , 2012. 

- [14] A. T. Ihler, W. F. John III, and A. S. Willsky, “Loopy belief propagation: Convergence and e↵ects of message errors,” _J. Machine Learning Research_ , vol. 6, no. May, pp. 905–936, 2005. 

_Bibliography_ 

- [15] J. S. Yedidia, W. T. Freeman, and Y. Weiss, “Generalized belief propagation,” _Advances Neural Inf. Proc. Sys._ , 2001, pp. 689–695. 

- [16] P. Sen and L. Getoor, “Empirical comparison of approximate inference algorithms for networked data,” _Open Problems in Statistical Relational Learning: Papers from the ICML Workshop. Pittsburgh, PA: www.cs.umd.edu/projects/srl2006_ , vol. 2, 2006. 

- [17] L. R. Bahl, J. Cocke, F. Jelinek, and J. Raviv, “Optimal decoding of linear codes for minimizing symbol error rate,” _IEEE Trans. Inf. Theory_ , vol. 20, no. 2, pp. 284–287, March 1974. 

- [18] C. Douillard, M. J´ez´equel, C. Berrou, A. Picart, P. Didier, A. Glavieux, “Iterative correction of intersymbol interference: Turbo-equalization,” _Eur. Trans. Telecommun._ , vol. 6, no. 5, pp. 507-511, Sept./Oct. 1995. 

36 

## **3 Approximate Message Passing** 

### **3.1 Introduction** 

Approximate Message Passing (AMP) [2] algorithms are a powerful tool for inference in _Generalized Linear Models (GLMs)_ of the form 

_<u>Y</u>_ = _f_ ( **A** _<u>X</u>_ + _<u>N</u>_ <u>)</u> (3.1) 

where<sup>1</sup> 

- _<u>X</u> 2 R_<sup>_n_</sup> contains i.i.d. random variables with distribution _pX_ 

- **A** _2 R_<sup>_m⇥n_</sup> is a known matrix 

- _<u>N</u> 2 R_<sup>_m_</sup> is a noise vector with i.i.d. entries 

- _f_ : _R ! R_ is a scalar function that is applied component-wise to a vector 

- _<u>Y</u>_ is an observation vector. 

Applications are, e.g.: 

   - **Compressed Sensing** : Here, _<u>X</u>_ represents a signal of interest that we wish to recover from possibly quantized measurements _<u>Y</u>_ <u>.</u> A typical example is that _<u>X</u>_ is an image which can (in a suitable linear basis) be modeled as a _sparse_ vector, i.e., a vector with only a few non-zero coefficients. In this case, **A** is a matrix that represents the physical measurement process, _<u>N</u>_ is a measurement noise vector and _f_ is a scalar quantization function which may be neglected for very fine quantization. Since physical measurements are often expensive, the goal is to approximate _<u>X</u>_ well from as few measurements as possible such that _<u>Y</u>_ has a much smaller dimension than _<u>X</u>_ <u>.</u> In this case, many solutions (for _<u>X</u>_ <u>)</u> to (3.1) exist and the challenge is to find one that is close to the original signal. Compressed Sensing is based on the insight this is often still possible even if _m ⌧ n_ [1]. 

   - **Massive MIMO** : In multiple-input, multiple-output communications (MIMO), a base station with a large amount of antennas receives the signals from many users at the same time. Thus, at each antenna it receives the superposition of each user’s signal after manipulation by a channel coefficent. In this setting, _<u>X</u>_ are the user’s transmit signals, **A** contains all channel coefficients, _<u>N</u>_ represents additive noise and _f_ may represent a quantizer at each antenna of the base station. _<u>Y</u>_ is then the quantized received signal from which we want to estimate _<u>X</u>_ <u>.</u> A detailed example for this setting is given in Section 3.4. 

- 1We will see in Section 3.4 that it is also possible to consider complex-valued random variables. 

_3.2 Message Passing Derivation_ 

- **Regression/Classification** : In Chapter 1, we studied GLMs in the context of linear or logistic regression. There, we wanted to learn a suitable weight vector in order to build a predictor from a single neuron. In these settings, the rows of **A** in (3.1) contain the training samples that are labeled by the elements of _<u>Y</u>_ <u>,</u> _<u>X</u>_ is the weight vector we wish to learn, _f_ is an activation function and _<u>N</u>_ represents noise. We will see that AMP provides an alternative way of learning _<u>X</u>_ (as compared to gradient descent) that can incorporate prior statistical information about _<u>X</u>_ <u>.</u> 

Suppose we wish to compute the minimum mean squared error estimator 

for a given observation _<u>y</u>_ . Clearly, this can be done if we have access to the marginals _pXi|Y_ ( _·|y_ <u>)</u> for each _i_ = 1 _, . . . , n_ . In Chapter 2, we studied the Belief Propagation algorithm that enables us to compute marginals efficiently on factor graphs. We will see that while a direct application of the Belief Propagation algorithm is computationally to costly in this setting, we can derive an _Approximate Message Passing (AMP)_ algorithm that works extremely well in certain settings. 

### **3.2 Message Passing Derivation** 

Since a rigorous derivation of AMP is beyond the scope of this lecture, we will focus on a heuristic derivation for the simpler linear model: 

Here, 

- The signal _<u>X</u>_ is a random vector with i.i.d. elements drawn from a known distribution _pX_ . A popular choice for _pX_ in Compressed Sensing is the _Bernoulli-Gaussian distribution_ 

which, for small _p_ , creates a sparse signal with roughly _np_ Gaussian _spikes_ and (1 _− p_ ) _n_ zeros. 

- The measurement matrix **A** is usually assumed to have random i.i.d. entries with variance 1 _/m_ . For the purpose of our derivation, however, we will assume that all entries have approximately constant magnitude: _|Aai| ⇡_ 1 _/_<sup>_p_</sup> _<u>m</u>_ for all _a_ = 1 _, . . . , m_ , _i_ = 1 _, . . . , n_ . 

- We assume that _m_ and _n_ are _large_ but that the ratio _m/n_ = _↵_ is fixed. 

While it is easy to solve (3.3) for _<u>x</u>_ if **A** is invertible, this is not the case if, for example, _m ⌧ n_ which is the case in Compressed Sensing. Thus, we wish to compute the marginal 

38 

Figure 3.1: Factor graph for the linear model (3.3). 

distributions of all _xi_ , _i_ = 1 _, . . . , n_ using the Belief Propagation algorithm. Our heuristic derivation is based on [3, Ch. 5]. 

Consider the factor graph depicted in Figure 3.1. Obviously, there are many loops since every variable node is connected to every factor node (except the ones representing the prior distribution). Thus, we need to update the messages iteratively and are not guaranteed to converge to the correct solution. Still, we will see that Belief Propagation can work very well in our setting. We start by writing down the messages. 

- Variable-to-factor message: 

- Factor-to-variable message: 

Unfortunately, these update equations are generally not tractable for three reasons: 

- (i) The messages are _functions_ . Hence, we need to compute them for infinitely many values of _x_ ˜ _i_ to characterize them precisely. 

- (ii) Every update in (3.6) contains integrals over _n −_ 1 variables. This is computationally too costly if _n_ is large. 

39 

_3.2.1 Simplifying the Messages_ 

- (iii) Since two messages are sent on each edge (one in each direction), there are 2 _nm_ messages to track. This is computationally costly for large _n_ and _m_ . 

The derivation of the computationally efficient AMP algorithm is based on two stages of approximations. In Section 3.2.1, we use the Central Limit Theorem to represent the messages by Gaussian distributions and remove the high-dimensional integrals (or summations) in (3.6). This addresses the issues (i) and (ii). Then, we drastically reduce the number of messages using a Taylor approximation to tackle (iii) in Section 3.2.2. We stress that the derivation here is heuristic and leaves out many mathematical details in order to make the main ideas more accessible. 

#### **3.2.1 Simplifying the Messages** 

Consider the factor node _fa_ and its corresponding message _mfa!xi_ to the variable node _xi_ . The factor node is defined by the observation 

which we can rewrite as 

i.e., the message _mfa!xi_ models a weighted average of all other variables _xj_ , _j_ = _i_ . Recall that the Central Limit Theorem (CLT) (loosely) states that for a sequence of independent and identically distributed random variables with ( _S_ 1 _, . . . , Sn_ ) with E [ _Si_ ] = _µ_ and Var[ _Si_ ] = _σ_<sup>2</sup> , the random variable 

converges in distribution<sup>2</sup> to a standard Gaussian random variable. Using this intuition and the assumption _|Aaj| ⇡_ 1 _/_<sup>_p_</sup> _<u>m</u>_ , we approximate the message _mfa!xi_ by a single Gaussian distribution that is fully characterized by its mean and variance. To this end, let _X_<sup>ˆ</sup> _xj !fa_ denote the random variable defined by the message _mxj !fa_ and let _x_ ˆ _xj !fa_ and _⌧xj !fa_ denote its mean and variance, respectively. We can now compute the mean of the factor-to-variable message as 

> 2We say that a sequence of random variables _Sn_ with cumulative density function _FSn_<sup>convergesto</sup> a random variable _S1_ with cumulative density function _FS1_ if lim _n!1 FSn_<sup>(</sup><sup>_x_)=</sup><sup>_FS_</sup> _1_<sup>forevery</sup> number _x 2 R_ where _FSn_<sup>iscontinuous.</sup> 

40 

and the variance as 

where in the last line we used that at the factor node _X_ ˆ _xj !fa_ are independent (see (3.6)). To simplify the variance even more, we assume that _fa_ , the incoming random variables _⌧xj !fa ⇡ ⌧xj !f_ is the same for all _a_ = 1 _. . . , m_ , i.e., the variance of all messages sent by variable node _j_ is the same. We approximate 

so that this part of the variance is the same for all _j_ = 1 _, . . . , n_ and we can track one value instead of _n_ . 

Using these simplifications, the factor-to-variable node updates can be written as: 

Let _φ_ ( _x_ ; _µ, σ_<sup>2</sup> ) denote the Gaussian probability density function with mean _µ_ and variance _σ_<sup>2</sup> evaluated at _x_ . We consider the variable-to-factor messages 

_3.2.1 Simplifying the Messages_ 

which can be simplified by noting that the product of Gaussian distributions again yields a Gaussian distribution. We can compute 

Since we assumed that _|Abi| ⇡_ 1 _/_<sup>_p_</sup> _<u>m</u>_ <u>,</u> we have that<sup>P</sup> _b_ = _a_<sup>_A_2</sup> _bi_<sup>_⇡_</sup><sup>_<u>m</u>_</sup> _m_<sup>_−_</sup><sup><u>1</u></sup> _⇡_ 1 for large _m_ and we can approximate 

In order to pass the variable-to-factor messages efficiently, we note that the factor-tovariable updates (3.13) only require the means _zfb!xi_ and variances _⌧fb!xi_ for all _b 6_ = _a_ . Further, we realize that the distribution (3.18) models<sup>P</sup> _b_ = _a_<sup>_Abizf_</sup> _b_<sup>_!x_</sup> _i_<sup>as an observation</sup> of _Xi_ in the presence of Gaussian noise with variance _⌧f !x_ . Thus, we can compute 

where the functions E and V are as 

where _N ⇠N_ (0 _, ⌧_ ) is independent of _X_ . A property that we will use later is that 

To satisfy the assumption that the variances are edge-independent, we first approximate 

and then combine the variance updates (3.15) and (3.22) to 

where we again used _|Aaj| ⇡_ 1 _/_<sup>_p_</sup> _<u>m</u>_ <u>.</u> Now we only need to keep track of a single variance. 

42 

_3.2.2 Reducing the Number of Messages_ 

At this point, we can formulate a first version of the message passing algorithm. Since the factor graph in Figure 3.1 is full of cycles, we get an iterative algorithm. We set the inializations _x_ ˆ<sup>1E [</sup><sup>_X_]forall</sup><sup>_i_=1</sup><sup>_, . . . , n_and</sup><sup>_⌧_1=Var [</sup><sup>_X_]</sup><sup>_/↵_.Foriterations</sup> _xi!fa_<sup>=</sup> _t_ = 1 _,_ 2 _, . . ._ , we compute the messages: 

#### **3.2.2 Reducing the Number of Messages** 

While the above update equations are now computationally feasible, we still need to track 2 _nm_ messages which is very costly if both _m_ and _n_ are large. The goal of this section is to reduce this to _O_ ( _m_ + _n_ ) messages. Suppose that we can set 

were the red colored terms are on the order of _O_ (1 _/_<sup>_p_</sup> _<u>n</u>_ <u>).</u> That is, we split the two messages each into a large edge-independent term and a smaller edge-dependent term. The edge-independent term ˆ _x_<sup>_t_</sup> _i_<sup>represents the mean computed from all incoming messages</sup> to variable node _i_ and is thus the current estimate of the conditional expectation of _Xi_ ˆ given _<u>y</u>_ . Thus, we are ultimately interested in finding and efficient way of computing _x_<sup>_t_</sup> _i_ instead of _x_ ˆ<sup>_t_allnodes!</sup> _xi!fa_<sup>for</sup> Plugging (3.27) into (3.24) yields 

where we can identify _δzf_<sup>_t_</sup> _a!xi_<sup>=</sup><sup>_Aaix_ˆ</sup><sup>_t_</sup> _i_<sup>sinceitistheonlytermdependingon</sup><sup>_i_after</sup> neglecting the last term. For the updates of _x_ ˆ<sup>_t_</sup> _x_<sup>+1</sup> _i!fa_<sup>,weperformaTaylorexpansion.</sup> Recall that the Taylor series of a function _g_ is given by 

43 

_3.2.2 Reducing the Number of Messages_ 

if these derivatives exist. We will expand the message _x_ ˆ<sup>_t_</sup> _x_<sup>+1</sup> _i!fa_<sup>aroundP</sup><sup>_m_</sup> _b_ =1<sup>_Abiz_</sup> _f_<sup>_t_</sup> _b!xi_ (an edge-independent term), use the approximation (3.28) and only consider the first two terms. Recall (3.21). We get 

where we again identified _x_ ˆ<sup>_t_</sup> _i_<sup>+1</sup> and _δx_ ˆ<sup>_t_</sup> _x_<sup>+1</sup> _i!fa_<sup>viathedependencieson</sup><sup>_a_.Using</sup><sup>_δz_</sup> _f_<sup>_t_</sup> _a!xi_<sup>=</sup> _Aaix_ ˆ<sup>_t_</sup> _i_<sup>,wecanfurthersimplifythefirstargumentofE(</sup><sup>_·_)andV(</sup><sup>_·_)to</sup> 

i.e., we have the updates: 

We can now plug the approximation of _δx_ ˆ<sup>_t_</sup> _xi!fa_<sup>backinto(3.30)toget</sup> 

Last, we plug (3.34) into (3.26) to simplify the variance update to 

_3.2.3 The AMP Algorithm_ 

#### **3.2.3 The AMP Algorithm** 

Algorithm 1 summarizes the AMP algorithm. Here, _⊙_ denotes element-wise multiplication and _h·i_ denotes the average of a vector. We see that it consists of a sequence of matrix-vector multiplications and the element-wise application of the functions E1, V1. 

### **3.3 From AMP to GAMP** 

We can now return to the setting of the GLM. In this setting, we model the noise and output function via a probability distribution. More generally, we can essentially use an arbitrary probability distribution that transforms each measurement into an observation. This system model is depicted in Figure 3.2. The resulting _Generalized Approximate Message Passing (GAMP)_ was first presented by Rangan<sup>3</sup> in [4]. 

Figure 3.2: Illustration of the GAMP system model. 

> 3To be precise, Rangan uses slightly fewer approximations in the derivation of the algorithm and thus gets slightly more complicated update equations than what we present here. 

45 

_3.3 From AMP to GAMP_ 

Compared to Algorithm 3.2.3, we need to change the factor-to-variable messages in (3.6) to include the output probability distribution _pY |Z_ that models _f_ and _<u>N</u>_ <u>.</u> This adds an estimation step to the factor update similar to the (3.20). This estimation is related to the scalar channel 

where _Z ⇠N_ ( _p, ⌧_ ) with parameters _p, ⌧_ that are computed in the GAMP iterations and _f_ and _N_ are defined by the specific problem setting. We define the functions 

The GAMP algorithm for the GLM (3.1) is stated in Algorithm 2. 

##### **Algorithm 2** Generalized Approximate Message Passing 

Input: measurement matrix **A** , observation _<u>y</u>_ <u>,</u> distribution _pX_ , _↵_ = _m/n_ . Initialize _t_ = 1, _<u>s</u>_ 1 = 0, _<u>x</u>_ ˆ1 = E [ _<u>X</u>_ <u>]</u> and _⌧_<sup>1</sup> = Var [ _<u>X</u>_ <u>]</u> _/↵_ . 

**while** convergence criterion is not met **do** 

_Factor Linear Step_ : Compute 

_Factor Nonlinear Step_ : Compute 

_Variable Linear Step_ : Compute 

_t_ = _t_ + 1 **end while return** _<u>x</u>_ ˆ _t_ 

**Remark:** GAMP works very well if the matrix **A** has i.i.d. entries with zero mean and a fixed variance. This was extended to a larger class of matrices in [5]. For these matrices, the asymptotic (as _m, n ! 1_ ) bevavior (such as the estimation error) can easily be 

46 

_3.4 Application: Massive MIMO_ 

predicted via the so-called _State Evolution_ [4]. Recently, there has been significant process in investigating the fundamental performance limits of GLMs and GAMP [6]. For general matrices, however, the GAMP iterations can diverge and cause the algorithm to fail. 

### **3.4 Application: Massive MIMO** 

In the following, we elaborate on a practical application of GAMP for detection in a _massive multiple-input, multiple-output (MIMO)_ uplink. Massive MIMO has its origins in the work by Marzetta [7] in 2010 and tries to address the continuously growing demand for higher data rates and increased energy efficiency. Massive MIMO uses a very large number (100 or more) antennas at the base station to simultaneously accommodate many users. 

Pratical implementations still face cost and complexity related difficulties, as a naive implementation requires a seperate radio frequency chain for each antenna. In particular, analog-to-digital converters (ADCs) and digital-to-analog converters (DACs) must be extremely high-speed to handle the large bandwidths in the millimeter wave range. Highprecision ADCs/DACs operating at gigasample-per-second rates have an extremely high fabrication cost and energy consumption. Low resolution ADCs and DACs, even down to a single bit of resolution, have therefore gained a lot of interest in the massive MIMO literature recently. 

AMP and its more general version GAMP [4] show very good performance in the large system limit. In particular, they clearly outperform classical detectors based on matched filtering (MF), zero-forcing (ZF) and linear minimum mean-square error (LMMSE) approaches. 

#### **3.4.1 System Model** 

We consider a typical massive MIMO system in the uplink with _n_ single antenna users and _m_ antennas at the base station. The transmitter and receiver have perfect channel knowledge. The channel coefficients are modelled as iid Gaussian with zero mean and unit variance, i.e., _<u>H</u>_ _~~i~~_<sup>_⇠CN_(0</sup><sup>_, I_</sup> _m_<sup>).The</sup><sup>_n_singleantennauserstransmittheirsymbols</sup> _Xi 2 X , i_ = 1 _, . . . , n_ , which come from a power normalized QPSK constellation _X_ . The transmitted symbols are corrupted by additive white Gaussian noise _<u>N</u> ⇠CN_ (0 _, σ_<sup>2</sup> _<u>I</u>_ _~~m~~_<sup>).</sup> We have 

47 

_3.4.1 System Model_ 

_·_ The function Q( ) models the 1-bit quantization at the basestation incurred by the coarse quantization ADCs.<sup>4</sup> It is defined as 

_·_ If Q( ) is applied to a vector, it operates on each component individually. If the argument is a complex value, i.e., _x 2_ C, it is applied to the real and imaginary part independently. 

#### **GAMP for Massive MIMO** 

We wish to recover the vector _<u>X</u>_ from the 1-bit measurements _<u>Y</u>_ using the GAMP algorithm. The GAMP algorithm allows formulations for both approximating the maximum a posteriori (MAP) _<u>x</u>_ ˆ = argmax _pX|Y_ and MMSE estimate _<u>x</u>_ ˆ = E [ _<u>X|Y</u>_ ]. In the context of BP, these approaches are also known as max-sum and sum-product formulations. We use the sum-product version that we introduced above. To so, we need to particularize Algorithm 2 to the setting of massive MIMO. 

#### **Factor Nonlinear Step** 

The expectation E2 and variance V2 are calculated according to _pZa|Ya_ ( _za|ya_ ) where _pZa|Ya_ ( _za|ya_ ) _/ PY |Z_ ( _ya|za_ ) _· pZa_ ( _za_ ) and _Za ⇠CN_ ( _p_<sup>_t_</sup> _a_<sup>_, ⌧t_)for</sup><sup>_a_= 1</sup><sup>_, . . . , m_.Theoutput</sup> distribution _PY |Z_ ( _ya|za_ ) is given as [8] 

where Φ( _x_ ) = R _x−1_ _~~p~~_ <u>12</u> _⇡_<sup>exp(</sup><sup>_−u_2</sup><sup>_/_2) d</sup><sup>_u_isthecumulativedistributionfunctionofthe</sup> standard normal distribution. The involved expressions for the expectation and variance can be solved in closed form as shown in [9, Ch. 3.9]. 

#### **Variable Nonlinear Step** 

The expectation E1 and variance V1 are calculated according to _pXi|Ri_ ( _xi|ri_ ) _/ pRi|Xi_ ( _ri|xi_ ) _· pX_ ( _xi_ ), where it is assumed _Ri|_ ( _Xi_ = _xi_ ) _⇠CN_ ( _xi, v_<sup>_t_</sup> ). In the considered setting, we have a uniform distribution over the QPSK constellation _X_ . In this special case, a closed form solution can be given as 

For general QAM constellations, the conditional mean E( _r, v_ ) has to be calculated numerically. 

> 4We note that the 1 bit case is an extremal one which will probably not be used in practice. Coarse quantization schemes having three or more bits seem more realistic. However, this assumption allows an easier analytic treatment in the following. 

48 

_3.4.1 System Model_ 

#### **Numerical Results** 

In the following, we show numerical results for an uplink scenario with _n_ = 50 single antenna users and _m_ = 200 antennas at the basestation. The SNR is defined as SNR = 10 log10(1 _/σ_<sup>2</sup> ). 

Figure 3.3: Uncoded BER performance of GAMP for _m_ = 200 and _n_ = 50. 

49 

## **Bibliography** 

- [1] S. Foucart and H. Rauhut, _A Mathematical Introduction to Compressive Sensing_ , Birkh¨auser Basel, 2013. 

- [2] D. L. Donoho, A. Maleki, and A. Montanari, “Message passing algorithms for compressed sensing,” _Proc. US Nat. Acad. Sci._ , vol. 106, no. 45, pp. 1891418919, Nov 2009. 

- [3] A. Maleki, “Approximate message passing algorithms for compressed sensing,” Ph.D. dissertation, Stanford University, Sep 2011. 

- [4] S. Rangan, “Generalized approximate message passing for estimation with random linear mixing,” in _Proc. IEEE Int. Symp. Inf. Theory (ISIT)_ , Jul. 2011, pp. 2168– 2172. 

- [5] S. Rangan, P. Schniter, and A. K. Fletcher, “ector approximate message passing,” _IEEE Trans. Inf. Theory_ , vol. 65, no. 10, pp. 6664 6684, Oct 2019. 

- [6] J. Barbier, F. Krzakala, N. Macris, L. Miolane, and L. Zdeborov´a, “Optimal errors and phase transitions in high-dimensional generalized linear models,” _Proc. US Nat. Acad. Sci._ , vol. 116, no. 12, pp. 54515460, Mar 2019. 

- [7] T. Marzetta, “Noncooperative Cellular Wireless with Unlimited Numbers of Base Station Antennas,” _IEEE Trans. Wireless Commun._ , vol. 9, no. 11, pp. 3590–3600, Nov. 2010. 

- [8] C. Risi, D. Persson, and E. G. Larsson, “Massive MIMO with 1-bit ADC,” _arXiv:1404.7736_ , Apr. 2014. 

- [9] C. E. Rasmussen, _Gaussian processes for machine learning_ . MIT Press, 2006. 

## **4 Unsupervised Learning: Expectation Maximization** 

### **4.1 Introduction** 

In the first part of this course, we considered neural networks as an instance of _supervised learning_ : During the training phase, we provided _features_ (the _<u>xi</u>_<sup>,</sup><sup>_i_=1</sup><sup>_, . . . , N_)and</sup> _labels_ (the _yi_ , _i_ = 1 _, . . . , N_ ) to the optimization procedure to find good parameters (the weights and bias terms) of the imposed model. For certain tasks, _unsupervised learning_ is required, for instance because no feature-label pairs are available. Instead, the training phase has access only to the labels (or observations) but not the features (or inputs). 

### **4.2 Preliminaries** 

Recall that the informational divergence of two probability distributions _PA_ and _PB_ whose domain is the same discrete set _X_ is given by 

Moreover, we have D( _PA||PB_ ) _≥_ 0 with equality if and only if _PA_ = _PB_ . 

To help our development, we use the _log-sum identity_ , see [1, Sec. 1.9.1]. 

_Theorem_ 1 _._ Consider positive _ak_ and non-negative _bk_ for _k_ = 1 _, . . . , K_ , and suppose that at least one of the _bk_ is positive. Let _Sa_ =<sup>P</sup><sup>_K_</sup> _k_ =1<sup>_ak_and</sup><sup>_Sb_= P</sup><sup>_K_</sup> _k_ =1<sup>_bk_,anddefine</sup> _PA_ ( _k_ ) = _ak/Sa_ and _PB_ ( _k_ ) = _bk/Sb_ for _k_ = 1 _, . . . , K_ . We have 

### **4.3 Maximum-Likelihood Estimation** 

We consider maximum likelihood (ML) estimation, where the task is to find the parameters _<u>✓</u> 2_ R<sup>_n_</sup> of a model that best explain the labels _yi, i_ = 1 _, . . . , N_ . The model is commonly given by probability density functions (PDFs) _pYi_ ( _·_ ; _<u>✓</u>_ <u>)</u> or probability mass functions (PMFs) _PYi_ ( _·_ ; _<u>✓</u>_ ). If we further model _Y_<sup>_N_</sup> as being distributed as 

_4.3 Maximum-Likelihood Estimation_ 

then the ML problem is 

where we have used our notational convention to write _p_ ( _yi_ ; _<u>✓</u>_ <u>)</u> for _pYi_ ( _yi_ ; _<u>✓</u>_ <u>).</u> We refer to the factors in (4.4) as _likelihoods_ , and the summands in (4.5) as _log-likelihoods_ . The function _L_ ( _·_ ) is called a log-likelihood function. The expression (4.5) follows from (4.4) by introducing the logarithm, which is a strictly increasing function. Taking logarithms is useful when dealing with exponential models [1] which often appear in practice. 

For example, consider the following wireless communication problem: To setup a communication link, the transmitter and receiver agree on certain parameters that depend on the channel and hardware characteristics. To estimate these parameters, the transmitter sends a sequence of _N pilot symbols_ that the receiver knows. We impose an additive white Gaussian noise (AWGN) model, i.e., we assume that 

where the _Zi_ are independent and identically distributed as _N_ ( _µ, σ_<sup>2</sup> ). We collect the model parameters _✓_ 1 = _µ_ and _✓_ 2 = _σ_<sup>2</sup> into the vector _<u>✓</u>_ and use (4.5) to obtain 

Taking derivatives and equating to zero, we have 

so that 

52 

_4.4 Expectation Maximization_ _<u>(EM)</u> Algorithm_ 

These are indeed the ML estimates because the problem is convex, as can be checked via the Hessian. The receiver now uses these parameter estimates for detection and decoding, e.g., for calculating soft-information for the forward error correction (FEC) decoder. 

Now suppose that we wish to perform the same task but _without_ sending pilots<sup>1</sup> . Suppose we know that the _Xi_ are taken from a discrete and finite set _X_ and have the respective distributions _PXi_ , _i_ = 1 _, . . . , N_ . The receiver has access only to the labels _yi, i_ = 1 _, . . . , N_ and the ML rule is 

Compared to (4.7), ML estimation requires marginalizing over the _latent_ or _hidden_ variables _Xi_ . If we knew the latent variables _Xi_ , then we could group the observations based on the originally transmitted constellation point, and apply supervised learning. This is illustrated in Fig. 4.1 for a 4-ASK constellation. In an unsupervised learning scenario, the receiver has the situation shown in (a) and does not know for a given observation from which this point originates. In situation (b), the receiver can “label” each observation by coloring it appropriately. 

To simplify computation, the idea of expectation maximization (EM) is the following: Break the ML estimation in two parts, namely: 

1. First, calculate a _“soft” assignment_ to the latent variables. 

2. Second, perform the desired _parameter optimization_ . 

The EM algorithm was described by Dempster, Laird and Rubin [6] in 1977. 

### **4.4 Expectation Maximization (EM) Algorithm** 

#### **4.4.1 Evidence Lower Bound (ELBO)** 

Consider the log-likelihood function from (4.11): 

> 1 Such problems are interesting, e.g., for ultra-reliable low latency communication (uRLLC) [2] for internet of things (IoT) applications. Here one wishes to send short packets of symbols, and using pilots might be too wasteful of time, frequency, and energy resources. 

53 

JES 2c RGR Qov (YF Ox i) 

Runwhich y belows tr which Xi — ML (3) (3) \ Sy Btimae which yi bebogs<sup><b</sup> which Xi 

( 

~~_~~ ) 

PamSTE iG Qrah (xy : | nz Pas (YL xis 9) - n> Quit (ly 

_4.4.2 Algorithmic Formulation_ 

#### **4.4.2 Algorithmic Formulation** 

We now summarize the EM algorithm as follows. 

**Algorithm 3** Expectation Maximization 

_t_ = 1 Initialize _<u>✓</u>_ (1) with a good starting value. **while** convergence criterion is not met **do** _E-step_ : Compute 

_M-step_ : Compute 

_t_ = _t_ + 1 **end while** 

We make several remarks. 

- Various approaches can be used to check convergence. For instance, the algorithm can be stopped after a certain number of iterations, the value _k✓_ ( _t_ ) _−_ _<u>✓</u>_ ( _t−_ 1) _k_ is small, or _L_ ( _<u>✓</u>_ ( _t_ )) _− L_ ( _<u>✓</u>_ ( _t−_ 1)) is small. 

- Initializing the EM algorithm with di↵erent starting values _<u>✓</u>_ (1) and choosing the outcome with the highest objective function value may improve performance. 

- The name _E-step_ (for “expectation step”) is not self-explanatory in the above formulation. The terminology comes from [6] where the _E-step_ is written as 

We prefer the presentation in Algorithm 1 as the required computational steps, i.e., computing the _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i|Yi_<sup>(</sup><sup>_·|yi_)andthemaxima,arespecifiedseparately.</sup> 

- The _pYiXi_ ⇣ _yi, x_ ; _<u>✓</u>_ ( _t_ )<sup>⌘</sup> and _PXi|Yi_ ⇣ _x|yi_ ; _<u>✓</u>_ ( _t_ )<sup>⌘</sup> are computed using _PXi_ . 

- We can use the EM algorithm to estimate parameters of _PX_ by including these parameters in the _M-step_ . See Section 4.6.2 below for an example. 

- Other variations on the EM algorithm are discussed in [7]. 

55 

_4.4.3 Convergence Analysis_ 

#### **4.4.3 Convergence Analysis** 

A convergence analysis of the EM algorithm was conducted in [6, Sec. III]. The main results are as follows: 

- The sequence _L_ ( _<u>✓</u>_ ( _t_ )) is non-decreasing in _t_ , i.e., we have _L_ ( _<u>✓</u>_ ( _t_ +1)) _≥ L_ ( _<u>✓</u>_ ( _t_ )) _, 8t_ . To see this, consider the following steps: 

where ( _a_ ) follows by (4.21) and ( _b_ ) follows by (4.15). By induction, we have the desired result. 

- If the EM algorithm converges to _<u>✓</u>_ ( _1_ ), then _<u>✓</u>_ ( _1_ ) is a stationary point. 

- Stationary points are not necessarily the ML estimate _<u>✓</u>_ _~~M~~ L_<sup>,i.e.,wecanhave</sup> _L_ ( _<u>✓</u>_ ( _1_ )) _< L_ ( _<u>✓ML</u>_<sup>).</sup> 

### **4.5 K-Means** 

An algorithm closely related to the EM algorithm is K-Means, a name coined by MacQueen [4] in 1967, though the concept was originally formulated by Steinhaus [3] in 1957. While at Bell Labs, Llyod also formulated K-Means to find an optimal quantizer for pulse code modulation (PCM), but the approach was not published until 1982. 

We begin with the K-means problem formulation and then relate it to our approach in Sec. 4.4. As a _clustering algorithm_ , K-Means aims at solving the following problem: Given a set of _N_ points _<u>y</u>_ _~~i~~_<sup>_2_R</sup><sup>_n, i_=1</sup><sup>_, . . . , N_,wewanttofind</sup><sup>_K_centers</sup><sup>_<u>x</u>_</sup> _j_<sup>_2_R</sup><sup>_n, j_=</sup> 1 _, . . . , K_ and corresponding assignments _δji 2 {_ 0 _,_ 1 _}_ such that the distance of each point to its (representative) center is minimized. We have 

56 

_4.6 Blind Parameter Estimation for Probabilistic Amplitude Shaping_ 

Because of the discrete nature of the assignment variables _δji_ , solving (4.25) directly with convex optimization techniques is not possible. However, we can pursue a two step approach, which first finds the optimal centers and then updates the assignments: 

1. ( _E-step_ replaced by _Decision-step_ ) Given the cluster centers _<u>xj</u>_<sup>_, j_=1</sup><sup>_, . . . , K_,the</sup> optimal assignment _δj_<sup>_⇤_</sup> _i_ for the _i_ -th point _<u>y</u>_ _~~i~~_<sup>isfoundvia</sup> 

2. ( _M-step_ ) Given an assignment _δij_ , we can find the optimal centers by solving for a stationary point _<u>x</u>_ _~~j~~_<sup>as</sup> 

such that 

Comparing (4.26) to the _E-step_ in the EM algorithm, the choice (4.26) corresponds to a _“hard”_ assignment of a point to its cluster, while EM performs a _soft_ -assignment of the _i_ -th observation _yi_ via the auxiliary distribution _QX_ _~~i~~_<sup>_|Y_</sup> _~~i~~_<sup>(</sup><sup>_·|y_</sup> _~~i~~_<sup>).Thesoftassignment</sup> expresses the reliability with which this point resulted from the “cluster center” _<u>xi</u>_<sup>.We</sup> can derive K-Means by using a uniform distribution _PX_ and a Gaussian noise model for the labels, which leads to a Euclidean distance based optimization metric (4.25). 

- K-Means is often used as a robust approach to find initial values for the EM-algorithm. 

### **4.6 Blind Parameter Estimation for Probabilistic Amplitude Shaping** 

Probabilistic amplitude shaping (PAS) [8] is a coded modulation technique that combines the transmission of non-uniformly distributed constellation points with FEC. Recently, PAS gained particular interest for optical communication as it allows to operate close to the Shannon limit and enables transmission rates. 

In this application example, we use the EM algorithm to estimate the parameters of a decoding metric blindly from a set of channel observations. For most scenarios, e.g., metro, long-haul, and transoceanic links, optical receivers perform coherent digital signal processing (DSP) nowadays. The receiver obtains a string _y_<sup>_N_</sup> of noisy symbols from a discrete constellation _X_ , which is used to detect the transmitted string _x_<sup>_N_</sup> . To facilitate analysis, we will consider real channels, or one-dimensional signaling alphabets. 

57 

_4.6.1 Decoder Soft-Information_ 

#### **4.6.1 Decoder Soft-Information** 

State-of-the-art FEC decoders use log-likelihood ratios (LLRs) as soft inputs to decoding algorithms, e.g., belief propagation, BCJR algorithm, Viterbi algorithm. Most FEC codes used in practice are binary. 

For higher-order modulation formats, we associate each constellation point _x_ in a _M_ = 2<sup>_m_</sup> -ary constellation _X_ with a binary label _χ_ ( _x_ ) = **_b_** _2 {_ 0 _,_ 1 _}_<sup>_m_</sup> . Usually, a binary reflected Gray code (BRGC) is used. The FEC decoder inputs are the LLRs 

for _j_ = 1 _, . . . , m_ and _Xj_<sup>_b_=</sup><sup>_{x2X_:[</sup><sup>_χ_(</sup><sup>_x_)]</sup><sup>_j_=</sup><sup>_b}_and</sup><sup>_pY |X_istheimposedstatistical</sup> model on the channel outputs. For an optical channel, an exact description is difficult because of its non-linearities and is subject to ongoing research. 

#### **4.6.2 Imposed Model** 

We employ an AWGN model: 

where the channel input _X_ takes values in the _M_ = 2<sup>_m_</sup> -ary amplitude shift keying (ASK) constellation _X_ . The parameter ∆ _2_ R<sup>+</sup> models the channel gain. The noise term _Z_ is Gaussian distributed with zero mean and variance _σ_<sup>2</sup> , i.e., _Z ⇠N_ (0 _, σ_<sup>2</sup> ). 

However, as a di↵erence to the above formulation, we also wish to determine the parameters of the distribution of _PX_ . A general approach is to consider as parameters all values _PX_ ( _xj_ ) = _pj, j_ = 1 _, . . . , M_ . As a second approach, we will consider a MaxwellBoltzmann (MB) distribution as a proxy for a Gaussian distribution. A MB distribution _PX_ can be specified by a single parameter _⌫_ for which 

The complete model is therefore specified by the parameter vector _<u>✓</u>_ = (∆ _, σ_<sup>2</sup> _, p_ <u>)</u> for a general _PX_ , or by _<u>✓</u>_ = (∆ _, σ_<sup>2</sup> _, ⌫_ ) for a MB distribution, respectively. 

#### **4.6.3 EM Formulation** 

The _E-step_ of Algorithm 1 in Sec. 4.4.2 is straightforward; the only change is that the specified _PX_ ( _·_ ) is replaced by _PX_ ( _·_ ; _<u>✓</u>_ ( _t_ )). For the _M-step_ , we use (4.23) and compute the derivatives with respect to _<u>✓</u>_ = (∆ _, σ_<sup>2</sup> _, p_ <u>).</u> We have 

58 

and therefore 

To optimize with respect to _<u>p</u>_ <u>,</u> we use a Lagrange multiplier to include the constraint P _k_<sup>_pk_= 1:</sup> 

Solving (4.32), (4.33), and (4.34), we have 

If we instead have a MB prior _PX_ , then the distribution parameter _⌫_ is the solution to the following equation: 

59 

## **Bibliography** 

- [1] M. J. Wainwright and M. I. Jordan, “Graphical Models, Exponential Families, and Variational Inference,” _Found. Trends Mach. Learn._ , vol. 1, no. 1-2, pp. 1–305, Jan. 2008. 

- [2] M. Bennis, M. Debbah, and H. V. Poor, “Ultra-Reliable and Low-Latency Wireless Communication: Tail, Risk and Scale,” _arXiv:1801.01270_ , Jan. 2018. 

- [3] H. Steinhaus, “Sur la division des corps mat´eriels en parties,” _Bulletin de l’Acad´emie Polonaise des Sciences, Classe 3_ , vol. 4, pp. 801–804, 1957. 

- [4] J. MacQueen, “Some methods for classification and analysis of multivariate observations,” in _Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics_ . The Regents of the University of California, 1967. 

- [5] G. Kramer, “Lecture notes in Information Theory,” Technical University of Munich, Oct. 2018. 

- [6] A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum Likelihood from Incomplete Data via the EM Algorithm,” _Journal of the Royal Statistical Society. Series B (Methodological)_ , vol. 39, no. 1, pp. 1–38, 1977 

- [7] G. McLachlan and T. Krishnan, _The EM Algorithm and Extensions_ , Wiley Series in Probability and Statistics. Wiley, 2007. 

- [8] G. B¨ocherer, F. Steiner, and P. Schulte, “Bandwidth Efficient and Rate-Matched Low-Density Parity-Check Coded Modulation,” _IEEE Trans. Commun._ , vol. 63, no. 12, pp. 4651–4665, Dec. 2015. 

- [9] F. Steiner, P. Schulte, and G. B¨ocherer, “Blind Decoding-Metric Estimation for Probabilistic Shaping via Expectation Maximization,” in _Proc. Eur. Conf. Optical Commun. (ECOC)_ , Sep. 2018, Paper Th.1H3. 

## **5 Sampling Techniques** 

### **5.1 Introduction** 

Monte Carlo sampling refers to a class of algorithms for obtaining a “good” sequence of random samples from a probability distribution. Sampling is useful for inference problems that require calculating the expected value 

of a function _f_ ( _·_ ) where _<u>X</u>_ is a high dimensional random variable (RV) with PDF _pX_ or PMF _PX_ . For instance, the M-step of the EM algorithm requires computing 

where _<u>X</u> ⇠ QX_ _~~i~~_<sup>_|Y_</sup> ( _·|yi_ ). This can be difficult if the dimension of _<u>X</u>_ is large, e.g., if _<u>X</u>_ has hundreds or thousands of entries<sup>1</sup> . 

The expectation in (5.1) is often dominated by a subset of vectors _<u>x</u>_ . For example, if we can restrict attention to the _<u>x</u>_ with the largest probabilities, and if _PX_ allows computationally efficient _sampling_ , then we can approximate E [ _f_ ( _X_ )] by relying on the weak law of large numbers, see [1, Appendix A.13]. 

_Theorem_ 2 (Weak law of large numbers) _._ Let _Y_ 1 _, . . . YN_ be independent and identically distributed (i.i.d.) real-valued RVs with _PYi_ = _PY , i_ = 1 _, . . . , N_ . The sample mean 

has E [ _SN_ ] = E [ _Y_ ] and Var [ _SN_ ] = Var [ _Y_ ] _/N_ . Tchebyche↵’s inequality gives 

so that _SN_ converges to E [ _Y_ ] in probability for _N ! 1_ . 

Theorem 2 implies that _SN_ is an _unbiased_ estimate of E [ _Y_ ] and the quality of the approximation is proportional to 1 _/N_ and Var [ _Y_ ]. 

> 1For instance, massive multi-input multi-output (MIMO) deploys hundreds of antennas at base stations. The channel matrices for such problems have thousands of entries that need to be estimated. 

_5.1 Introduction_ 

Figure 5.1: Shell in high dimensions. 

Returning to (5.1), we have 

where _<u>x</u>_ ~~1~~<sup>_, . . . , x_</sup> _~~N~~_<sup>aresamplesdrawnfromtheunderlyingPDForPMF.Thevariance</sup> 

can be used to assess the quality of our estimate. 

Unfortunately, many distributions cannot be sampled easily. To illustrate this, consider a ball 

with radius _r_ in _N_ -dimensions. Its volume is proportional to _r_<sup>_N_</sup> . If we now consider a smaller ball with radius _r − "_ and consider the ratio of their volumes, we have 

For fixed _"_ and _n ! 1_ , we see that the fraction vanishes exponentially, see Figure. 5.1. In other words, most of the volume and probability mass is _concentrated in a shell around the sphere_ (shaded in gray in Figure 5.1). Thus, sampling points randomly from _B_ can give inaccurate results, which will be expressed by a large variance. In fact, naive sampling with a small number of points might not provide a point within the shell. 

In this chapter, we study approaches to calculate E [ _f_ ( _<u>X</u>_ <u>)] by providing samples</u> _<u>x</u>_ _~~i~~_<sup>that</sup> improve accuracy. We assume a PMF _PX_ in the following, but extensions to continuous RVs are straightforward. The accuracy depends on the structure of _PX_ . 

In the following, we consider 

_5.2 Importance Sampling_ 

where _Z_ is a normalization constant such that _PX_ is a valid PMF. In statistical physics, _Z_ is referred to as the _partition function_ and calculating _Z_ is often difficult. Indeed, also for our EM example of (5.2), it can be challenging to compute 

We distinguish di↵erent scenarios which give rise to di↵erent algorithms: 

- Sampling from _PX_ is difficult but evaluating _PX_ <u>(</u> _<u>x</u>_ <u>)</u> is possible. 

- Sampling from and evaluating _PX_ is difficult but evaluating _PX_<sup>_⇤_</sup> ( _<u>x</u>_ ) is possible. 

### **5.2 Importance Sampling** 

Importance sampling (IS) considers cases where numerically evaluating _PX_ <u>(</u> _<u>x</u>_ <u>) is possible,</u> but sampling is difficult. We introduce an _auxiliary PMF Q_ that can be computed and sampled efficiently. Suppose that supp( _PX_ ) = supp( _Q_ ) = _X_ . We then have 

where the samples _<u>x</u>_ _~~i~~_<sup>_, i_=1</sup><sup>_, . . . , N_,aredrawni.i.d.from</sup><sup>_Q_.Agood</sup><sup>_auxiliaryPMF_</sup> _Q_ provides samples that contribute the most to the mean. Of course, finding a good biasing distribution _Q_ may be challenging. The ratios<sup>_P_</sup> _Q_<sup>_<u>X</u>_</sup> (( _<u>xx</u>_ _~~ii~~_<sup>))</sup> can be understood as _weights_ , indicating the contribution of the current sample to the overall mean. 

IS is sometimes called a _variance reduction technique_ : a good _Q_ reduces the number of samples for a given accuracy, or improves accuracy for a given number of samples. To see this, consider the variance of (5.11): 

From the above equations, we observe the following [6]. 

63 

_5.2 Importance Sampling_ 

- (5.12) is small if _f_ ( _<u>X</u>_ <u>)</u> _PX_ <u>(</u> _<u>X</u>_ <u>)</u> _−Q_ ( _<u>X</u>_ <u>) E [</u> _f_ ( _<u>X</u>_ <u>)]</u> _⇡_ 0, i.e., when _Q_ ( _<u>X</u>_ <u>) is proportional</u> to _f_ ( _<u>X</u>_ <u>)</u> _PX_ <u>(</u> _<u>X</u>_ <u>).</u> In fact, the variance of the estimator is zero if 

However, this choice is usually not possible because it requires knowing the mean. Nevertheless, it shows that _Q_ ( _<u>x</u>_ ) should be proportional to _f_ ( _<u>x</u>_ <u>)</u> _PX_ <u>(</u> _<u>x</u>_ <u>), which might</u> have been expected. 

- <u>1 (</u> _<u>f</u>_ <u>(</u> _<u>x</u>_ <u>)</u> _PX_ <u>(</u> _<u>x</u>_ <u>))</u><sup>2</sup> 

- _•_ From (5.14), an upper bound on the estimator variance is _N_ P _<u>x2X</u> Q_ ( _<u>x</u>_ <u>)</u> . 

- We obtain computational savings for importance sampling (IS) when (5.14) is smaller than (5.6). 

For extensions of IS, we refer to [6]. For instance, one can adapt IS to the case where the normalization constant (5.9) is not known. The resulting algorithms are called _self normalizing_ IS. 

##### **Example: Calculating Tail Probabilities** 

Consider the average block error probability Pr[ **_U_** = **_U_**<sup>ˆ</sup> ] of transmitting a message **_U_** over a channel _PY |X_ under a random code ensemble with 2<sup>_k_</sup> codewords. A random coding union bound (RCU) is (see e.g. [7]) 

with pairwise error probabilities 

The expression (5.17) is a tail probability of a high dimensional RV. If the tail probabilities are small and we calculate (5.16) via (5.5), we would need many samples for _<u>x</u>_ and _<u>y</u>_ . However, IS can make the calculation feasible. This will be described in a tutorial in more detail by using the method of _exponential tilting_ [6, Sec. 9.3]. 

We study a simpler example. Consider the tail probability Pr[ _Z ≥ a_ ], _a 2_ R<sup>+</sup> , where _Z ⇠N_ (0 _,_ 1). The answer is given by the _Q_ ( _·_ ) function (not to be confused with the auxiliary distribution) or the standard normal distribution Φ( _·_ ) for which 

Both Φ( _·_ ) and _Q_ ( _·_ ) can be computed using numerical algorithms such as series expansions, recurrence relations, continued fraction expansions, numerical quadrature, etc., so one does not need Monte-Carlo (MC) sampling. Nevertheless, to illustrate the method we write 

64 

alse exp| 22% + 2b) > ep} ¥ ab} 

~~<u>= 7 | at da, za) eh | >) £[1 (220) gota) E [| + FI] |</u> ae bh~~ ~~<u>Ne</u> :~~ ~~<u>4 ft (azale | - £[ 1 (2%)6 |</u>~~ 

~~b~~ ~~<u>shift the rf ‘h the Wet siole</u>~~ 

~~<u>Vovlsu “4 Voy] [274)) = 7] (ae 4 Prlz)) | = a ( Prleved- Fe at) Lh shift lawl S27: x Jelawye® [A \270) é]| <a |e Pel274] - smu #) ain of Vorione & > Pal 220] et <4 [Prl27a) -P, [270] Hf ais lave onougk</u>~~ 

_5.3 Rejection Sampling_ 

Figure 5.3: Illustration of rejection sampling. 

### **5.3 Rejection Sampling** 

In contrast to IS, rejection sampling (RS) evaluates _PX_<sup>_⇤_</sup> without the normalization constant. Furthermore, RS uses an auxiliary distribution _Q_ with a constant _c >_ 0 such that (see Figure 5.3) 

_c · Q_ ( _<u>x</u>_ <u>)</u> _≥ PX_<sup>_⇤_</sup> <u>(</u> _<u>x</u>_ <u>)</u> _, 8x._ (5.22) 

As for IS, sampling from _Q_ should be computationally efficient. The RS procedure is summarized in Algorithm 4. 

##### **Algorithm 4** Rejection Sampling 

1: _i_ = 1 

|2: **w**<br>3:|**hile** more samples needed **do**<br>Sample _x_<br>_⇤_from _Q_.||
|---|---|---|
|4:|Sample _u_ from _U_(0_, c · Q_(_x_<br>_⇤_)).|_. U_(0_, A_) denotes the uniform distr. on [0_, A_]|
|5:|**if** _u < P _<sup>_⇤_</sup><br>_X_<br>(_x_<br>_⇤_) **then**||
|6:|_x_<br>_~~i~~_ <sup>=</sup> <sup>_x_</sup><br>_⇤_||
|7:|_i_=_i_+ 1||
|8:|**end if**||
|9: **e**|**nd while**||

To see that RS yields samples from _PX_ , observe that the test in line 5 accepts the current candidate _<u>x⇤</u>_ as a new sample _<u>xi</u>_<sup>withaprobabilityof</sup> 

This is proportional to _PX_<sup>_⇤_</sup> and hence _<u>x⇤</u>_ must be a sample from _PX_ . The efficiency of RS depends on how often the candidates are rejected. 

66 

~~<u>4 Inpe X =2(m-4) £(M-3) +4</u>~~ <u>. Lp -(yp</u> ~~<u>xb Suppo wish 4» compre: Pleiy)= At) lpi ee [ie Pa</u> z~~ ~~<u>P(x)</u> PLY) Sonctins Pst) roby~~ ~~<u>Now we Rane 2) %) --- 2m and can Compute ow =a 3 fexi) %? Ep, Tf) > Qix) Ex Tf -z(u< Px) |</u>~~ 7 ~~_'~~ * ~~<u>aie | x | (0 (x) Cal) f(%) 4</u>~~ <u>{</u> ~~<u>uc Pa mI Rex) x</u> =~~ ~~<u>Pe(x) a P(x)</u> 2 ie~~ ~~<u>| c+) om 2 BY fe) C =), Fe flo oh</u> 2~~ ~~<u>=</u> x~~ ~~<u>nici) nen Ey, L4(v< Pete)]</u> -5 kW5~~ ~~<u>Bf FRM(x) 2</u> x C¢ x C C~~ ~~<u>Go CauLfod - 1 ne Pv | |</u> Ep, Lf» J =| TT Ee Lfoo| d(ue Ps cx>) |~~ ~~<u>Py (Ue Rica)</u>~~ 

—— i ii a a a aS SS eon kev Chain Monte Colo (NCHO Saroling gook: Elfed] =27 = +x) P (xi) | den ) Sue curred. generate 4%, Ay Indepandend(y Memon| le dish1s?ibution] | -Moy Suppose ot have 4 Markov Chaw Poa) Pe; (=wv (Vala APCD) Vistsles) aop)ety, Leal) Poe)=i peix) pe) Tr 0%)<sup>=PQ)%)p helm)Pes</sup> | TS . homonenrous -) in dep ole tnd: Nt LE Aye as Ale ovleut of a Markov Chace Pte) --- PiXw) é uh steily stale dichibution Ve 

i) 

_5.4 Markov Chain Monte Carlo_ _<u>(MCMC)</u> Sampling_ 

### **5.4 Markov Chain Monte Carlo (MCMC) Sampling** 

Markov Chain Monte Carlo (MCMC) methods date back to early works in physics by Metropolis _et al._ [10]. The general idea is to represent the probability distribution from which we want to sample as the ergodic distribution of a Markov chain. In the following, we briefly recapitulate the underlying properties of a Markov chain before discussing two important MCMC algorithms, namely the _Metropolis-Hastings_ and _Gibbs_ sampling algorithms in Sections 5.4.2 and 5.4.3, respectively. 

#### **5.4.1 Markov Chains** 

The RVs _X_ 1 _, X_ 2 _, . . . , Xn_ form a Markov chain if their joint PMF _PX_ factors as 

where _<u>X</u>_ = [ _X_ 1 _, . . . , Xn_ ]. An intuitive interpretation of (5.24) is that each RV _Xi, i_ = 1 _, . . . , n_ , represents a _state_ and the transition probability of going from one state to another depends only on the current state but not the past. Each state _Xi_ may have its own state space _Xi_ . In the following, we consider finite state spaces. For extensions to countably infinite or continuous state spaces, we refer to [13, Sec. 3.3]. 

The Markov chain is specified by the set _{PXi_ +1 _|Xi}i_ =2 _,...,n_ of transition probabilities. If the transition probabilities _PXi_ +1 _|Xi_ are the same for _i_ = 2 _, . . . , n_ , then the Markov chain is _homogeneous_ . The expression _PXi_ +1 _|Xi_ is then called the _transition kernel_ of the Markov chain. Markov chains can be represented with _state diagrams_ . For homogeneous Markov chains, the state diagram for _i_ = 1 describes the Markov chain. 

Figure 5.4: State diagram of a Markov chain with _X_ = _{_ 1 _,_ 2 _,_ 3 _}_ . 

Figure 5.4 shows the state diagram of a homogeneous Markov chain with a ternary state space, i.e., _X_ = _{_ 1 _,_ 2 _,_ 3 _}_ . We list the transition probabilities in a _transition probability matrix_ 

67 

_5.4.1 Markov Chains_ 

The matrix **_P_** is a stochastic matrix [9, Sec. 10.3], i.e., the entries are non-negative and the column entries sum to one. Similarly, we call a vector stochastic if all its entries are non-negative and sum to one. We investigate properties of such matrices in detail in one of the tutorials. 

The probability _PXn_ ( _xn_ ) depends only on the initial distribution _PX_ 1 and the set _{PXi|Xi−_ 1 _}i_ =2 _,...,n_ . of transition probabilities. 

For Markov chains with _K_ -ary state spaces, we define 

so that (5.28) can be written as 

For homogeneous Markov chains, this reduces to 

A Markov chain is _irreducible_ if it is possible to get to any state from any state or more formally, if we have 

Important characteristics of an irreducible Markov chain are its _invariant (or stationary) distributions QX_ , which are defined such that 

From the right hand side of (5.33), an invariant distribution of a Markov chain is an eigenvector of the transition matrix for an eigenvalue of one. A finite Markov chain always has at least one invariant distribution. For homogeneous Markov chains, the conditions of (5.33) reduce to a single condition. For example, the transition matrix (5.25) has the invariant distribution 

68 

_5.4.2 Metropolis-Hastings Sampling_ 

Next, a Markov chain is _ergodic_ with _limiting distribution P1_ ( _x_ ) if 

Clearly, _P1_ ( _x_ ) must be an invariant distribution of the Markov chain. Ergodic Markov chains have a _unique_ invariant distribution, which is also referred to as the _equilibrium distribution_ . We refer to [13, Sec. 3.3] for more details. 

#### **5.4.2 Metropolis-Hastings Sampling** 

Metropolis _et al._ described an approach to create the states of a physical system with a Boltzmann distribution [10]. In 1970, Hastings extended this idea to arbitrary distributions [11]. As for RS in Sec. 5.3, the Metropolis-Hastings (MH) algorithm requires to evaluate only _PX_<sup>_⇤_</sup> and not _PX_ . 

Algorithm 5 describes the procedure. We begin by choosing an initial sample _<u>x</u>_ . We _0_ then use a sampling (or _proposal_ ) distribution _Q_ ( _·|x_ <u>)</u> to choose a candidate sample _<u>x</u>_ . The candidate sample is accepted with probability 

The original Metropolis algorithm considered symmetric sampling distributions with _Q_ ( _<u>x|x0</u>_ ) = _Q_ ( _<u>x0|x</u>_ <u>) for all</u> _<u>x</u>_ <u>,</u> _<u>x0</u>_ , so that the ratio in (5.36) is _P ⇤X_ <u>(</u> _<u>x0</u>_ ) _/PX ⇤_ <u>(</u> _<u>x</u>_ <u>).</u> For example, common choices for _Q_ ( _·|·_ ) include Gaussian, Cauchy and uniform distributions centered on _<u>x</u>_ <u>.</u> An analysis of the performance of di↵erent sampling distributions is given in [12]. 

The transition probability of the MH algorithm is 

where the _<u>x0</u>_ = _<u>x</u>_ terms reflect the probability of accepting a new state _<u>x0</u>_ . The choice (5.37) has _PX_<sup>_⇤_</sup> as an invariant distribution, as can be checked by inserting (5.36) into (5.37). 

Several variants of the MH algorithm can be formulated. For example, _single component MCMC_ updates only one entry of _<u>xi</u>_<sup>toobtain</sup><sup>_<u>x</u>_</sup> _i_ +1<sup>forall</sup><sup>_i_.Thisapproachcanbe</sup> computationally efficient [14]. The acceptance probability (5.36) is modified as 

#### **5.4.3 Gibbs Sampling** 

Gibbs Sampling is a single component MH algorithm [15]. Gibbs sampling uses a distribution _PX_<sup>_⇤_</sup> and samples from the conditional distributions _PX_<sup>_⇤_</sup> _i|X_ _~~⇠~~ i_<sup>=</sup><sup>_P ⇤_</sup> _Xi|X_ 1 _...Xi−_ 1 _Xi_ +1 _...Xn_<sup>,</sup> i.e., the sampling distributions are 

69 

~~<u>=</u>~~ XK FX4 ~~<u>P(x): Q (xia LX) d(xal ¥) + — Pelxin) [4 2</u>~~ XEXi-+ ~~<u>AR Kin) QI Kin) | =</u> = Aix~~ ~~<u>Qlwalk)al ming</u>~~ <u>.</u> ~~<u>4,</u> Fehee QUE]Ua) Xe) +~~ <u>. K+ Kien</u> ~~<u>{ P*cx) Q(nn</u>~~ <u>|</u> ~~<u>X) bk</u>~~ <u>(xis!)</u> ~~X#Xia % _~~ <u>P* Cain) QUE | Xie)</u> ~~<u>br rnin ¢ fx(@) Oxnlt)_, B Over) AO in) AO in) in) + fe Ova) —_ min Px (xin) QUE Xt) > POR) Q Cre Fay</u> THKi4~~ ~~<u>equoul</u>~~ 

~~<u>ia br rnin ¢ fx(@) Oxnlt)_, B Over) AO in) AO in) in) + fe Ova)</u>~~ 

_5.4.4 Example: Conditional Mean_ 

**Algorithm 5** Metropolis-Hastings Sampling 

1: Choose _x_<sup>(0)</sup> at random. 

2: Choose sampling distribution _Q_ ( _·|·_ ). 

3: _t_ = 1 

4: **while** more samples needed **do** 

5: Sample _x_<sup>_⇤_</sup> from _Q_ ( _·|x_<sup>(</sup><sup>_t_)</sup> ) 6: Calculate the acceptance probability _P_<sup>_⇤_</sup> _<u>X</u>_<sup><u>(</u></sup><sup>_x⇤_</sup><sup><u>)</u></sup><sup>_<u>Q</u>_</sup><sup><u>(</u></sup><sup>_x_(</sup><sup>_t−_1)</sup><sup>_<u>|x⇤</u>_</sup><sup><u>)</u></sup> _↵_ ( _x_<sup>_⇤_</sup> _|x_<sup>(</sup><sup>_t_)</sup> ) = min 1 _, P_<sup>_⇤_</sup> _X_<sup>(</sup><sup>_x_(</sup><sup>_t−_1))</sup><sup>_Q_(</sup><sup>_x⇤|x_(</sup><sup>_t−_1))</sup> ! 7: Sample _u_ from _U_ (0 _,_ 1). _. U_ (0 _,_ 1) denotes the uniform distr. on [0 _,_ 1] 8: **if** _u < ↵_ ( _x_<sup>_⇤_</sup> _|x_<sup>(</sup><sup>_t_)</sup> ) **then** 9: _x_<sup>(</sup><sup>_t_+1)</sup> = _x_<sup>_⇤_</sup> 10: **else** 11: _x_<sup>(</sup><sup>_t_+1)</sup> = _x_<sup>(</sup><sup>_t_)</sup> 12: **end if** 13: _t_ = _t_ + 1 14: **end while** 

Using (5.39) in (5.38), the acceptance probability becomes one causing each candidate to be accepted. The problem of finding appropriate kernels is thus resolved. Algorithm 6 summarizes the procedure. 

Note that the samples _x_<sup>(</sup> _j_<sup>_t_)</sup><sup>_, j_= 1</sup><sup>_, . . . ,_(</sup><sup>_i −_1)areusedforcalculating</sup><sup>_x_(</sup> _i_<sup>_t_).</sup> 

Many important issues for efficient implementation of MCMC cannot be covered in the lectures. Some of the most important issues are the following. 

- How should the sampling procedure be _initialized_ ? 

- How many samples should be _neglected_ during initialization? 

- How can the _steady state be tracked_ ? 

We refer to the tutorials and [13, Sec. 6.3] for more details. 

#### **5.4.4 Example: Conditional Mean** 

Consider a MIMO system with _N_ t transmit and _N_ r receive antennas over a frequency flat fading channel _<u>H</u> 2_ R<sup>_N_r</sup><sup>_⇥N_t</sup> . The channel model is 

where _<u>X</u> ⇠N_ (0 _<u>, σX</u>_<sup>2</sup><sup>_<u>I</u>_</sup> <u>)</u> and _<u>Z</u> ⇠N_ (0 _<u>, σZ</u>_<sup>2</sup><sup>_<u>I</u>_</sup> <u>).</u> The entries of _<u>H</u>_ are i.i.d. _N_ (0 _,_ 1), and _<u>H</u>_ is known to the receiver. 

Suppose we want to compute the minimum mean-squared error estimate E ⇥ _<u>X|Y</u>_ = _<u>y</u>_ <u>⇤.</u> A MCMC approach is not needed, as there is a closed form expression 

_5.4.4 Example: Conditional Mean_ 

**Algorithm 6** Gibbs Sampling 

- 1: Choose _<u>x</u>_ (0) at random. 

2: _t_ = 1 

- 9: **end while** 

However, this problem helps to become familiar with the MCMC approach and provides a solution that we can reproduce. 

We first derive an expression for _pXk|Y_ _<u>, X</u>_ _~~⇠~~ k_<sup>(</sup><sup>_xk|y_</sup> _<u>, x⇠k</u>_<sup>).</sup> Observe that _Xk|y, x_ _~~⇠~~ k_<sup>is</sup> Gaussian with 

where _<u>A</u>_ = � _<u>Y</u> T_ _<u>X</u> T_ _~~⇠~~ k_ �T, and the covariance expressions are **_C_** _XkA_ = E ⇥ _Xk_ � _<u>Y</u>_ T _<u>X</u>_ T _~~⇠~~ k_ �⇤ = E h _Xk_ ⇣(<sup>P</sup><sup>_N_</sup> _i_ =1<sup>t</sup><sup>_<u>h</u>_</sup> _~~i~~_<sup>_Xi_+</sup><sup>_<u>Z</u>_</sup> )<sup>T</sup> _<u>X</u>_ T _~~⇠~~ k_ ⌘i = � _σx_<sup>2</sup><sup>_<u>h</u>_</sup> T _~~k~~_ **0** 1 _⇥_ ( _N_ t _−_ 1)� 

where _<u>hi</u>_<sup>isthe</sup><sup>_i_-thcolumnof</sup><sup>_<u>H</u>_</sup> and 

The individual submatrices appearing in (5.42) are 

where _<u>H</u>_ _~~⇠~~ k_<sup>isthematrixformedbyremovingthe</sup><sup>_k_-thcolumnof</sup><sup>_<u>H</u>_</sup> <u>.</u> 

From these expressions, we can sample from the Gaussian PDFs (line 4, 5, 7 in Algorithm 6) with mean and variance given in (5.40) and (5.41). A demo illustrating the Gibbs sampling process for this example can be accessed on Google Colab<sup>2</sup> . 

> 2 `https://colab.research.google.com/drive/1xhNjcFpwwjfR1SIl0AtuDh-RaF8ncG88` 

71 

## **Bibliography** 

- [1] G. Kramer, “Information Theory Lecture Notes,” Technical University of Munich, Oct. 2018. 

- [2] L. Dolecek, Z. Zhang, M. Wainwright, V. Anantharam, and B. Nikolic, “Evaluation of the Low Frame Error Rate Performance of LDPC Codes Using Importance Sampling,” in _Proc. IEEE Inf. Theory Workshop (ITW)_ , Sep. 2007, pp. 202–207. 

- [3] S. Iyengar, “Importance Sampling for Tail Probabilities,” Stanford University, Department of Statistics, Tech. Rep., 1991. 

- [4] J. Hammersley, _Monte Carlo Methods_ , ser. Monographs on Statistics and Applied Probability. Springer Netherlands, 1964. 

- [5] R. Rubinstein and D. Kroese, _Simulation and the Monte Carlo Method_ , ser. Wiley Series in Probability and Statistics. Wiley, 2011. 

- [6] A. B. Owen, _Monte Carlo Theory, Methods and Examples_ , 2013. 

- [7] Y. Polyanskiy, H. V. Poor, and S. Verdu, “Channel Coding Rate in the Finite Blocklength Regime,” _IEEE Trans. Inf. Theory_ , vol. 56, no. 5, pp. 2307–2359, May 2010. 

- [8] A. Doucet and X. Wang, “Monte Carlo methods for signal processing: A review in the statistical signal processing context,” _IEEE Signal Process. Mag._ , vol. 22, no. 6, pp. 152–170, Nov. 2005. 

- [9] G. Strang, _Introduction to Linear Algebra_ . Wellesley-Cambridge Press, 2016. 

- [10] N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller, “Equation of State Calculations by Fast Computing Machines,” _The Journal of Chemical Physics_ , vol. 21, no. 6, pp. 1087–1092, Jun. 1953. 

- [11] W. K. Hastings, “Monte Carlo Sampling Methods Using Markov Chains and Their Applications,” vol. 57, no. 1, pp. 97–109, 1970 

- [12] J. Geweke and H. Tanizaki, “Note on the Sampling Distribution for the MetropolisHastings Algorithm,” _Communications in Statistics - Theory and Methods_ , vol. 32, no. 4, pp. 775–789, Jan. 2003. 

- [13] R. M. Neal, “Probabilistic inference using Markov chain Monte Carlo methods,” University of Toronto, Department of Computer Science, Tech. Rep. CRG-TR-93-1, 1993. 

_Bibliography_ 

- [14] W. Gilks, S. Richardson, and D. Spiegelhalter, _Markov Chain Monte Carlo in Practice_ , ser. Chapman & Hall/CRC Interdisciplinary Statistics. Taylor & Francis, 1995. 

- [15] S. Geman and D. Geman, “Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. PAMI-6, no. 6, pp. 721–741, Nov. 1984. 

73 

## **6 Dimensionality Reduction** 

### **6.1 Introduction** 

Dimensionality reduction is the process of mapping high dimensional data into lower dimensional data while preserving information. Consider the _M_ -dimensional data points _<u>x</u>_ ~~1~~<sup>_, x_</sup> ~~2~~<sup>_, . . . x_</sup> _~~N~~_<sup>_2_R</sup><sup>_M_andatargetdimension</sup><sup>_K_with</sup><sup>_K<M_.Dimensionalityreduction</sup> is performed by an encoder 

that maps every data point to a lower dimensional representation, and a decoder 

that approximates the given data point. We will choose _f_ and _g_ to minimize the approximation error 

The choice of cost function, such as the sample squared error above, will of course depend on the application. Dimensionality reduction is used for many problems. 

- **Visualization:** By choosing _K_ = 2 or _K_ = 3, we can visualize a high dimensional data set and identify similar data points. 

- **Compression:** Since _K < M_ , we are compressing the data. Efficient _digital_ compression usually applies a dimensionality reduction algorithm followed by a quantizer. Many image compression algorithms work in this fashion. 

- **Speeding up Learning Algorithms:** Consider training a neural network on very high dimensional data. The input layer to the network has the same dimension as the data, even if the data has some intrinsic lower dimensional structure. When training a high dimensional network is infeasible, one can first reduce the dimension of the training data and then train the network with the lower dimensional data. 

- **Noise Reduction:** If the data has a low dimensional structure but is subject to high dimensional noise, an adequate dimensionality reduction algorithm can reduce the noise prior to further processing. 

_6.2 Principal Component Analysis_ 

### **6.2 Principal Component Analysis** 

Principal Component Analysis (PCA) is a widely used tool to choose a linear projection into a subspace. The goal is to minimize the error (6.3). We will see that the solution maximizes the variance of the projected data, which is intuitively pleasing because this suggests that we are capturing most of the information of the original data. 

Suppose we are given a data set with _N_ column vectors _<u>x</u>_ ~~1~~<sup>_, x_</sup> ~~2~~<sup>_, . . . , x_</sup> _~~N~~_<sup>_2_R</sup><sup>_M_.We</sup> model the data as being sampled independently from the same _unknown_ distribution, say that corresponds to the random vector _<u>X</u>_ <u>.</u> Every data point _<u>x</u>_ _~~i~~_<sup>contains</sup><sup>_M_di↵erent</sup> _features_ or _measurements_ of a random process. We would like to find a linear encoder from R<sup>_M_</sup> to R<sup>_K_</sup> , given by the matrix **Q** , to create the low dimensional vectors 

Intuitively, we would like the entries of _<u>y</u>_ _~~i~~_<sup>tobeuncorrelated,whichisachievedby</sup> requiring the columns of **Q** to be orthogonal. We further normalize the columns so that **Q**<sup>T</sup> **Q** = **I** _K_ , where **I** _K_ is the _K ⇥ K_ identity matrix. From the _<u>y</u>_ _~~i~~_<sup>,wecanapproximate</sup> the original data via 

where we defined the projection matrix **P** := **QQ**<sup>T</sup> . Observe that **P**<sup>2</sup> = **P** = **P**<sup>T</sup> **P** . This property of a projection matrix is also known as _idempotence_ . Now define the sample mean and covariance matrix as the respective 

The normalization by _N −_ 1 makes the covariance estimate unbiased, i.e., E[ **C** _X_ ] is equal ˜ to the true covariance matrix. To see this, define _<u>xi</u>_<sup>=</sup><sup>_<u>x</u>_</sup> _~~i~~_<sup>_−_E[</sup><sup>_<u>X</u>_</sup> <u>] and</u> _<u>X</u>_<sup>˜</sup> = _<u>X</u> −_ E[ _<u>X</u>_ <u>] and</u> compute 

We remark that some authors normalize by _N_ rather than _N −_ 1, and this gives the maximum likelihood (ML) estimate of the covariance matrix for Gaussian processes. For large _N_ , the two estimators are almost the same. 

75 

~~<u>4. 1 am Why</u>~~ <u>is</u> ~~<u>WHT 2 (4</u>~~ <u>- Mi) 0s</u> ~~<u>my"</u>~~ <u>unbiased HG = Covlya)-{</u> ~~<u>a}-6057)|</u>~~ <u>Xi -£042)' =0 because independ 2X yt steer Nx LY le Sampling</u> ~~Wek ais wake wey Mi = y Xi~~<sup>~~4~~</sup> ~~=~~ ~~<u>Sewv GE)got)</u> _~~ <u>-</u> ~~<u>FeLRROmCal</u>~~ <u>- 4 er</u> ~~alatay~~ ~~<u>+ ete 81]wd</u> Von n -l~~ ~~<u>> aa § elas") | 4-) ~ ay</u> a2~~ ~~<u>EL -ELA) (x-E0K1)- = E[ee-E0n[x £000</u>~~ **~~<u>)</u>~~** ~~<u>]</u> 2. Roof of PCA let % = X-EDX], 8 is the eotimation of~~ <u>Ki,</u> ~~Siz PH We Want tD minimize 42 a- GIF <GZ ls -Pall= GSI GPS = WE KM e-PY (n-P)%~~ ~~<u>= 0%, S"(In-P-P+PP)%</u>~~ <u>7</u> ~~~~~ ~~<u>=</u>~~ <u>Io</u> ~~<u>72</u>~~ <u>a/</u> ~~<u>[ x78mate - LEK)v</u> ~~~ <u>=</u> ~~<u>Leon(FR</u> -(peye))~~ <u>~ a ~</u> ~~= welll’~~ <u>14-2</u> ~~-~~ ~~<u>7</u>~~ <u>12</u> ~~I Ps~~ <u>~ {2</u> ~~Constant mini mize qe Il -¥ | i Ay Waximizea~~ ~~<u>We</u> | Ps | » ~ Max WI 2Woe~~ <u>lI</u> ~~P&I] |~~ <u>~~ us 2</u> ~~war a 2~~ <u>|]</u> ~~PL xi ~Erx)]I tet = oar PAIS~~ ~~<u>re wax at (0 EL PP LW ~F0a7) E Ata = 14d”) e174) 1</u> a maxjrs 2, byl (ke -Em7)'8 87x -61«)) a max ty (6° WA = (x1-E007) [xi -fa1 0) ~ wax ty oT Cx ) wv max le wAw'B) IS maximized when B= (41 go -- gk) = Cm -- We)~~ 

_6.2.1 Derivation of PCA_ 

The goal of PCA is thus to the find a **Q** with **Q**<sup>T</sup> **Q** = **I** _K_ that minimizes (6.3), namely 

where we have used ( **I** _M −_ **P** )<sup>T</sup> ( **I** _M −_ **P** ) = **I** _M −_ **P**<sup>T</sup> **P** . The first summand is a constant, so we can alternatively _maximize_ the sample second moment 

#### **6.2.1 Derivation of PCA** 

From (6.10), we wish to maximize 

where we have normalized by _N −_ 1 rather than _N_ . We compute 

where we have used tr ( _AB_ ) = tr ( _BA_ ). We thus have 

76 

_6.2.2 Singular Value Decomposition_ 

Using a theorem by Ky Fan [1], the above expression is maximized when the columns of **Q** are the eigenvectors of **C** _X_ corresponding to its _K_ largest eigenvalues. Consider the eigendecomposition of the symmetric covariance matrix [7, 5.3.1] 

where **W** _2_ R<sup>_M⇥M_</sup> is an orthogonal matrix and **⇤**<sup>_M⇥M_</sup> is a non-negative diagonal matrix with the eigenvalues of **C** _X_ , denoted by _λi_ ( **C** _X_ ), _i_ = 1 _,_ 2 _, . . . , M_ , on its diagonal. For convenience, suppose the _λi_ ( **C** _X_ ) are non-increasing: _λ_ 1( **C** _X_ ) _≥ λ_ 2( **C** _X_ ) _≥ . . . ≥ λM_ ( **C** _X_ ). We write _<u>w</u>_ _~~i~~_<sup>fortheeigenvectorcorrespondingto</sup><sup>_λi_(</sup><sup>**C**</sup><sup>_X_)andwechoose</sup> 

Inserting (6.15) and (6.14) into (6.13), we have 

where we used **W** _K_<sup>T</sup><sup>**W**= [</sup><sup>**I**</sup><sup>_K_</sup><sup>**0**</sup><sup>_K⇥_(</sup><sup>_M−K_)]and</sup><sup>**0**</sup><sup>_K⇥L_isthe</sup><sup>_K ⇥L_all-zerosmatrix.</sup> _Remark:_ There is more freedom in choosing **Q** . Consider the matrix **QR** , where **R** is a _K ⇥ K_ orthogonal matrix for which **R**<sup>T</sup> **R** = **RR**<sup>T</sup> = **I** _K_ . One can check that (6.13) is not changed by replacing **Q** with **QR** : 

Thus, we can choose **Q** = **W** _K_ **R** for any orthogonal matrix **R** . 

#### **6.2.2 Singular Value Decomposition** 

The above approach to PCA requires computing **C** _X_ followed by an eigenvalue decomposition of **C** _X_ . Instead, it can be simpler to operate directly on the data matrix **_X_** = [ _<u>x</u>_ ~~1~~ _. . ._ _<u>x</u>_ _~~N~~_<sup>].Toseethis,considerthesingularvaluedecomposition(SVD)[7,</sup> 5.3] of the shifted data 

where 

- **1** _N_ is the _N_ -dimensional all-ones column vector 

77 

_6.2.3 Pre-Processing_ 

- **U** is an _M ⇥ M_ orthogonal matrix 

- **V** is an _N ⇥ N_ orthogonal matrix 

- **⌃** is an _M ⇥ N_ diagonal matrix with the singular values _σi_ ( **_X_** _−_ _<u>µ</u>_ **1**<sup>T</sup> _N_<sup>) of</sup><sup>**_X_**</sup><sup>_−_</sup><sup>_<u>µ</u>_</sup> **1**<sup>T</sup> _N_ on the diagonal. 

We continue by writing **C** = **C** _X_ and _<u>µ</u>_ = _<u>µX</u>_<sup>.Wehave</sup> 

where the last equality follows from (6.14). Hence, we have **U** = **W** and 

This observation is useful since the SVD is, in general, numerically more stable than the eigenvalue decomposition. 

#### **6.2.3 Pre-Processing** 

We derived the PCA algorithm by maximizing the empirical variance of our projected data. In many cases, however, the dynamic ranges of the di↵erent features vary over several orders of magnitude. Consider, for example, a data set containing house prices in euro and house sizes in square meters. Since a square meter costs several thousand euro, absolute variations in size will look negligible compared to absolute variations in price. Since PCA is sensitive to absolute variations and means, we usually pre-process our features to be on the same scale. 

A typical scaling normalizes every feature to have unit variance, so that PCA treats variations of a feature equally across all features. Of course, this step can be omitted if we know that the features have a similar dynamic range. If, for example, every feature represents a pixel in a gray scale image, all elements of our data points are in the set _{_ 0 _, . . . ,_ 255 _}_ . In this case, it can be undesirable to rescale every pixel to unit variance as this might amplify small variations (if one pixel has a similar brightness across all images). 

#### **6.2.4 Choosing the Dimension** _K_ 

One advantage of PCA is that the only parameter we need to choose is the target dimension _K_ . If the goal is visualization, _K_ = 2 (or even _K_ = 3) is the most reasonable choice. Otherwise, there are mainly two intuitive ways for choosing _K_ . 

78 

_6.2.5 Algorithm Description_ 

Figure 6.1: The largest 200 of 784 eigenvalues for the MNIST training set. 

One approach is to plot the eigenvalue profile, called _scree plot_ , of **C** and identify a good _K_ . If the data has a low dimensional structure, the eigenvalues will decrease quickly, as shown in Figure 6.1 for the _MNIST dataset_ of handwritten digits. 

A second and more objective method is to set a target fraction _↵_ (e.g., 95% or 99%) of the variance to be preserved by PCA. Using Eq. (6.16), the required target dimension can conveniently be computed as 

#### **6.2.5 Algorithm Description** 

We summarize the PCA algorithm. Given the data matrix **_X_** having as columns the _N_ data samples _<u>x</u>_ ~~1~~<sup>_, . . . , x_</sup> _~~N~~_<sup>,wefixthedimension</sup><sup>_K_oftheprojecteddataorafractionof</sup> variance to be preserved. 

79 

_6.2.5 Algorithm Description_ 

**Algorithm 7** Principle Component Analysis 

1: data matrix **_X_** _2_ R<sup>_M⇥N_</sup> , target dimension _K_ or target conserved variance _↵_ 2: Centering and scaling: 3: _<u>µ</u> N_<sup><u>1</u></sup> P _Ni_ =1<sup>_<u>x</u>_</sup> _~~i~~_ 4: **_X X_** _−_ _<u>µ</u>_ **1**<sup>T</sup> _N_ 5: **_X_** scaled version of **_X_** _._ proper rescaling if necessary 6: Compute eigenvectors and eigenvalues of **C** : 7: _<u>w</u>_ ~~1~~<sup>_, . . . , w_</sup> _~~M~~_<sup>eigenvectorsof</sup><sup>**C**</sup> _._ use numerically 8: _λ_ 1 _, . . . , λj_ eigenvalues of **C** _._ appropriate method 9: **if** _↵_ given instead of _K_ **then** 10: Compute _K_ : 11: _K_ = min _K_<sup>_0_</sup> _2_ N : <u>P</u> _KiN_ <u>=1</u> _0_<sup>_λi_</sup><sup><u>(</u></sup><sup>**C**</sup><sup><u>)</u></sup> ⇢ ~~P~~ _i_ =1<sup>_λi_(</sup><sup>**C**)</sup><sup>_≥↵_</sup> � 

12: **end if** 13: Dimensionality reducing matrix: 14: **W** _K_ = [ _<u>w</u>_ ~~1~~<sup>_, w_</sup> ~~2~~<sup>_, . . . , w_</sup> _~~K~~_<sup>]</sup> 15: Create lower dimensional data: 16: **_Y_ W**<sup>T</sup> _K_<sup>**_X_**</sup> 17: projected data **_Y_** PCA matrix **W** _K 2_ R<sup>_M⇥K_</sup> 

80 

_6.3 Probabilistic PCA_ 

### **6.3 Probabilistic PCA** 

We motivated PCA as a method to reduce the dimension of high dimensional data by an orthogonal projection. In this section, we view dimensionality reduction from a _generative_ perspective. We model a data point _<u>xi</u>_<sup>_2_R</sup><sup>_M_asbeingcreatedbydrawinga</sup> lower dimensional latent variable _<u>z</u>_ _~~i~~_<sup>_2_R</sup><sup>_K_fromaGaussiandistribution.Thevariable</sup> _<u>z</u>_ _~~i~~_<sup>isthenlinearlytransformedintotheobservationspace,shiftedbyameanvalue,and</sup> perturbed by Gaussian noise. We thus model each observed data point _<u>xi</u>_<sup>as a realization</sup> of the Gaussian model 

where _<u>Zi</u>_<sup>_⇠N_(0</sup><sup>_,_</sup><sup>**I**</sup><sup>_K_),</sup><sup>**Q**</sup><sup>_2_R</sup><sup>_M⇥K_,</sup><sup>_<u>m</u>_</sup> _2_ R<sup>_M_</sup> , and _<u>V</u>_ _~~i~~_<sup>_⇠N_(0</sup><sup>_, σ_2</sup><sup>**I**</sup><sup>_M_)ishighdimensional</sup> i.i.d. Gaussian noise. 

Our goal is now to find the ML estimates for **Q** _, m, σ_<sup>2</sup> based on the set of observations _<u>x</u>_ ~~1~~<sup>_, . . . , x_</sup> _~~N~~_<sup>.Oncethisisaccomplished,dimensionalityreductionwith</sup><sup>_probabilistic_</sup> PCA, or PPCA, is performed by computing the minimum mean squared error (MMSE) estimate 

where we applied the (very useful) matrix identity 

This identity lets one reduce a _M ⇥ M_ matrix inversion to a _K ⇥ K_ matrix inversion (recall that _K < M_ ). Note that since _<u>Zi</u>_<sup>and</sup><sup>_<u>V</u>_</sup> _i_<sup>,1</sup><sup>_iN_,areindependentand</sup> indentically (i.i.d) distributed, the samples _<u>X</u>_ ~~1~~<sup>_, . . . , X_</sup> _~~N~~_<sup>arealsoindependent.</sup> 

#### **6.3.1 Maximum Likelihood Estimation** 

For notational convenience, we denote the parameters to be determined by _<u>✓</u>_ = _{_ **Q** _, m, σ_<sup>2</sup> _}_ . Given the observations _<u>x</u>_ ~~1~~<sup>_, . . . , x_</sup> _~~N~~_<sup>,thelikelihoodfunctionis</sup> 

where _p✓_ <u>(</u> _<u>x</u>_ _~~i~~_<sup>)denotestheprobabilitydensityfunction(pdf)ofany</sup><sup>_<u>x</u>_</sup> _~~i~~_<sup>computedvia</sup> MMSE estimation of the latent variable 

_6.3.1 Maximum Likelihood Estimation_ 

Since _<u>Z</u>_ _~~i~~_<sup>and</sup><sup>_<u>V</u>_</sup> _~~i~~_<sup>areindependentGaussianvectors,by(6.22)</sup><sup>_<u>X</u>_</sup> _~~i~~_<sup>isalsoGaussianwith</sup> 

where the second last line follows since _<u>V</u>_ _~~i~~_<sup>and</sup><sup>_<u>Z</u>_</sup> _~~i~~_<sup>areindependent.</sup> 

_Remark:_ The marginal _p✓_ <u>(</u> _<u>x</u>_ _~~i~~_<sup>) is determined by the parameters</sup><sup>**Q**,</sup><sup>_<u>m</u>_</sup> _, σ_<sup>2</sup> . As for PCA, there˜ is an ambiguity because _<u>Z</u>_ _~~i~~_<sup>hasanisotropic1distribution.Toseethis,consider</sup> **Q** = **QR** where **R** is an orthogonal matrix. The covariance term from the latent variable in (6.28) then becomes 

and thus **C** is independent of **R** . Therefore, there is a whole family of matrices **Q**<sup>˜</sup> - parameterized by all possible rotations in the latent variable space - that gives rise to the same distribution in the observation space R<sup>_M_</sup> . 

_Remark:_ Evaluating the distribution of _<u>X</u>_ _~~i~~_<sup>involvesevaluating</sup><sup>**C**</sup><sup>_−_1.UsingaWood-</sup> bury identity [7, Eq. (166)], this inverse can be computed via 

where **M** _2_ R<sup>_K⇥K_</sup> is given by 

Since _K < M_ , inverting **M** is usually simpler than inverting **C** . We can now write the log-likelihood function as 

> 1We say that a random vector has an _isotropic_ probability distribution if its covariance matrix is a scaled identity matrix 

82 

_6.3.1 Maximum Likelihood Estimation_ 

Di↵erentiating with respect to _<u>m</u>_ <u>,</u> the maximum is attained for 

Inserting _<u>m</u>_ ~~M~~ L<sup>=</sup><sup>_<u>µ</u>_</sup> into (6.32), the quadratic term can be rewritten as 

where we defined the ML estimate **C** ML = _N_ <u>1</u> P _Ni_ =1<sup>(</sup><sup>_<u>x</u>_</sup> _~~i~~_<sup>_−_</sup><sup>_<u>µ</u>_</sup> <u>)(</u> _<u>x</u>_ _~~i~~_<sup>_−_</sup><sup>_<u>µ</u>_</sup> <u>)</u><sup>T</sup> . Using (6.33) and (6.34), we have 

Maximizing with respect to **Q** and _σ_<sup>2</sup> is more involved, but a closed-form solution exists. We again consider the eigendecomposition **C** ML = **W⇤W**<sup>T</sup> and assume that the eigenvalues are ordered to be non-increasing. Let **⇤** _K_ be the diagonal matrix with the _K_ largest eigenvalues of **C** ML, and let **W** _K 2_ R<sup>_M⇥K_</sup> be the matrix with the corresponding eigenvectors. The ML estimates for **Q** and _σ_<sup>2</sup> are the respective [4] 

Observe that the noise variance is the average variance of the discarded dimensions. Observe also that **Q** ML consists of the _K_ principal eigenvectors of **C** ML, each scaled by _λi − σ_ ML<sup>2.Thus,thevarianceinthedirectionofaneigenvector</sup><sup>_<u>w</u>_</sup> _~~i~~_<sup>,1</sup><sup>_iK_,</sup> ~~q~~ is decomposed into a contribution of _λi − σ_ ML<sup>2fromtheunit-varianceisotropiclatent</sup> variable mapped into R<sup>_M_</sup> plus a contribution _σ_ ML<sup>2from the independent isotropic noise.</sup> In fact, **Q** ML is more generally given by 

where **R** is an arbitrary rotation matrix (that is, **R** is orthogonal) in the latent variable space (cf. the discussion around (6.29)). For simplicity, we choose **R** = **I** _M_ . 

83 

_6.3.2 Dimensionality Reduction with PPCA_ 

#### **6.3.2 Dimensionality Reduction with PPCA** 

In the previous section, we computed the ML estimates for _<u>✓</u>_ = _{_ **Q** _, m, σ}_ in (6.33), (6.36) and (6.37), respectively. Applying (6.23), we thus have 

where 

The reverse mapping can be done via 

Note that we are no longer applying an (orthogonal) linear projection. This is because the MMSE estimation step (6.39) shrinks the latent variables towards the origin, which then shrinks the reconstruction (6.41) towards its mean. 

#### **6.3.3 Relationship Between PCA and PPCA** 

We review the PPCA model in the limit for small _σ_<sup>2</sup> . The ML estimate (6.38) is 

The posterior mean (6.39) is 

and the reconstruction (6.41) is 

which is exactly the same orthogonal projection derived from the classical PCA! Thus, PPCA coincides with PCA in the small noise limit. 

84 

_6.4 Expectation Maximization_ 

### **6.4 Expectation Maximization** 

#### **6.4.1 EM for Probabilistic PCA** 

In (6.36), a closed-form expression for the ML estimate of the **W** _K_ is given. If both _M_ and _K_ are very large, however, computing the eigendecomposition of the data covariance matrix can be computationally intractable. Furthermore, many extensions of PPCA exist (such as (P)PCA mixture models [5] or _Factor Analysis_ ) that do not have a closed-form solution. To help simplify the calculations, we can use an ExpectationMaximization (EM) formulation of the problem. 

We define the complete likelihood function for the observations _<u>x</u>_ ~~1~~<sup>_, . . . , x_</sup> _N_<sup>andthe</sup> latent variables _<u>z</u>_ ~~1~~<sup>_, . . . , z_</sup> _~~N~~_<sup>by</sup> 

EM is an iterative procedure, and we denote the estimates at iteration _t_ by _<u>✓</u>_ _~~t~~_<sup>=</sup> _{_ **Q** _t, µ, σt_<sup>2</sup><sup>_}_.Themeanisfixedat</sup><sup>_<u>m</u>_</sup> = _<u>µ</u>_ which is easy to compute. Recall that EM has two main steps: 

- **E step:** Calculate 

where E _<u>✓</u>_ _~~t~~_<sup>[</sup><sup>_·_]specifiesthattheexpectationiscomputedaccordingtotheposterior</sup> distribution of the latent variables _p✓_ _~~t~~_<sup>(</sup><sup>_<u>z</u>_</sup> ~~1~~<sup>_, . . . , z_</sup> _~~N~~_<sup>_|x_</sup> 1<sup>_, . . . , x_</sup> _~~N~~_<sup>)forfixed</sup><sup>_<u>✓</u>_</sup> _~~t~~_<sup>.</sup> 

Recall that we model the samples as being independent. We can thus write the expectation in (6.46) as 

Since _<u>z</u>_ _~~i~~_<sup>_⇠N_(0</sup><sup>_,_</sup><sup>**I**</sup><sup>_K_),thefirstsummandin(6.48)is</sup> 

85 

_6.4.1 EM for Probabilistic PCA_ 

Using (6.22) and _<u>m</u>_ = _<u>µ</u>_ <u>,</u> the vector _<u>xi</u>_<sup>isarealizationofarandomvectorthat,condi-</sup> tioned on _<u>zi</u>_<sup>, is Gaussian with mean</sup><sup>**Q**</sup><sup>_<u>z</u>_</sup> _~~i~~_<sup>+</sup><sup>_<u>µ</u>_</sup> and covariance _σ_<sup>2</sup> **I** _N_ . The second summand in (6.48) can thus be written as 

Putting the previous results together and neglecting the constant terms, in the E step we need to compute: 

Note that the expectations depend on the current value of _<u>✓</u>_ _~~t~~_<sup>.</sup> 

To evaluate these, we need the posterior distribution _p_ ( _<u>z</u>_ _~~i~~_<sup>_|x_</sup> _i_<sup>),whichisGaussian.We</sup> already computed the posterior mean for a given **Q** in (6.39). Using the inverse of **C** given by (6.28) and defining **M** _t_ as in (6.31), the posterior covariance matrix can be 

86 

_6.4.1 EM for Probabilistic PCA_ 

computed via 

Combining (6.39) and (6.52), the posterior distribution is 

Using this distribution and E⇥ _<u>Z</u>_ _~~i~~_<sup>_<u>Z</u>_</sup> T _~~i~~_ ⇤ = Cov [ _<u>Z</u>_ _~~i~~_<sup>] + E[</sup><sup>_<u>Z</u>_</sup> _i_<sup>] E[</sup><sup>_<u>Z</u>_</sup> _i_<sup>]T,wegettheE-stepequa-</sup> tions of probabilistic PCA: 

Optimizing over **Q** and _σ_<sup>2</sup> while keeping _<u>✓</u>_ _~~t~~_<sup>fixedyieldstheupdates[4]:</sup> 

It turns out [4] that the EM algorithm for PPCA has only one stable fix point, i.e., EM converges to the globally optimal solution after sufficiently many iterations. 

87 

_6.4.2 EM for PCA_ 

#### **6.4.2 EM for PCA** 

We showed in Section 6.3.3 that PPCA converges to PCA in the small noise limit _σ_<sup>2</sup> _!_ 0. Interestingly, we can exploit this property to derive an EM algorithm for the standard PCA [9]. Although the distribution of _<u>X</u>_ _~~i~~_<sup>becomes singular for</sup><sup>_σ_2= 0, the EM algorithm</sup> _σ_<sup>2</sup> _!_ 0 still works in this case. Since **M** _t !_ **Q**<sup>T</sup> **Q** , we have the simple update rule 

_•_ **E step:** 

We can specify the EM algorithm more concisely in terms of the centered data matrix. Define similarly to (6.18) 

and note that the first term in (6.60) becomes 

Proceeding similarly with the second term, we can write the EM algorithm for PCA as 

- **E step:** 

- **M step:** 

Note that we could have written the update rules in terms of **C** ML = _N_<sup><u>1</u></sup> **_X_**<sup>˜</sup> **_X_**<sup>˜T</sup> . This is, however, undesirable since the complexity of computing **C** ML is the multiplication of an _M ⇥ N_ with an _N ⇥ M_ matrix, which scales with the dimensions _NM_<sup>2</sup> . If implemented as above, the most computationally costly operation is the multiplication **Q**<sup>T</sup> **_X_**<sup>˜</sup> , i.e., multiplication of a _K ⇥ M_ with an _M ⇥ N_ matrix. This scales with the dimensions _KMN_ . Since we choose _K < M_ , and often _K ⌧ M_ , this lets us apply PCA via EM for problems that are too large for regular PCA. 

88 

## **Bibliography** 

- [1] Ky Fan, “On a theorem of Weyl concerning eigenvalues of linear transformations,” _Proceedings of the National Academy of Sciences of the United States of America_ , vol. 36, no 1, pp. 31–35, Jan 1950 

- [2] C. M. Bishop, “Pattern recognition and machine learning”, Springer, 2006 

- [3] Jonathon Shlens, “A tutorial on principal component analysis,” _arXiv preprint arXiv:1404.1100_ , 2014 

- [4] Tipping, M. E. and C. M. Bishop “Probabilistic principal component analysis,” _Journal of the Royal Statistical Society_ , vol. 21, no. 3, pp. 611–622, 1999 

- [5] Tipping, M. E. and C. M. Bishop “Mixtures of probabilistic principal component analysers”, _Neural Computation_ , vol. 11, no. 2, pp. 443–482, 1999 

- [6] K. P. Murphy, “Machine learning: a probabilistic perspective,” MIT Press, 2012. 

- [7] K. B. Petersen and M. S. Pedersen”, “The Matrix Cookbook,” Technical University of Denmark, Nov 2012 

- [8] _MNIST database of handwritten digits_ , at `http://yann.lecun.com/exdb/ mnist/` http://yann.lecun.com/exdb/mnist/ 

- [9] S. Roweis, “EM algorithms for PCA and SPCA”, _Advances in Neural Information Processing Systems_ , vol. 10, pp. 626-632

---

## 源文件

- [script.pdf](attachments/documents/AI_Machine-Learning-in-Communication-6e716d50d0b9/script.pdf)
