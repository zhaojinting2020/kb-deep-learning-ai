---
title: mock_exam_sol
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-825faaed6f18/mock_exam_sol.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-825faaed6f18/mock_exam_sol.pdf
  title: mock_exam_sol.pdf
---

# Machine Learning for Communications Mock Exam 

February 07, 2018 

Institute for Communications Engineering Technical University of Munich Prof. Dr. sc. techn. Gerhard Kramer 

_1 Linear Regression and Cost Functions_ 

## **1 Linear Regression and Cost Functions** 

Table 1 contains data points form the training set _D_ . We model the relation between the input variables _x_ and output variables _y_ as 

where _U ⇠U_ [ _−_ 1 _,_ 1] is a random variable uniformly distributed on the [ _−_ 1 _,_ 1] interval, i.e., the probability density function of _U_ is 

Table 1: Training set _D_ . 

- (a) Derive the cost function for the model as in (1) using the maximum likelihood (ML) principle, i.e., the minimizing parameters of the cost function should be the MLparameters for the model. (Hint: you can use log2 likelihoods) 

- (b) Assume the following parameters _w_ = 1 _, b_ = _−_ 1. Compute the cost for these parameters. (Hint: you may find it useful to plot the model and the data points) From Figure 1 (or calculations), we can observe that for _i_ = 1 ( _y_<sup>(</sup><sup>_i_)</sup> _− wx_<sup>(</sup><sup>_i_)</sup> _− b_ ) _2/_ [ _−_ 1; 1], and hence _C_ = _1_ . 

- (c) Assume the following parameters _w_ =<sup><u>1</u></sup> 2<sup>_, b_= 1.Computethecostfortheseparam-</sup> eters. 

From Figure 1 (or calculations), we can observe that _8i_ ( _y_<sup>(</sup><sup>_i_)</sup> _− wx_<sup>(</sup><sup>_i_)</sup> _− b_ ) _2_ [ _−_ 1; 1], and hence _C_ = 4. 

2 

_1 Linear Regression and Cost Functions_ 

Figure 1: Figure comparing di↵erent models. 

- (d) State ML-parameters for the model (1) and the training set _D_ . Justify the choice. Are the ML-parameters unique? Justify. 

The parameters ( _w, b_ ) = (<sup><u>1</u></sup> 2<sup>_,_1)attaintheminimumofthecostfunction.Therefore</sup> they are the ML-parameters. The choice is not unique as we can fit many models such that ( _y_<sup>(</sup><sup>_i_)</sup> _− wx_<sup>(</sup><sup>_i_)</sup> _− b_ ) _2_ [ _−_ 1; 1] _8i_ . 

Now we change the model. The relation between the input variables _x_ and output variables _y_ is now modeled as 

where _N ⇠N_ (0 _,_ 1) is a Gaussian random variable with zero mean and unit variance, i.e., the probability density function of _N_ is 

- (e) Derive the cost function for the model as in (3) using the maximum likelihood (ML) principle, i.e., the miniming parameters of the cost function should be the ML-parameters for the model. (Hint: you can use log likelihoods). 

- (f) Assume the following parameters _w_ = 1 _, b_ = _−_ 1. Compute the gradient for these 

3 

_2 Neural Networks_ 

parameters. 

- (g) Are the parameters from the previous question the ML-parameters? Justify. No, they are not as it is necessary for the ML-parameters to have zero gradient. 

## **2 Neural Networks** 

Consider neural network as in Figure 2 with the following notation: 

- _w_<sup>[</sup><sup>_k_]</sup> _ij_<sup>denotes a weight associated with a signal going from</sup><sup>_j_-th neuron in the (</sup><sup>_k−_1)-</sup> 

- th layer to the _i_ -th neuron in the _k_ -th layer. 

- _b_<sup>[</sup> _i_<sup>_k_]</sup> denotes the bias term for the _i_ -th neuron in the _k_ -th layer. 

- _a_<sup>[</sup> _i_<sup>_k_]</sup> denotes the output of the _i_ -th neuron in the _k_ -th layer. 

- _zi_<sup>[</sup><sup>_k_]</sup> is the (total) input to the _i_ -th neuron in the _k_ -th layer. 

We use the network for a regression problem with the training set _D_ from Table 1 (from the previous exercise). We model _y_ as being an output of the network with input _x_ . We set the neuron in the output layer to use the identity activation function, i.e., _g_ ( _z_ ) = _z_ . Other neurons use the rectified linear unit (ReLU) activation function, i.e., _g_ ( _z_ ) = max( _z,_ 0). Assume the following parameters: 

4 

_2 Neural Networks_ 

Figure 2: Neural Network for Exercise 2. 

To measure the performance of the network we use the quadratic cost function 

where _C_<sup>(</sup><sup>_i_)</sup> = _c_ ( _x_<sup>(</sup><sup>_i_)</sup> _, y_<sup>(</sup><sup>_i_)</sup> ) is the cost for the _i_ -th training sample. 

(a) Compute the cost for the 3rd training sample, i.e., the cost _C_<sup>(3)</sup> for the pair ( _x_<sup>(3)</sup> _, y_<sup>(3)</sup> ). 

> (b) Compute the gradients of the cost for the 3rd training sample with respect to network parameters, i.e., compute _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(3)</sup> [1]<sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> **W**<sup>(3)[2]</sup><sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(3)</sup> [3]<sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_ [1]<sup>(3)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_ [2]<sup>(3)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_<sup>[3](3).(Use the back-</sup> propagation). 

5 

_2 Neural Networks_ 

We are going to need the values _a_<sup>[0]</sup> _, z_ [1] _, a_ [1] _, z_ [2] _, a_ [2] _, z_ [3] _, a_ [3] which we have from the previous point and the derivative of the ReLU activation 

Starting from the last layer we back-propagate as follows 

The gradients are 

> (c) Assume that the gradients for other training samples are zero, i.e., _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(</sup> [1]<sup>_i_)</sup><sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> **W**<sup>([2]</sup><sup>_i_)</sup><sup>_,_</sup> _@_<sup>_<u>@C</u>_</sup> _<u>W</u>_<sup>(</sup> [3]<sup>_i_)</sup><sup>_,_</sup> _<u>@C@b</u>_ [1]<sup>(</sup><sup>_i_)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_ [2]<sup>(</sup><sup>_i_)</sup><sup>_,_</sup><sup>_<u>@C</u>_</sup> _@b_<sup>[3](</sup><sup>_i_)arezerofor</sup><sup>_i_= 1</sup><sup>_,_2</sup><sup>_,_4.Arethenetworkparametersoptimalinthis</sup> case? Justify. 

6 

_2 Neural Networks_ 

Knowing that the other gradients are zero we could compute the total gradient 

However, zero gradient states only that the network parameters are a local extremum of the cost function. Therefore no conclusion can be made in general whether the parameters are optimal. This is because the problem is non-convex. 

- (d) Assume that the cost for other training samples is zero, i.e, _C_<sup>(</sup><sup>_i_)</sup> = 0 for _i_ = 1 _,_ 2 _,_ 4. Are the parameters optimal in this case? Justify. Knowing that the other costs are zero we can compute the total cost 

Observe that always _C ≥_ 0, and hence, we can conclude that the parameters are the minimizing parameters. 

7 

_3 Universal Source Coding_ 

## **3 Universal Source Coding** 

1. _S_ does not form a source tree because it is not a free set. 

2. The updated Context tree is given in Fig. 3. The probability fed to the arithmetic 

Figure 3: Context tree after encoding 5 bits 

encoder is<sup><u>151</u></sup> 2<sup>18andthecodewordlengthisgivenby</sup><sup>_`_(</sup><sup>_x_9) =</sup> log 151<sup><u>218</u></sup> + 1 = 12 l m 

8 

_3 Universal Source Coding_ 

3. We can calculate the model loss as follows. 

so the model loss is less than 5 bits. 

9 

_4 Inference in Graphical Models_ 

## **4 Inference in Graphical Models** 

2. The factor graph is: 

10 

_4 Inference in Graphical Models_ 

4. not cycle-free 

5. Use notation _ma!b_ ( _x_ ) = [ _ma!b_ (0) _, ma!b_ (1)] for binary message. All messages are initialized to [0 _._ 5 _,_ 0 _._ 5]. 

_•_ After variable node update of _x_ 1 _, x_ 2 _, x_ 3 _, x_ 4 we have _mx_ 1 _!f_ 5( _x_ 1) = _mx_ 1 _!f_ 6( _x_ 1) = _mf_ 1 _!x_ 1( _x_ 1) = [0 _._ 2 _,_ 0 _._ 8] _mx_ 2 _!f_ 5( _x_ 2) = _mx_ 2 _!f_ 6( _x_ 2) = _mf_ 2 _!x_ 2( _x_ 2) = [0 _._ 9 _,_ 0 _._ 1] _mx_ 3 _!f_ 5( _x_ 3) = _mf_ 3 _!x_ 3( _x_ 3) = [0 _._ 3 _,_ 0 _._ 7] _mx_ 4 _!f_ 6( _x_ 4) = _mf_ 4 _!x_ 4( _x_ 4) = [0 _._ 5 _,_ 0 _._ 5] 

_•_ After factor node update of _f_ 5: _mf_ 5 _!x_ 1( _x_ 1) = [0 _._ 34 _,_ 0 _._ 66] _mf_ 5 _!x_ 2( _x_ 2) = [0 _._ 62 _,_ 0 _._ 38] _mf_ 5 _!x_ 3( _x_ 3) = [0 _._ 26 _,_ 0 _._ 74] 

_•_ After factor node update of _f_ 6: _mf_ 6 _!x_ 1( _x_ 1) = [0 _._ 5 _,_ 0 _._ 5] _mf_ 6 _!x_ 2( _x_ 2) = [0 _._ 5 _,_ 0 _._ 5] _mf_ 6 _!x_ 4( _x_ 4) = [0 _._ 18 _,_ 0 _._ 82] 

_•_ Compute the marginals: _p_ ˜ _X_ 1 _|_ **_Y_** ( _x_ 1 _|_ **_y_** ) = _mf_ 1 _!x_ 1( _x_ 1) _mf_ 5 _!x_ 1( _x_ 1) _mf_ 6 _!x_ 1( _x_ 1) _⇡_ [0 _._ 114 _,_ 0 _._ 886] _p_ ˜ _X_ 2 _|_ **_Y_** ( _x_ 2 _|_ **_y_** ) = _mf_ 2 _!x_ 2( _x_ 2) _mf_ 5 _!x_ 2( _x_ 2) _mf_ 6 _!x_ 2( _x_ 2) _⇡_ [0 _._ 936 _,_ 0 _._ 064] _p_ ˜ _X_ 3 _|_ **_Y_** ( _x_ 3 _|_ **_y_** ) = _mf_ 3 _!x_ 3( _x_ 3) _mf_ 5 _!x_ 3( _x_ 3) _⇡_ [0 _._ 131 _,_ 0 _._ 869] ˜ _pX_ 4 _|_ **_Y_** ( _x_ 4 _|_ **_y_** ) = _mf_ 4 _!x_ 4( _x_ 4) _mf_ 6 _!x_ 4( _x_ 4) = [0 _._ 18 _,_ 0 _._ 82] 

11 

_5 Expectation Maximization_ 

## **5 Expectation Maximization** 

1. The log-likelihood function is 

2. The latent variable is _X_ . If we knew which observations _yi, i_ = 1 _, . . . , N_ originate from the transmission of _x_ 0 and _x_ 1, we could cluster them accordingly, define a new log-likelihood function based on _pY |X_ ( _y|x_ ) = _pN_ ( _y − x_ ) and facilitate the parameter estimation. 

3. In the E-step of the EM algorithm we are supposed to find an auxiliary distribution _QXi_ . As we have shown in class, this auxiliary distribution can be calculated for the _t_ -th iteration as 

where we 

4. In the M-step we try to solve the following optimization problem (to set **_✓_**<sup>(</sup><sup>_t_+1)</sup> = **_✓_**<sup>_⇤_</sup> ): 

where 

In the last step, we would have to calculate the derivative with respect to the parameter **_✓_** . In the given example this requires a case distinction as _pN_ is defined in a piecewice fashion. 

5. We use the `numpy` feature of broadcasting to accomplish this task: 

```
>>>y1=y.respape((N,1))
>>>f1=f.reshape((1,K))
>>>g1=g.reshape((1,K))
>>>z=np.sum((y1-f1)*g1,axis=1)
```

12 

_5.1_ 

### **5.1** 

You are given the following data sets with realizations of the random vector _<u>X</u>_ = ( _<u>X</u>_ (1) _, X_ <u>(2)):</u> 

Determine a reasonable estimate of the sample covariance matrix for each case. 

#### _Solution:_ 

a) We can infer that both _<u>X</u>_ <u>(1)</u> and _<u>X</u>_ <u>(2)</u> are uniformly distributed on the interval [0 _._ 5 _,_ 1 _._ 5]. The variance of a uniform random variable on [ _a, b_ ] is given by ( _b − a_ )<sup>2</sup> _/_ 12 and thus Var [ _<u>X</u>_ <u>(1)]</u> = Var [ _<u>X</u>_ <u>(2)]</u> = 1 _/_ 12. Further, we can guess from the data that _<u>X</u>_ <u>(1)</u> and _<u>X</u>_ <u>(2)</u> are independent and thus uncorrelated. Our estimate for the covariance matrix is thus 

b) We can infer that _<u>X</u>_ <u>(1)</u> is uniformly distributed on [ _−_ 1 _,_ 1] and _<u>X</u>_ <u>(2)</u> uniformly distributed on [ _−_ 2 _,_ 2]. Hence, we guess Var [ _<u>X</u>_ <u>(1)]</u> = 1 _/_ 3 and Var [ _<u>X</u>_ <u>(2)]</u> = 4 _/_ 3. In this case, however, they are not independent. In fact, we have _<u>X</u>_ <u>(2)</u> _⇡−_ 2 _·_ _<u>X</u>_ <u>(1).</u> Thus, we have 

Our estimate for the covariance matrix is thus 

c) _<u>X</u>_ is concentrated very close to a circle of radius 1. Further, the realizations seem to be uniform angle. Looking at the projections onto the axes, we see that _<u>X</u>_ <u>(1)</u> and _<u>X</u>_ <u>(1)</u> take values in [ _−_ 1 _− ",_ 1 + _"_ ] with somewhat larger probabilities for higher magnitudes. Hence, the variances are larger than 1 _/_ 3, which is the variance of a uniform distribution, but certainly smaller than 1. We thus guess Var [ _<u>X</u>_ <u>(1)]</u> = Var [ _<u>X</u>_ <u>(2)]</u> _⇡_ 0 _._ 5. Further, although _<u>X</u>_ <u>(1)</u> and _<u>X</u>_ (2) are clearly not independent, they appear to be uncorrelated (as the distribution looks isotropic) and we guess E [[] _<u>X</u>_ <u>(1)</u> _<u>X</u>_ <u>(2)] = 0.</u> Our estimate for the covariance matrix is thus 

13 

_5.2_ 

### **5.2** 

Show that the sample covariance matrix **C** _X 2_ R<sup>_M⇥M_</sup> of a data set is symmetric and T positive semi-definite, i.e., **C** _X_<sup>T=</sup><sup>**C**</sup><sup>_X_and</sup><sup>_<u>z</u>_</sup> **C** _X_ _<u>z</u> ≥_ 0 for all _<u>z</u> 2_ R<sup>_M_</sup> . 

_Solution:_ Let _<u>µ</u>_ be the sample mean and denote the sample covariance matrix by **C** _X_ = <u>1</u> _N N −_ 1 P _i_ =1<sup>(</sup><sup>_<u>x</u>_</sup> _~~i~~_<sup>_−_</sup><sup>_<u>µ</u>_</sup> )( _<u>x</u>_ _~~i~~_<sup>_−_</sup><sup>_<u>µ</u>_</sup> <u>)</u><sup>T</sup> . Symmetry holds since 

Positive-semidefiniteness holds since for any _<u>z</u> 2_ R<sup>_M_</sup> , 

## **6 Mixtures of probabilistic PCA** 

For probabilistic PCA, we consider a Gaussian model 

In the following, we will consider a mixture of two probabilistic PCA models with parameters _<u>✓</u>_ ~~1~~<sup>and</sup><sup>_<u>✓</u>_</sup> 2<sup>where</sup><sup>_<u>✓</u>_</sup> _i_<sup>= (</sup><sup>**Q**</sup><sup>_i, m_</sup> _~~i~~_<sup>_, σ_</sup> _i_<sup>2)and</sup><sup>_P_(</sup><sup>_<u>✓</u>_</sup> ~~1~~<sup>) = 1</sup><sup>_/_3and</sup><sup>_P_(</sup><sup>_<u>✓</u>_</sup> ~~2~~<sup>) = 2</sup><sup>_/_3.</sup> 

14 

— ~~_ |~~ 

~~<u>a</u> | | |~~ <u>~</u> ]) |} <u>-</u> ~~J)~~ 

~~~~~ 

_6 Mixtures of_ _<u>probabilistic</u> PCA_ 

3. Consider a data point _<u>x</u>_ = [0 _,_ 0]<sup>T</sup> . Is _<u>x</u>_ assigned to the first or second model according to a maximum likelihood criterion? 

   - _Solution:_ 

_)_ Assign _<u>x</u>_ to second model. 

4. Use the model chosen in 3. to reduce the dimension of _<u>x</u>_ to one dimension. We use the second model and get 

16

---

## 源文件

- [mock_exam_sol.pdf](attachments/documents/AI_Machine-Learning-in-Communication-825faaed6f18/mock_exam_sol.pdf)
