---
title: mlcomm_exam
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-cf890de50131/mlcomm_exam.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-cf890de50131/mlcomm_exam.pdf
  title: mlcomm_exam.pdf
---

# mlcomm_exam

Technische Universit¨at M¨unchen Lehrstuhl f¨ur Nachrichtentechnik Prof. Dr. sc. techn. Gerhard Kramer 

# **Machine Learning for Communications** 

### **Prof. Dr. techn. sc. Gerhard Kramer WS 2017/2018 February 20th, 2018** 

- Write your name and student ID on every sheet. Please have your student card ready for examination. 

- The exam duration is 90 minutes. 

- This exam has 6 questions on 17 pages (excluding the cover page). 

- Do not write with pencils or red pens. 

- You can get full credit if and only if you give reasons for your answer. 

- Problem parts that are marked with an = _)_ can be solved independently of previous parts of the same problem. 

- Please remain seated until the end of the exam. 

- You must hand in this problem set! 

- You are allowed to use a non-programmable calculator and one DIN-A4 sheet (i.e., two pages) of handwritten notes. 

## **Good Luck!** 

Name: 

|Student ID:<br>Course of Studies:|
|---|
|**Please read and sign below:**|
|I hereby confrm that I have been informed prior to the beginning of t<br>to notify the examination supervisors immediately if sudden illness <br>tion. This will be noted in the examination protocol. An application<br>be fled immediately to the board of examiners being in charge. A me<br>the physicians acknowledged by the Technische Universit¨at M¨unchen<br>the examination must be forwarded without delay. In case the exami<br>of illness, a subsequent withdrawal due to illness cannot be accepte<br>is ended due to illness it will not be graded.<br>Signature:|

I hereby confirm that I have been informed prior to the beginning of the examination that I have to notify the examination supervisors immediately if sudden illness occurs during the examination. This will be noted in the examination protocol. An application for exam withdrawal has to be filed immediately to the board of examiners being in charge. A medical certificate from one of the physicians acknowledged by the Technische Universit¨at M¨unchen issued on the same day as the examination must be forwarded without delay. In case the examination is completed despite of illness, a subsequent withdrawal due to illness cannot be accepted. In case the examination is ended due to illness it will not be <u>graded.</u> 

## **For internal use only:** 

|Aufgabe|Punktzahl|Davon erreicht|
|---|---|---|
|Short Questions|5||
|Linear Models and Neural Networks|25||
|Arithmetic Source Coding|8||
|Inference in Graphical Models|22||
|Principal Component Analysis|16||
|Expectation Maximization|14||
|**Summe**|90||

**(5 points)** 

## **Problem 1: Short Questions** 

Mark the right answers in the following. For each question, _only one answer is correct_ . 

- a) We are interested in calculating E [ _f_ ( _X_ )], where _X ⇠ pX_ . An auxiliary distribution (1 pt) with the same support as _pX_ is _qX_ . We want to use importance sampling to calculate an estimate for the mean. 

We need to be able to sample from _pX_ . 

- <u>X</u> We need to be able to sample from _qX_ . The probability density function _qX_ must be chosen such that E _X⇠qX_ [ _X_ ] = E _X⇠pX_ [ _X_ ]. The probability density function _qX_ must be chosen such that E _X⇠qX_ [ _f_ ( _X_ )] = E _X⇠pX_ [ _f_ ( _X_ )]. 

- b) We are interested in calculating E [ _f_ ( _X_ )], where _X ⇠ pX_ . An auxiliary distribution (1 pt) with the same support as _pX_ is _qX_ . We have _pX_ ( _x_ ) = _p_<sup>_⇤_</sup> _X_<sup>(</sup><sup>_x_)</sup><sup>_/Zp_and</sup><sup>_qX_(</sup><sup>_x_) =</sup><sup>_q_</sup> _X_<sup>_⇤_(</sup><sup>_x_)</sup><sup>_/Zq_,</sup> where _Zp_ and _Zq_ are normalization constants. We want to use rejection sampling to calculate an estimate for the mean. 

We need to be able to evaluate both _pX_ and _qX_ . 

- <u>X</u> We need to be able to evaluate both _p_<sup>_⇤_</sup> _X_<sup>and</sup><sup>_q_</sup> _X_<sup>_⇤_.</sup> We require _pX_ ( _x_ ) _ c · qX_ ( _x_ ) _, 8x_ , _c >_ 0. The sample _xi_ drawn from _qX_ is kept with probability _pX_ ( _xi_ ). 

c) We now consider Markov Chain Monte Carlo (MCMC) sampling approaches for **_X_** _⇠_ (1 pt) _p_ **_X_** . An auxiliary distribution with the same support as _p_ **_X_** is _q_ **_X_** . MCMC methods try to _. . ._ 

- <u>X</u> _. . ._ construct an ergodic Markov chain which has _p_ **_X_** as an invariant distribution. _. . ._ construct an invariant Markov chain which has _p_ **_X_** as an ergodic distribution. _. . ._ construct an ergodic Markov chain which has _q_ **_X_** as an invariant distribution. _. . ._ find an auxiliary distribution _q_ **_X_** , which is easy to sample and can be used to approximate the mean of **_X_** . 

d) We now consider stochastic matrices. 0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 3 ✓ ◆ imately) (0 _._ 46 _,_ 0 _._ 54)<sup>T</sup> . 0 _._ 3 0 _._ 6 <u>X</u> The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 4 ✓ ◆ imately) (0 _._ 46 _,_ 0 _._ 54)<sup>T</sup> . 0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 4 ✓ ◆ imately) (0 _._ 76 _,_ 0 _._ 65)<sup>T</sup> . 0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. The eigenvalues are 0.46 and 0.54. 0 _._ 7 0 _._ 4 ✓ ◆ 

(2 pt) 

_Page 1 - Please turn page_ 

**Problem 2: Linear Models and Neural Networks (25 points)** 

Table 1 contains data points from the training set _D_ . We model the relation between the input variables _x_ and output variables _y_ as 

where _N_ ( _µ, σ_<sup>2</sup> ) is the Gaussian distribution with mean _µ_ and variance _σ_<sup>2</sup> . We model _a_ A( _x_ ) as an output of a single neuron (we refer to this model as model A) 

Table 1: Training set _D_ . 

a) Derive the cost function for the model as in (1) using the maximum likelihood (ML) (3 pt) principle, that is, the _minimizing parameters_ of the cost function should be the MLparameters for the model _(Hint: You can use the logarithm to simplify the cost function. There are many correct answers here. Find the cost function of a simple form for easier calculations in the next problems.)_ 

### **Solution:** 

_Page 2 - Please turn page_ 

b) Find the optimal parameters ( _w, b_ ) for the model A. 

(5 pt) 

### **Solution:** 

The model corresponds to a linear regression, which is a convex problem, and hence: 

c) Compute the value of the cost function for the optimal parameters. 

(2 pt) 

### **Solution:** 

_Page 3 - Please turn page_ 

We would like to extend our model to explain nonlinear dependencies. One technique is to introduce non-linear basis functions. We choose the model B 

We train the models A and B with the gradient descent (GD) algorithm. The notation _a_<sup>@</sup> Z<sup>_n_</sup> denotes the model Z (Z can be replaced by A or B) after _n_ iterations of the GD algorithm, for example, _a_<sup>@100</sup> A ( _x_ ) is the output of the model A after 100 iterations of the GD algorithm for the input _x_ . We define the cost for the model Z after _n_ iterations as 

for example, _C_ B(100) is the value of the cost function for the model B after 100 iterations and _C_ B( _1_ ) is the value of the cost function for the model B after the GD algorithm converges (we assume a properly chosen learning rate). 

= _)_ 

d) We apply the gradient descent (GD) algorithm to train the models A and B. Assume random initialization and a properly chosen learning rate. Tick the corresponding box if the statement is true. _Only one answer_ is correct. _Hint: The optimal parameters are obtained by minimizing the cost function._ 

(2 pt) 

_C_ A(100) _≥ C_ B(100) 

_C_ A(100) _ C_ B(100) 

<u>X</u> _C_ A( _1_ ) _≥ C_ B( _1_ ) 

None of the statements is true. 

Weight decay is a regularization technique that modifies the gradient used in the GD algorithm. We use the notation _a_<sup>@</sup> Z _,R_<sup>_n_to denote the model Z after</sup><sup>_n_iterations of the GD</sup> algorithm with weight decay. We define the cost for the model Z after _n_ iterations with weight decay as 

= _)_ e) Assume a random initialization, a properly chosen learning rate, and weight decay regularization. For the regularization, models A and B are using the same non-zero parameter _λ_ . Tick the corresponding box if the statement is true. _Only one answer_ is correct. _Hint: The optimal parameters are obtained by minimizing the cost function._ 

(2 pt) 

<u>X</u> _C_ A<sup>_R_(</sup><sup>_1_)</sup><sup>_≥C_A(</sup><sup>_1_)</sup> 

_C_ A<sup>_R_(100)</sup><sup>_≥C_A(100)</sup> _C_ A( _1_ ) _≥ C_ B<sup>_R_(</sup><sup>_1_)</sup> 

None of the statements is true. 

_Page 4 - Please turn page_ 

Another way to extend our model to explain nonlinear dependencies would be to use a network of neurons, that is, a neural network. We introduce model C such that _a_ C( _x_ ) is the output of the neural network from Figure 1 with input _x_ . We use the following notation: 

- _wij_<sup>[</sup><sup>_k_]denotes a weight associated with a signal going from</sup><sup>_j_-th neuron in the (</sup><sup>_k−_1)-th</sup> layer to the _i_ -th neuron in the _k_ -th layer. 

- _b_<sup>[</sup> _i_<sup>_k_]</sup> denotes the bias term for the _i_ -th neuron in the _k_ -th layer. 

- _a_<sup>[</sup> _i_<sup>_k_]</sup> denotes the output of the _i_ -th neuron in the _k_ -th layer. 

- _zi_<sup>[</sup><sup>_k_]</sup> is the (total) input to the _i_ -th neuron in the _k_ -th layer. 

- _g_<sup>[</sup><sup>_k_]</sup> ( _·_ ) activation function for the neurons in _k_ -th layer (all neurons in the layer use the same activation function). 

Figure 1: Neural Network for Model C. 

- = _)_ f) How many hidden layers does the neural network have? How many neurons are in each (1 pt) hidden layer? 

**Solution:** 2 hidden layers, 2 neurons per layer. 

We initialize the network with random parameters: 

and we use the identity activation function, that is, _g_<sup>[</sup><sup>_k_]</sup> ( _x_ ) = _x_ for _k_ = 0 _,_ 1 _,_ 2 _,_ 3. 

_Page 5 - Please turn page_ 

- = _)_ g) Show that the implied model by the neural network is _a_ C( _x_ ) = _↵x_ + _β_ for some _↵, β 2_ R. (5 pt) Identify the parameters _↵_ and _β_ and express them in terms of the neural network parameters. 

### **Solution:** 

Using forward-propagation we get (thanks to linear activation functions): 

Back-propagation is an e↵ective algorithm for computing partial derivatives of the cost function with respect to neural networks’ parameters. The partial derivatives are needed to apply the GD algorithm to train the network. In back-propagation, we compute the derivatives in the following order 

and than we compute the derivatives of interest, that is, _@_ **W** _<u>@C</u>_<sup>[</sup><sup>_k_]</sup><sup>_,_</sup> _@@C_ **_b_**<sup>[</sup><sup>_k_].</sup> 

= _)_ h) Assume that _@z@C_<sup>[3]=1andperformasingleback-propagationstep.Thatis,compute</sup> (3 pt) _@@C_ **_z_**<sup>[2](or,equivalently,find</sup> _@z@C_ 1<sup>[2]</sup> and _@z@C_ 2<sup>[2]).Expresstheresultintermsofthenetwork</sup> parameters. 

### **Solution:** 

_Page 6 - Please turn page_ 

- i) Suppose we train the neural network for model C with the GD algorithm using back(2 pt) propagation to compute the gradients. We use _C_ C( _n_ ) to denote the cost for the model C after _n_ iterations of the GD algorithm without regularization, just like in equation (4). Assume a properly chosen learning rate. Tick the corresponding box if the statement is true. _Only one answer_ is correct. _Hint: The optimal parameters are obtained by minimizing the cost function._ 

   - _C_ A( _1_ ) _> C_ C( _1_ ) and _C_ B( _1_ ) _> C_ C( _1_ ) 

- <u>X</u> _C_ C( _1_ ) = _C_ A( _1_ ) 

   - _C_ C( _1_ ) = _C_ B( _1_ ) 

   - None of the statements is true 

_Page 7 - Please turn page_ 

**(8 points)** 

## **Problem 3: Arithmetic Source Coding** 

We want to encode a block of _n_ source symbols _x_<sup>_n_</sup> using Arithmetic Source Coding to generate a codeword over a binary alphabet, i.e., _|U|_ = 2. 

- a) What can you already deduce about the codeword _f_ ( _x_<sup>_n_</sup> ) �not _F_ ( _x_<sup>_n_</sup> _|x_<sup>0</sup> _−1_<sup>) or</sup><sup>_P_(</sup><sup>_xn|x_0</sup> _−1_<sup>)</sup> � (3 pt) if you are given _F_ ( _x_<sup>_n−_1</sup> _|x_<sup>0</sup> _−1_<sup>)and</sup><sup>_P_(</sup><sup>_xn−_1</sup><sup>_|x_0</sup> _−1_<sup>)?</sup> 

**Solution:** We know that 

If the fractional binary representation of _F_ ( _x_<sup>_n−_1</sup> _|x_<sup>0</sup> _−1_<sup>)sharesthefirst</sup><sup>_k_bitsofthe</sup> fractional binary representation of _F_ ( _x_<sup>_n−_1</sup> _|x_<sup>0</sup> _−1_<sup>)+</sup><sup>_P_(</sup><sup>_xn−_1</sup><sup>_|x_0</sup> _−1_<sup>) then any</sup><sup>_F_¯(</sup><sup>_xn|x_0</sup> _−1_<sup>)</sup> will also share the same first _k_ bits of the fractional binary representation, and hence the first _k_ bits of the codeword will also be the same as of the binary representation of _F_ ( _x_<sup>_n−_1</sup> _|x_<sup>0</sup> _−1_<sup>)or</sup><sup>_F_(</sup><sup>_xn−_1</sup><sup>_|x_0</sup> _−1_<sup>) +</sup><sup>_P_(</sup><sup>_xn−_1</sup><sup>_|x_0</sup> _−1_<sup>).</sup> 

- b) What can you deduce about the codeword if _F_ ( _x_<sup>_n−_1</sup> _|x_<sup>0</sup> _−1_<sup>) = 0</sup><sup>_._35and</sup><sup>_P_(</sup><sup>_xn−_1</sup><sup>_|x_0</sup> _−1_<sup>) =</sup> (2 pt) 0 _._ 14? 

**Solution:** 0 _._ 35 has a fractional binary representation _._ 010 _. . ._ and 0 _._ 49 has a binary fractional representation starting with 0 _._ 011 _. . ._ hence the codeword has first 2 bits as 01. 

- c) Derive a lower bound for _`_ ( _f_ ( _x_<sup>_n_</sup> )) if we decide to use the following value _F_<sup>¯</sup> ( _x_<sup>_n_</sup> _|x_<sup>0</sup> _−1_<sup>)in</sup> (3 pt) the interval [ _F_ ( _x_<sup>_n_</sup> _|x_<sup>0</sup> _−1_<sup>)</sup><sup>_, F_(</sup><sup>_xn|x_0</sup> _−1_<sup>) +</sup><sup>_P_(</sup><sup>_xn|x_0</sup> _−1_<sup>))toencode</sup><sup>_xn_:</sup> _F_ ¯( _x_<sup>_n_</sup> _|x_<sup>0</sup> _−1_<sup>) =</sup><sup>_F_(</sup><sup>_xn|x_0</sup> _−1_<sup>) +3</sup> _−1_<sup>)</sup> 4<sup>_P_(</sup><sup>_xn|x_0</sup> 

(In the lecture we used _F_<sup>¯</sup> ( _x_<sup>_n_</sup> _|x_<sup>0</sup> _−1_<sup>) =</sup><sup>_F_(</sup><sup>_xn|x_0</sup> _−1_<sup>) +</sup><sup><u>1</u></sup> 2<sup>_P_(</sup><sup>_xn|x_</sup> _−1_<sup>0)toencode</sup><sup>_xn_).</sup> 

**Solution:** As discussed in the lecture we need _`_ ( _f_ ( _x_<sup>_n_</sup> )) to satisfy 

Based on the two equations we get 

hence 

_Page 8 - Please turn page_ 

**(22 points)** 

## **Problem 4: Inference in Graphical Models** 

Consider the random variables _Xi, i_ = 1 _,_ 2 _,_ 3 _,_ 4. 

- _X_ 1 and _X_ 2 are uniformly and independently distributed over _{_ 0 _,_ 1 _}_ . 

- _X_ 3 = _X_ 1 **or** _X_ 2. 

where **xor** and **or** are defined by 

Let _Yi_ denote a noisy version of _Xi_ . Suppose that we have 

||_pYi|Xi_(_yi|_0)|_pYi|Xi_(_yi|_1)|
|---|---|---|
|_i_= 1|0_._2|0_._8|
|_i_= 2|0_._9|0_._1|
|_i_= 3|0_._3|0_._7|
|_i_= 4|0_._7|0_._3|

= _)_ a) What is set of possible realizations of the random variable **_X_** = _X_ 1 _X_ 2 _X_ 3 _X_ 4? 

(2 pt) 

**Solution:** The random variable _X_ 1 _X_ 2 _X_ 3 _X_ 4 can take on values in the set 

. 

= _)_ b) How does the distribution _P_ **_X_** factorize? Determine the distribution _P_ **_X_** ( **_x_** ). 

**Solution:** The distribution factorizes as 

_Page 9 - Please turn page_ 

c) Compute the marginal distribution of _Xi, i_ = 1 _,_ 2 _,_ 3 _,_ 4, given **_Y_** = _Y_ 1 _Y_ 2 _Y_ 3 _Y_ 4, i.e., (4 pt) compute _pXi|_ **_Y_** ( _xi|_ **_y_** ). It is enough to provide the result up to a multiplicative constant. 

### **Solution:** 

d) Draw a factor graph of the given problem and relate the variable and factor nodes to (3 pt) the given quantities, i.e., probability mass functions and constraints. 

_Page 10 - Please turn page_ 

- e) Use the BP algorithm to compute the marginal distribution of _X_ 2 _|_ **_Y_** = **_y_** after one iter(10 pt) ation: _p_ ˜ _X_ 2 _|_ **_Y_** ( _x_ 2 _|_ **_y_** ). Why does the BP algorithm only provide an approximate solution? _Hint: Use the notation ma!b_ ( _x_ ) = [ _ma!b_ (0) _, ma!b_ (1)] _for a binary message from node a to node b._ 

   - _Compute the required messages for the variable nodes._ 

   - _Compute the required messages for the factor nodes._ 

   - _Compute the marginal distribution._ 

### **Solution:** 

Use notation _ma!b_ ( _x_ ) = [ _ma!b_ (0) _, ma!b_ (1)] for binary message. All messages are initialized to [0 _._ 5 _,_ 0 _._ 5]. [0.5 pt _⇥_ 4] After variable node update of _x_ 1 _, x_ 2 _, x_ 3 _, x_ 4 we have _mx_ 1 _!f_ 5( _x_ 1) = _mx_ 1 _!f_ 6( _x_ 1) = _mf_ 1 _!x_ 1( _x_ 1) = [0 _._ 2 _,_ 0 _._ 8] _mx_ 2 _!f_ 5( _x_ 2) = _mx_ 2 _!f_ 6( _x_ 2) = _mf_ 2 _!x_ 2( _x_ 2) = [0 _._ 9 _,_ 0 _._ 1] _mx_ 3 _!f_ 5( _x_ 3) = _mf_ 3 _!x_ 3( _x_ 3) = [0 _._ 3 _,_ 0 _._ 7] _mx_ 4 _!f_ 6( _x_ 4) = _mf_ 4 _!x_ 4( _x_ 4) = [0 _._ 7 _,_ 0 _._ 3] [2 pt] After factor node update of _f_ 5: _mf_ 5 _!x_ 1( _x_ 1) _⇡_ [0 _._ 3269 _,_ 0 _._ 6731] _mf_ 5 _!x_ 2( _x_ 2) _⇡_ [0 _._ 4697 _,_ 0 _._ 5303] _mf_ 5 _!x_ 3( _x_ 3) = [0 _._ 18 _,_ 0 _._ 82] [2 pt] After factor node update of _f_ 6: _mf_ 6 _!x_ 1( _x_ 1) _⇡_ [0 _._ 5667 _,_ 0 _._ 4333] _mf_ 6 _!x_ 2( _x_ 2) = [0 _._ 45 _,_ 0 _._ 55] _mf_ 6 _!x_ 4( _x_ 4) = [0 _._ 63 _,_ 0 _._ 37] [1 pt _⇥_ 4] Compute the marginals: _p_ ˜ _X_ 1 _|_ **_Y_** ( _x_ 1 _|_ **_y_** ) = _mf_ 1 _!x_ 1( _x_ 1) _mf_ 5 _!x_ 1( _x_ 1) _mf_ 6 _!x_ 1( _x_ 1) _⇡_ [0 _._ 1370 _,_ 0 _._ 8630] _p_ ˜ _X_ 2 _|_ **_Y_** ( _x_ 2 _|_ **_y_** ) = _mf_ 2 _!x_ 2( _x_ 2) _mf_ 5 _!x_ 2( _x_ 2) _mf_ 6 _!x_ 2( _x_ 2) _⇡_ [0 _._ 8671 _,_ 0 _._ 1329] _p_ ˜ _X_ 3 _|_ **_Y_** ( _x_ 3 _|_ **_y_** ) = _mf_ 3 _!x_ 3( _x_ 3) _mf_ 5 _!x_ 3( _x_ 3) _⇡_ [0 _._ 0860 _,_ 0 _._ 9140] ˜ _pX_ 4 _|_ **_Y_** ( _x_ 4 _|_ **_y_** ) = _mf_ 4 _!x_ 4( _x_ 4) _mf_ 6 _!x_ 4( _x_ 4) = [0 _._ 7990 _,_ 0 _._ 2010] 

_Page 11 - Please turn page_ 

**(16 points)** 

## **Problem 5: Principal Component Analysis** 

- a) You are given di↵erent examples of a mean vector _<u>µ</u>_ and a covariance matrix **C** _X_ . (4 pt) In a two-dimensional diagram, sketch samples from a data distribution qualitatively consistent with the following means and covariances. 

For which of the three distributions does PCA work best when reducing the dimension from two to one? Why? 

PCA works best for (iii) since the data is supported on a one-dimensional affine space. Thus, there is no loss in applying PCA with _K_ = 1. 

_Page 12 - Please turn page_ 

- = _)_ b) Find the largest eigenvalue and corresponding eigenvector (of unit length) of the matrix (4 pt) **C** _X_ in the following cases: 

   - (i) **C** _X_ = _<u>xx</u>_ T for some non-zero vector _<u>x</u> 2_ R<sup>_M_</sup> . 

   - (ii) **C** _X_ is the _M ⇥ M_ all-ones matrix. 

### **Solution:** 

- (i) For a unit length eigenvector _<u>v</u>_ of **C** _X_ , we have 

where (*) follows from the Cauchy-Schwartz inequality and is met with equality if _<u>x</u>_ and _<u>v</u>_ are colinear. Hence, the largest eigenvalue is _kxk_ which corresponds to the eigenvector _<u>v</u>_ = _<u>x/kxk</u>_ 

- (ii) This case corresponds to the case in (i) with _<u>x</u>_ = [1 _,_ 1 _, . . . ,_ 1]<sup>T</sup> . Thus, 

_Page 13 - Please turn page_ 

(8 pt) 

= _)_ c) Consider the probabilistic PCA (PPCA) model 

with **Q** _2_ R<sup>_M⇥K_</sup> , _<u>Z</u> ⇠N_ (0 _,_ **I** _K_ ), _<u>µ</u> 2_ R<sup>_M_</sup> and _<u>V</u> ⇠N_ (0 _, σ_<sup>2</sup> **I** _M_ ). 

- (i) Suppose we wish to estimate the above model parameters **Q** _, µ, σ_<sup>2</sup> for a given data set _<u>x</u>_ 1<sup>_, . . . , x_</sup> _~~N~~_<sup>.</sup> Write down the corresponding optimization problem in a nonlogarithmic, making the parameters _µ,_ **Q** _, σ_<sup>2</sup> explicit. 

- (ii) Let **Q** = [1 _,_ 1 _/_ 2]<sup>T</sup> , _<u>µ</u>_ = [ _−_ 1 _,_ 1]<sup>T</sup> , and _σ_<sup>2</sup> = 1 _/_ 4. Determine the probability distribution of _<u>X</u>_ and sketch its contour lines.. 

- (iii) When performing dimensionality reduction with PPCA, the data is shifted towards its mean. Explain this phenomenon. How does it depend on _σ_<sup>2</sup> ? 

### **Solution:** 

- (ii) Since _<u>Z</u>_ and _<u>V</u>_ are independent Gaussians, _<u>X</u> ⇠N_ (E[ _<u>X</u>_ <u>]</u> _,_ **C** _X_ ) is Gaussian as well. We have 

- (iii) This is because _<u>V</u>_ is considered as noise in our model. Since dimensionality reduction is performed via MMSE estimation from these ”noisy” data samples, the reconstructions will be shifted towards the data mean. The larger _σ_<sup>2</sup> , the more pronounced is the shift. 

_Page 14 - Please turn page_ 

**(14 points)** 

## **Problem 6: Expectation Maximization** 

Consider transmission of BPSK modulated symbols over an additive white Gaussian noise channel. Depending on which symbol was sent, the symbols experience di↵erent fading conditions. This can be modeled as 

The BPSK symbols _−x_ 0 and + _x_ 0 ( _x_ 0 _>_ 0) are sent equally likely. The Gaussian random variables _Z_ 1 and _Z_ 2 have zero mean and the respective variances _σ_ 1<sup>2and</sup><sup>_σ_</sup> 2<sup>2,i.e.,</sup><sup>_Z_1</sup><sup>_⇠_</sup> _N_ (0 _, σ_ 1<sup>2)and</sup><sup>_Z_2</sup><sup>_⇠N_(0</sup><sup>_, σ_</sup> 2<sup>2).Thereceivercollects</sup><sup>_N_observations</sup><sup>_yi, i_= 1</sup><sup>_, . . . , N_andtries</sup> to estimate the fading coefficients _h_ 1 and _h_ 2 solely based on the observed samples. 

= _)_ a) What is the loglikelihood function _L_ ( _h_ 1 _, h_ 2) for estimating the model parameters _h_ 1 and (2 pt) _h_ 2 from the observations _yi, i_ = 1 _, . . . , N_ ? Your expression should depend only on the observations, the model parameters and the distribution of _Z_ 1 and _Z_ 2. 

### **Solution:** 

Now we want to find a solution to the problem of maximizing _L_ ( _h_ 1 _, h_ 2) with respect to the model parameters using the expectation maximization algorithm. 

_Page 15 - Please turn page_ 

- = _)_ b) Derive the expression for the _i_ -th auxiliary distribution _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(</sup><sup>_x_)forthe</sup><sup>_t_-thiteration</sup> (4 pt) in the M-step. The resulting expression should depend only on the observations, the model parameters of the ( _t −_ 1)-th iteration, and the distribution of _Z_ 1 and _Z_ 2. 

= _)_ 

### **Solution:** 

- c) Derive the expression for the objective in the _t_ -th iteration of the E-step. Simplify the (4 pt) resulting term as far as possible. Any occurence of _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>_doesnot_havetobereplaced</sup> with the expression obtained in b). 

### **Solution:** 

_Page 16 - Please turn page_ 

- d) Formulate the necessary conditions for _h_<sup>_⇤_</sup> 1<sup>and</sup><sup>_h⇤_</sup> 2<sup>tobeoptimizersoftheobjectivein</sup> (4 pt) c). Give expressions of _h_<sup>_⇤_</sup> 1<sup>and</sup><sup>_h⇤_</sup> 2<sup>thatsolelydependon</sup><sup>_QX_</sup> _i_<sup>and</sup><sup>_yi_.</sup> 

_Page 17 - End of exam_

---

## 源文件

- [mlcomm_exam.pdf](attachments/documents/AI_Machine-Learning-in-Communication-cf890de50131/mlcomm_exam.pdf)
