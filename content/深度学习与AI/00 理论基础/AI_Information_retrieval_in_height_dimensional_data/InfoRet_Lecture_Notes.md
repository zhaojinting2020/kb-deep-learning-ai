---
title: InfoRet_Lecture_Notes
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-86147004342c/InfoRet_Lecture_Notes.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-86147004342c/InfoRet_Lecture_Notes.pdf
  title: InfoRet_Lecture_Notes.pdf
---

# **Information Retrieval In High Dimensional Data** 

### **Lecture Notes** 

Martin Kleinsteuber, Hao Shen, Matthias Seibert, Alexander Sagel 

November 8, 2018 

# Contents 

|**1**|**Intr**|**oduction**|**7**|
|---|---|---|---|
||1.1|Information Retrieval and Machine Learning<br>. . . . . . . . . . . . . . . .|7|
||1.2|Preliminaries on Random Variables . . . . . . . . . . . . . . . . . . . . . .|9|
|**2**|**Sta**|**tistical Decision Making and Machine Learning**|**11**|
||2.1|General Setting for Supervised Decision Making and the Generalization<br>Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|13|
||2.2|_k_-nearest neighbors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||2.3|Curse of dimensionality<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
|**3**|**Log**|**istic Regression**|**17**|
||3.1|Alternative approach to logistic regression . . . . . . . . . . . . . . . . . .|20|
||3.2|Linear Separability and Logistic Regression . . . . . . . . . . . . . . . . .|22|
|**4**|**Prin**|**cipal Component Analysis and Its Modern Interpretations**|**25**|
||4.1|A geometric interpretation . . . . . . . . . . . . . . . . . . . . . . . . . . .|25|
||4.2|Statistical Interpretation . . . . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||4.3|Error Model Interpretation<br>. . . . . . . . . . . . . . . . . . . . . . . . . .|28|
||4.4|Relation to Autoencoders . . . . . . . . . . . . . . . . . . . . . . . . . . .|29|
|**5**|**(De**|**ep) Feedforward Neural Networks**|**31**|
||5.1|Defnition and Motivation of FNNs . . . . . . . . . . . . . . . . . . . . . .|31|
||5.2|Training FNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|32|
||5.3|Multiclass Classifcation with FNNs<br>. . . . . . . . . . . . . . . . . . . . .|34|
|**6**|**Ker**|**nels and the Kernel Trick**|**37**|
|**7**|**Ker**|**nel Principal Component Analysis**|**41**|
||7.1|Linear PCA expressed with inner products. . . . . . . . . . . . . . . . . .|41|
||7.2|Transition to Kernel PCA . . . . . . . . . . . . . . . . . . . . . . . . . . .|43|
|**8**|**Sup**|**port Vector Machines**|**47**|
||8.1|Some Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|47|
||8.2|Basic Linear SVM<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|48|
|||8.2.1<br>Karush-Kuhn-Tucker Conditions . . . . . . . . . . . . . . . . . . .|49|
|||8.2.2<br>Lagrangian Duality . . . . . . . . . . . . . . . . . . . . . . . . . . .|50|
|||8.2.3<br>Linear SVM: Primal and Dual Problem<br>. . . . . . . . . . . . . . .|51|
||8.3|Soft Margin Linear SVM . . . . . . . . . . . . . . . . . . . . . . . . . . . .|52|

3 

8.4 Kernel SVM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55 **57** 

**Bibliography** 

4 

|Symbol|meaning|
|---|---|
|_X, Y, Z_|random variables, possibly mulivariate|
|_S, T_|Sets|
|_p_|dimension of “raw data space”|
|_k_|dimension of “reduced data space” or ”Feature Space”|
|**X**_,_**Y**_,_**Z**|matrices; _X_ often denotes the (centered) observation matrix|
|_X_|centered observation matrix|
|**a**_,_**b**_,_**c**_,_**u**_,_**v**_,_**w**|vectors|
|(_·_)<sup>_>_</sup>|transpose of a matrix or a vector|
|**a**<sup>_>_</sup>**b**|scalar (or _dot_-)product of **a** and **b**|
|tr**A**|trace of**A**_2_R<sup>_n⇥n_</sup>, i.e. the sum of the diagonal elements of**A**|
|tr(**A**<sup>_>_</sup>**B**)|scalar product for matrices|
|E[_X_]|expected value of _X_|
|Var[_X_]|covariance matrix of the multivariate random variable _X_; by<br>abuse of notation often also an estimation of the covariance<br>matrix|
|ˆ_µ,_ <sup>ˆ</sup>⌃_, . . ._|estimated quantities; often the hat is omitted|
|_pdf_|probability density function|
|_L_<br>�<br>(_·_)_,_(_·_)<br>�|loss function|
|_EPE_|expected prediction error|
|Pr(_X_)|probability measure induced by _X_|
|_W_(_X_ =_x_)|probability for random variable _X_ to take the value of _x_|
|⌦|probability space|
|_F_(⌦)|_σ_-algebra of ⌦|
|_µX_|induced probability measure|
|Pr(_Y |X_)|conditional probability measure of _Y_ given _X_|
|pr_U_|projection onto the subspace _U_|
|span(**u**1_, . . . ,_**u**_k_)|subspace spanned by the vectors **u**1_, . . . ,_**u**_k_|
|diag(_a_1_, . . . , am_)|short for an _m ⇥m_ matrix with only the diagonal entries<br>_a_1_, . . . , am_ being di↵erent from zero|
|_Ik_|_k ⇥k_ identity matrix|
|1_n_|vector in R<sup>_n_ </sup>consisting of 1_n_ = [1_, . . . ,_1]<sup>_>_</sup>|

5 

# 1 Introduction 

## 1.1 Information Retrieval and Machine Learning 

The problem of extracting information out of high dimensional data is inseparably connected to the question of dimensionality reduction, i.e. to the problem of extracting _a few_ reasonable features out of typically high dimensional observations. Consider for example the humans ability to recognize faces on an image. The image, seen as a high dimensional vector of, say 800 _⇥_ 600 pixel values, can surely not be stored in a humans brain as _raw_ pixel-data. Instead, there must be some features that we extract, which may be e.g. relative distance between eyes, length of the nose, and more abstract the interplay of di↵erent face regions as a whole. The ability of storing and recalling these few abstract features make it possible to recognize a face regardless of varying background, sunglasses or partial occlusions and to distinguish between di↵erent faces. There are more examples in the wide field of data analysis, where the extraction of features allows to squeeze information out of high dimensional data, ranging from gene data classification to audio signal processing, from data visualization to the analysis of electroencephalography (EEG) data. 

Formally, the problem of dimensionality reduction is the following: Given a _p_ -dimensional _>_ real valued random variable _X_ = ⇥ _X_ 1 _. . . Xp_ ⇤ , find a map or algorithm 

such that _S_ = _f_ ( _X_ ) contains “as much information from _X_ as possible”. In the spirit of the above mentioned example, R<sup>_p_</sup> will be referred to as _raw data space_ and R<sup>_k_</sup> as _reduced data space_ or _feature space_ . 

For example the preservation of information might be measured by the variance, so that the variance of _S_ should reflect the variance of _X_ . This can also be interpreted as to eliminate redundancy in the data. Consider the following example: the temperature is measured, one time in degree Celcius (which would be the first entry _X_ 1 of the random variable) and in degree Fahrenheit ( _X_ 2). It is obvious that this information can be reduced to one variable, say _S_ 1 = _X_ 1, even without any loss of information. 

The matrix **X** _⇢_ R<sup>_p⇥n_</sup> where its ( _i, j_ )-entry _xij_ denotes the realization of the random variable _Xi_ at observation _j_ is called _observation matrix_ . Its columns are realizations of the _p_ -dimensional random variable _X_ . 

The expected value is referred to by E( _X_ ) = _µ 2_ R<sup>_p_</sup> . Since we are dealing with a multivariate random variable, the variance is now expressed by the _covariance matrix_ (also called the variance-covariance matrix) which is defined as 

7 

Its ( _i, j_ )-entry is the covariance between the _i_<sup>th</sup> and the _j_<sup>th</sup> random variables. The covariance matrix is symmetric, i.e. ⌃= ⌃<sup>_>_</sup> , and positive semidefinite<sup>1</sup> , i.e. ⌃ _≥_ 0 _, x_<sup>_>_</sup> ⌃ _x ≥_ 0 _8x_ . 

**Example 1.1.** Consider two constant random variables _X_ 1 _⌘_ const. _, X_ 2 _⌘_ const.. This means we have a 2-dimensional random variable with the covariance matrix ⌃= 0. This example shows that ⌃is not necessarily positive definite. 

As the actual distribution of a random variable is typically unknown, the expectation value is usually estimated on the base of _n_ observations: 

Using this estimated expectation and the Kronecker product<sup>2</sup> denoted by _⌦_ , the centered observation matrix **X**<sup>¯</sup> may be computed as follows: 

With the centered observation matrix X,<sup>¯</sup> the covariance matrix ⌃= Cov( _X_ ) can be estimated by 

Since in practice _n_ tends to be large, it is also possible to use the approximation _n_<sup><u>1</u></sup><sup>**X**¯¯</sup><sup>**X**</sup><sup>_>_.</sup> 

> 1in contrast to positive definite, i.e. _x>_ ⌃ _x >_ 0 _8x 6_ = 0 and _x>_ ⌃ _x_ = 0 _, x_ = 0 

8 

## 1.2 Preliminaries on Random Variables 

We want to recall some basic definitions and notations from probability theory that we will need occasionally throughout this lecture notes. It will be sufficient for our purposes to consider _real multidimensional_ random variables that are either continuous or discrete. More formally, let _X_ : ⌦ _!_ R<sup>_p_</sup> be a random variable and denote its density with respect to the usual Lebesgue measure as _pX_ ( _x_ ). We will use the very sloppy but very convenient notation _X 2_ R<sup>_p_</sup> to indicate that the random variable _X_ takes values in R<sup>_p_</sup> . 

For (absolutely) continuous random variables, the density is a continuous function from R<sup>_p_</sup> to R. In case of a discrete random variable which takes values _xi_ with probability _pi_ , we employ the Dirac-Delta-Function<sup>3</sup> to describe its density, namely 

So, if _A ⇢_ R<sup>_p_</sup> is a subset<sup>4</sup> , the probability that _X_ takes values in _A_ is given by 

Note, that in the case of a discrete random variable, this expression is just 

By knowing the joint density _pX,Y_ ( _x, y_ ) of two random variables _X 2_ R<sup>_p_</sup> and _Y 2_ R<sup>_k_</sup> , it is possible to deduce the individual densities of _X_ and _Y_ , respectively. These are called the _marginal_ densities, and they are given by 

If the joint density function is given, the knowledge of a certain realization of one of the two variables, say _X_ , allows to infer information about the distribution of _Y_ . The resulting density function is called _conditional density function_ , and, if the realisation of _X_ is _x 2_ R<sup>_p_</sup> , it is given by 

> 3The Dirac-Delta-Function fulfills the condition that _δ_ ( _t_ ) = 0 for _t 6_ = 0 and RR<sup>_p δ_(</sup><sup>_t_)d</sup><sup>_t_= 1</sup><sup>_p_.i.e.</sup><sup>_δ_has</sup> an infinitely high peak at 0. 

> 4Formally, this set has to be _measurable w.r.t. the Borel σ-algebra_ , but if you do not know what measurable is, all subsets that you can imagine satisfy this condition. 

9 

There are two quantities that play a prominent role in describing statistical properties of a random variable _X 2_ R<sup>_p_</sup> . These are the first and the second moment, also known as the _expectation value_ 

and the _variance/covariance_ 

Note, that _µ 2_ R<sup>_p_</sup> and that Var[ _X_ ] is a positive semidefinite matrix in R<sup>_p⇥p_</sup> . Exercise: Show that the variance/covariance matrix is positive semidefinite. 

||_x_1|_x_2|_x_3|_x_4|_py_(_Y_) _#_|
|---|---|---|---|---|---|
|_y_1|1<br>8|1<br>16|1<br>32|1<br>32|1<br>4|
|_y_2|1<br>16|1<br>8|1<br>32|1<br>32|1<br>4|
|_y_3|1<br>16|1<br>16|1<br>16|1<br>16|1<br>4|
|_y_4|1<br>4|0|0|0|1<br>4|
|_px_(_X_)<br>_!_|1<br>2|1<br>4|1<br>8|1<br>8|1|

Table 1.1: This table shows an exemplary joint probability distribution. 

**Example 1.2.** An example of a joint probability distribution of a two dimensional discrete random variable is given in Table 1.1. The marginal densities are denoted by _pY_ ( _y_ ) and _pX_ ( _x_ ), respectively. As an exercise, compute the conditional density of _X_ given _Y_ = _y_ 2. The answer is given in this footnote<sup>5</sup> . 

> 5Answer: _pX|Y_ = _y_ 2 ( _x_ ) =<sup>~~P~~</sup> _i_<sup>_piδ_(</sup><sup>_x −xi_),with</sup><sup>_p_1= 1</sup><sup>_/_4,</sup><sup>_p_2= 1</sup><sup>_/_2,</sup><sup>_p_3= 1</sup><sup>_/_8,</sup><sup>_p_4= 1</sup><sup>_/_8.</sup> 

10 

# 2 Statistical Decision Making and Machine Learning 

The basic problem is that for some observation of the random variable _X 2_ R<sup>_p_</sup> , we want to obtain the ”most probable” value of the random variable _Y_ . For simplicity, we assume that _Y_ takes realizations in R. We further assume for the moment that the induced joint probability density _pX,Y_ ( _x, y_ ) is given. 

**Example 2.1.** _X_ is a (vectorized) image of a person. In Figure 2.1 below this observation is represented a picture that contains just one pixel with varying levels of gray. 

The aim is to predict _Y_ for a given _X_ . If person A is wearing predominantly dark clothes 

Figure 2.1: Joint PDF of the two discrete random variables _X_ and _Y_ . 

it is reasonable to assume that the respective probabilities for the events behave like this: 

_p_ 42 _> p_ 32 _> p_ 22 _> p_ 12 and _p_ 11 _> p_ 21 _> p_ 31 _> p_ 41. 

In order to formalize what we mean by ”most probable”, we consider the _quadratic loss function_<sup>1</sup> 

> 1Note, that also other loss functions are possible and also do make sense indeed, and that the choice of the squared distance is only due to convenience reasons. 

11 

We want to choose _f_ such that the expected value of the loss function, the so called Expected Prediction Error 

is minimized. A simple computation that uses to tools described in subsection 1.2 yields 

and hence _f_ ( _X_ ) = arg min E _Y |X_ = _x_ ⇥( _Y − c_ )<sup>2⇤</sup> . Due to the linearity of E[ _·_ ], this reduces _c_ to 

The quadratic term can easily be minimized with the optimal value 

**Theorem 2.2.** _The best prediction of Y given X is the_ conditional mean _, if_ best _is measured with respect to the squared loss._ 

As stated above, the conditional mean as best prediction relies on the fact that we use the squared loss as a quality measure. One can show that if we employ the absolute value _|Y − f_ ( _X_ ) _|_ as a loss function - the so-called _`_ 1 _-loss_ - instead, the best prediction is the _conditional median_ . Although a rigorous proof of this statement is beyond the scope of this lecture notes, we can easily see that the median is optimal for the minimization of the _empirical `_ 1-loss 

All we need is a result from non-smooth convex optimization which states that for convex functions with subgradients, the minimum _c_<sup>_⇤_</sup> is achieved if and only if the subgradient at _c_<sup>_⇤_</sup> contains 0. In our case, the subgradient of the _empirical `_ 1-loss with respect to _c_ is _n_<sup><u>1</u></sup> P _i_<sup>sign(</sup><sup>_yi−c_)for</sup><sup>_c_=</sup><sup>_yi_and</sup> n _n_ <u>1</u> P _i_ = _j_<sup>sign(</sup><sup>_yi−c_) +</sup><sup>_t| −_1</sup><sup>_t_1</sup> o for _c_ = _yj_ . So we see that the subgradient at _c_ contains zero if an only if there are as many _yi_ ’s that are _smaller_ than _c_ than there are _yi_ ’s that are _larger_ . In case of odd _n_ , _c_ has to coincide with the ( _n_ + 1) _/_ 2-th greatest value of the _yi_ ’s. This is exactly the definition of the median. 

12 

## 2.1 General Setting for Supervised Decision Making and the Generalization Error 

Let us resume the above section on a higher level perspective. We have introduced a loss function that, based on training samples ( _xi, yi_ ) _, i_ = 1 _, . . . , N_ , allows to learn the _best_ prediction function _f_ out of a given function class _F_ . Such methods are called _supervised learning methods_ , since they require a training set where to each sample _xi_ we have a corresponding _yi_ . The general problem statement is: 

_Let_ ( _X, Y_ ) _2_ R<sup>_p_</sup> _⇥_ R _be a random variable and let F be a class of functions from_ R<sup>_p_</sup> _!_ R _that can be parameterized by finitely many parameters, say_ ⇥ _2_ R<sup>_M_</sup> _. Moreover, let L_ : R _⇥_ R _!_ R<sup>+</sup> 0<sup>_beaLossfunction,thatmeasuresthedeviationofYandf_(</sup><sup>_X_)</sup><sup>_.The_</sup> _aim of a supervised prediction method is then to find f_<sup>ˆ</sup> _2 F that minimizes the expected prediction error_ E[ _L_ ( _Y, f_ ( _X_ )] _._ 

In fact, there are three issues that we need to clarify in order to build a supervised machine learning method. 

- We have to specify the loss function _L_ . 

- We have to specify the function class _F_ . 

- Since in practice, we do not know the joint distribution of ( _X, Y_ ), we need training samples ( _xi, yi_ ) _, i_ = 1 _, . . . , N_ to approximate the expected prediction error. 

Once this is done, we arrive at the optimization problem 

These optimization problems can be more or less hard to solve, but in principle, they can be fed into an optimization toolbox in the above form. 

**Example: Linear Regression** . For linear regression, we choose _L_ to be the quadratic loss, i.e. _L_ ( _Y, f_ ( _X_ )) = ( _Y − f_ ( _X_ ))<sup>2</sup> , and we choose _F_ to be the class of affine functions 

where _x_<sup>(</sup><sup>_k_)</sup> are the entries of the vector _x_ . Given _N_ training samples, the optimization problem then is 

13 

**Definition:** The di↵erence between the empirical loss and the expected loss for a given function _f_ 

is called the generalization error of _f_ . 

We know by the law of large numbers, that _GN_ tends to 0 as _N_ tends to infinity. However, very often the number of training samples is limited, and moreover, the speed of converges highly depends on the (unknown) probability distribution of ( _X, Y_ ). One goal in Machine Learning is therefore to bound the generalization error _with high probability_ . In practice, we use cross validation to check how well _f_ fits the data. TODO: Explain cross validation!! 

TODO: Explain Overfitting!! 

## 2.2 _k_ -nearest neighbors 

In practice, we face the problem that we do not know the joint distribution of _X_ and _Y_ and we thus have to estimate the conditional mean in Eq. (2.3). Typically, this would be done with the relative frequency of _Y_ given the particular realization _x_ . However, in the continuous case the problem arises, that even among many observations ( _xi, yi_ ), there is probably none with _xi_ = _x_ . This problem is resolved by enlarging the region around _x_ and taking all observations into account where _xi_ is _near_ the desired _x_ . This leads to the concept of _k_ -nearest neighbors. 

Figure 2.2: Given the events (1 _,_ 1) _,_ (2 _,_ 5) _,_ (3 _,_ 3) _,_ (4 _,_ 1) _,_ (5 _,_ 4) for the bivariate distribution ( _X, Y_ ). While E _Y |X_ =4[ _Y_ ] = 5, averaging over the 3 nearest neighbors provides _f_<sup>ˆ</sup> _k_ =3(4) = (2 + 3 + 5) _/_ 3 = 10 _/_ 3. 

14 

The approach of _k_ -nearest neighbors is to estimate _f_ by 

where _Nk_ ( _x_ ) denotes the set of the _k_ nearest neighbors of _x_ . There are two approximations made here: 

1. Expectation is approximated by averaging. 

2. Conditioning at a point is approximated by conditioning in some region (around that point). 

If _N_ is the number of observations, the following is to be noted. If _N_ increases, the _xi_ come close to _x_ . Moreover, if _k_ increases, the average tends towards the expectation value. More precisely, under mild conditions on the joint probability measure, we have 

However, the sample size _N_ is generally limited, and therefore not large enough. There are two ways to overcome this problem: 

- by imposing model assumptions on _f_ , e.g. _f_ ( _x_ ) _⇡ x_<sup>T</sup> _β_ gives linear regression. 

- by “reducing” the dimension of _X_ . This leads to the motivation for dimensionality reduction: 

15 

## 2.3 Curse of dimensionality 

This section will give the motivation for dimensionality reduction as a whole. The term _curse of dimensionality_ was coined by Bellman in 1961. It refers to several phenomena that occur when analyzing data in high dimensional space and interfere with statistical decision making. 

Let _X 2_ R<sup>_p_</sup> be random variable. There are several things to be observed: 

First, the absolute noise in _X_ increases with the number of features (i.e. with _p_ ) since the noise accumulates over all dimensions. 

Second, the number of observations required for estimating the density function of _X_ increases dramatically with the dimension _p_ . Density functions are usually estimated by the help of the relative frequency of an incidence. As an example assume that _N_ observations in a 1-dimensional space are required to estimate a certain density function up to a predefined accuracy. If the observation space is increased to 2 dimensions, the number of observations has to be increased to order _N_<sup>2</sup> to maintain the same accuracy. For a 3-dimensional space around _N_<sup>3</sup> observations are required, and so forth. This means that the number of measurements required to give the same accuracy for an estimation of a density function increases exponentially with the dimension of the measurement space. 

One cause for the curse of dimensionality is the _empty space phenomenon_ (Scott), which states that high dimensional spaces are inherently sparse. Given data points uniformly distributed in a 10-dimensional unit sphere, the probability that a point is closer than 0 _._ 9 to the center is less than 35%. Therefore, the tail in high-dimensional distributions is far more important than in one-dimensional ones. Given a high dimensional multivariate random variable the fact that one component lying in the tail of its distribution is enough to cause the whole sample to lie in the tail of the common density function. The 1-dimensional case is illustrated in Figure 2.3. 

Figure 2.3: With one-dimensional random variables most observations will be around the mean value. This is not the case for higher dimensions. 

16 

# 3 Logistic Regression 

The general setting when talking about logistic regression is that we have data points _X 2_ R<sup>_p_</sup> and output variables _Y 2 {−_ 1 _,_ +1 _}_ . This is a so-called binary classification problem. 

The task then is to find the function _f_ in a predefined class of functions _F_ such that sign _f_ ( _X_ ) predicts _Y_ as good as possible. A commonly used loss function to measure the ‘accuracy’ of the predicting functions is motivated by the number of misclassifications, i.e. if the sign of _f_ ( _X_ ) does not coincide with the sign of the true output _Y_ . The so called 0 _,_ 1-loss function 

does the job. 

In order to find an optimal prediction function _f 2 F_ for a given set of training samples ( _xi, yi_ ) _i_ =1 _,...,n_ the goal is to find _f 2 F_ that minimizes the empirical expected loss 

However, it is numerically infeasible to find a minimum of this problem. Even just considering the class of affine functions _F_ a↵, this is hard to solve numerically, since the loss function is neither continuous nor di↵erentiable. In order to make it easier to solve we approximate the (non-continuous, non-convex) function _L_ 0 _,_ 1 by a convex loss function. 

#### **Interlude: Convexity** 

**Definition 3.1.** Let _C ⇢_ R<sup>_n_</sup> be a convex set, i.e. for any pair of elements **x** 1 _,_ **x** 2 _2 C_ , the point _t_ **x** 2 + (1 _− t_ ) **x** 1 is also an element of _C_ for all _t 2_ [0 _,_ 1]. A function _f_ : _C !_ R is called _convex_ if _tf_ ( **x** 2) + (1 _− t_ ) _f_ ( **x** 1) _≥ f_ ( _t_ **x** 2 + (1 _− t_ ) **x** 1) for all **x** 1 _,_ **x** 2 _2 C, t 2_ [0 _,_ 1]. It is called _strictly convex_ if the inequality is strict. 

Example: _f_ : R<sup>+</sup> _!_ R _, x 7!_ 1 _/x_ is convex. 

**Theorem 3.2.** _If f, g are convex, then so are_ 

_• h_ = max( _f, g_ ) 

_• h_ = _f_ + _g • h_ = _g ◦ f if g is non-decreasing_ 

17 

BRE efowe sofa) += often 4 +¢(-Vfo) a 

as eRe eA ttt ar 

~~<u>vglw) =</u>~~ <u>|</u> ~~<u>| aaaee 7 =</u>~~ <u>T</u> ~~<u>A exp (I w'k ) d z yee) 0] My)= (YX) a[at ep yr wte | ionep</u>~~ <u>Ye</u> ~~<u>wt) gS TH = /— ext lye”)</u>~~ <u>|</u> ~~<u>ye ae exp</u>~~ <u>(</u> ~~<u>yi wd) Xi o(yiwa) A oly we)</u>~~ 

~~<u>i</u>~~ <u>p</u> ~~<u>a /izZadiKZ 44 24 quidiors Elgcu eee) aapCbd nh toh C1, Ab BME (ox,4) {r</u> aii a oo,~~<sup><u>Ww)))XpMp</u></sup> ~~<u>Hes MK 91 435 B= (>) 84°F Av f(a) ors (=p Kp > ) Cle ACT</u>~~<sup><u>Ap</u></sup> <u>> Fun AZe2.10 4</u> ~~HINARI~~ <u>42 Re Py</u> ~~ieaqs -s acy € 10,1) « 8S 30. ISO —ne Li- me) >~~ ~~<u>noe nl -4p%p 5 Ci ACHR9) 4 Sp 0,</u> pelP~~ <u>Ape</u> ~~<u>7°45)2 2</u>~~ <u>Sp</u> ~~<u>M Ao } qekrlo 7-H, PER 2.10 PEI , ho ta sh Fo HOTS (4,20) of 34 7 1th APR. 774,</u> 5)~~ ~~<u>44</u> Hi Spe odgeHotd Ik, 4.3)~~ ~~<u>Ren G4.</u> Giz fbn /id RADIA~~ ~~<u>ZO,</u>~~ 

We also use the auxiliary function _g_ ( _z_ ) = 1 _/_ (1 + _e_<sup>_−z_</sup> ), where _g_<sup>_0_</sup> ( _z_ ) = _g_ ( _z_ )(1 _− g_ ( _z_ )). Then, the first and second partial derivative for _F_ are 

and 

where _yi_<sup>2= 1.Toshowthatthefunctionisnon-negativedefinite,weneedtoshowthat</sup> _a_<sup>_>_</sup> _r_<sup>2</sup> _Fa ≥_ 0 for all _a_ . We define the auxiliary variables _Pi_ = _g_ ( _yi_ **w**<sup>_>_</sup> **x** _i_ )(1 _− g_ ( _yi_ **w**<sup>_>_</sup> **x** _i_ )) and _⇢_<sup>(</sup> _i_<sup>_j_)</sup> = _x_<sup>(</sup> _i_<sup>_j_)</sup> _pPi_ . Then 

This holds for any concatenation of convex and affine functions. 

In the following, we give a probabilistic interpretation of this optimization problem. First of all, note that the conditional probability of _Y_ = _y_ given the observation **x** is as 

In order to find a solution to (3.3), it is common practice to use gradient based methods. The simplest form is gradient descent where at a given iteration point, we compute the gradient and then take a step in the negative direction of that gradient. The gradient of the function _F_ ( **w** _, b_ ) =<sup>P</sup> _i_<sup>log(1+exp(</sup><sup>_−yi_(</sup><sup>**w**</sup><sup>_>_</sup><sup>**x**</sup><sup>_i_+</sup><sup>_b_))) can be determined by computing</sup> the partial derivatives 

The factors 1 _/_ (1 + exp( **w**<sup>_>_</sup> **x** _i_ + _b_ )) and 1 _/_ (1 + exp( _−_ **w**<sup>_>_</sup> **x** _i_ + _b_ )) are the probability of incorrect prediction. (cf. (3.4)) 

Thus, when we take a step in the direction of the negative gradient, we go in the ‘opposite direction’ of these mistakes. This is the reason why gradient descent methods are also called _mistake driven methods_ . Mistakes of the current model (here given by 

19 

some weights ( **w** _, b_ )) are used to improve it. The gradient points in the direction to best minimize mistakes in the current model. 

In summary: 

Logistic regression is a supervised classification method where the decision function is affine and the loss is measured with _L_ ( _y, f_ ( **x** )) = log(1+ _e_<sup>_−yf_(</sup><sup>**x**)</sup> )). The best parameters **w**<sup>_?_</sup> _, b_<sup>_?_</sup> are found by minimizing the empirical expected loss, i.e. min **w** _,b N_<sup><u>1</u></sup> P log(1 + _e_<sup>_−yi_(</sup><sup>**w**</sup><sup>_>_</sup><sup>**x**</sup><sup>_i_+</sup><sup>_b_)</sup> ). Once the optimal **w**<sup>_?_</sup> _, b_<sup>_?_</sup> are determined, we can classify a new data point **x** new by computing sign( **w**<sup>_?>_</sup> **x** new + _b_<sup>_?_</sup> ). We can also compute the probability that this classification is correct via Eq. (3.4). 

## 3.1 Alternative approach to logistic regression 

First, note that the name logistic regression can be misleading, since in fact logistic regression is not a regression method, but a classification method. While the previous section was more optimization-motivated, here we provide a more statistical approach to logistic regression. 

**Example** : We try to predict the probability of death given some parameters. Let _x_ 1 be the age of a person, _x_ 2 the gender (0 corresp. to male, 1 corresp. to female), and _x_ 3 the cholesterol level. We assume that we can combine these values in a linear fashion so get a real value that is in some way correlated to the probability of death 

with **x** = [1 _, x_ 1 _, x_ 2 _, x_ 3]<sup>_>_</sup> , **w** = [ _w_ 0 _, w_ 1 _, w_ 2 _, w_ 3]<sup>_>_</sup> . The values _wi_ are called weights, _w_ 0 is called the o↵set. The resulting value is in R. To shape this into a probability, we need a function _σ_ that squashes this value into the interval [0 _,_ 1]. A function that achieves this is the logistic function 

The resulting model is _P_ (death _|x_ ) = _σ_ ( **w**<sup>_>_</sup> **x** ). 

More general, we consider the training data for a binary classifiaction problem _D_ = _{_ ( **x** 1 _, z_ 1) _, . . . ,_ ( **x** _n, zn_ ) _}_ , **x** _i 2_ R<sup>_d_</sup> , _zi 2 {_ 0 _,_ 1 _}_ and model the dependency of input and output variable as _zi /_ Bernoulli( _σ_ ( **w**<sup>_>_</sup> **x** )), where we assume the _zi_ to be independent. 

To train this model, we want to find the maximum likelihood estimate of **w** given _D_ , i.e. 

with 

20 

For optimization purposes, it is common to use the negative log of the above conditioned probability, namely 

The respective gradient w.r.t. **w** is given by 

with **X** = [ **x** 1 _, . . . ,_ **x** _n_ ] _2_ R<sup>_d⇥n_</sup> . The corresponding Hessian w.r.t. **w** is given as 

with **B** = diag( _σ_ ( **w**<sup>_>_</sup> **x** _i_ )(1 _− σ_ ( **w**<sup>_>_</sup> **x** _i_ ))) _2_ R<sup>_n⇥n_</sup> . The Hessian is positive semi-definite (Task: show this) and therefore _L_ is convex. 

Assume that the Hessian is invertible. Then the iteration of Newton’s method has the form 

with **r** _t_ = **X**<sup>_>_</sup> **w** _t −_ **B**<sup>_−_1</sup> ( _σ_ ( **X**<sup>_>_</sup> **w** _t_ ) _−_ **z** ). This is the solution to the weighted least squares problem arg min **w** P _i_<sup>_bi_(</sup><sup>_ri −_</sup><sup>**w**</sup><sup>_>_</sup><sup>**x**</sup><sup>_i_)2.</sup> 

**Exercise.** Show, that Equation (3.7) is equivalent to Equation (3.3) up to the scalar factor 1 _/n_ . 

**Proof.** First, note that log( _σ_ ( **w**<sup>_>_</sup> **x** )) = _−_ log(1+exp( _−_ **w**<sup>_>_</sup> **x** )) and log(1 _−σ_ ( **w**<sup>_>_</sup> **x** )) = _−_ log(1 + exp( **w**<sup>_>_</sup> **x** )). Thus, Equation (3.7) is equivalent to 

Since the _zi_ are either 0 or 1, we can rewrite the sum as 

which coincides with Equation (3.3) up to the factor 1 _/n_ . ⇤ 

21 

## 3.2 Linear Separability and Logistic Regression 

It is important to be aware that logistic regression can overfit on linearly separable training sets. When the two classes 1 and _−_ 1 are linearly separable, we can find a hyperplane ( **w** _s, bs_ ) such that the following inequalities hold. 

Consider the following theorem. 

**Theorem 3.4.** _For linearly separable, non-empty training sets, the loss function_ 

_has no global minimum in_ R<sup>_p_+1</sup> _._ 

**Proof.** Let us first characterize a global minimum of _F_ . It is a point ( **w**<sup>_⇤_</sup> _, b_<sup>_⇤_</sup> ) _2_ R<sup>_p_+1</sup> such that 

holds. If the training set is non-empty, then the loss function is strictly positive. Therefore, the minimal value of _F_ is some positive number _"_ , i.e. 

We will now show the lack of the global minimum by contradiction. Assume that there is a point ( **w** _s, bs_ ) and a real number _" >_ 0 such that the following to conditions hold. 

For each _i_ , define the following scalar value 

Consider the function 

It is easy to see that _⇠i_ is strictly positive for any _i_ and therefore _fi_ ( _h_ ) approaches 0 as _h_ approaches _1_ . The same is true if we consider the sum over _i_ , i.e. 

In words, for any _" >_ 0, we can find a real number _⌘>_ 0, such that the following inequality holds for any _h ≥ ⌘_ . 

22 

This directly contradicts the assumption (3.18), because we can choose **w** = _h_ **w** _s_ and _b_ = _hbs_ with _h ≥ ⌘_ . ⇤ 

The proof shows that, once a separating hyperplane is found, the value of the loss function can be always further decreased by increasing the magnitude of the hyperplane parameters. Note that this is true for _any_ hyperplane that separates the classes. In practice, an optimization algorithm could choose a hyperplane with a ”bad” position and orientation and increase the magnitude of the parameters until the maximum number of iterations is reached. 

To prevent this scenario, it is common to penalize the magnitude of ( **w** _, b_ ) by introducing a regularizer, e.g. by fixing a real constant _λ >_ 0 and adjusting the original cost function as 

23 

# 4 Principal Component Analysis and Its Modern Interpretations 

**Unsupervised Learning Methods** : Input variable is _X 2_ R<sup>_n_</sup> , output variable is the _reduced variable S 2_ R<sup>_k_</sup> . _S_ is computed from _X_ via some map _f_ that minimizes a particular Loss function _L_ ( _f_ ( _X_ )). Loss functions in unsupervised learning aim at reducing ”the volume of the distribution of the input variable”. By this, we mean that samples of _f_ ( _X_ ) concentrate in a much smaller volume than samples of _X_ . In many cases, this is done by doing _unsupervised dimensionality reduction_ , i.e. by choosing _k_ smaller than _n_ . Also clustering, where _S_ consists of finitely many cluster centers, reduces the volume of the input distribution. 

Among all unsupervised dimensionality reduction techniques Principal Component Analysis (PCA) is the most famous one. We refer to unsupervised learning methods, if the data does not have to be labeled (by a _supervisor_ ) before we employ it for the learning algorithm. 

The success of PCA is due to its simplicity and broad applicability in many real world data analysis tasks. This may be the reason that it has many aliases, namely the discrete Karhunen–Lo´eve transform, the Hotelling transform or proper orthogonal decomposition. Its core assumption is that the distribution of the raw data is concentrated around some lower dimensional plane, or, equivalently, that most of the variance in the data can be described by the variance of its projection onto this plane. 

## 4.1 A geometric interpretation 

The geometric interpretation of PCA is that it reduces the dimension by projecting the centered data onto a lower dimensional subspace and describes it with coordinates of an appropriate basis of that subspace. Figure 4.1 illustrates this interpretation. 

We can formalize this task as follows. Let _k < p_ (usually _k ⌧ p_ ) and **X** = ⇥ **x** 1 _, . . . ,_ **x** _n_ ⇤ _2_ R<sup>_p⇥n_</sup> be the centered observation matrix. Recall, that centered means<sup>P</sup> _i_<sup>**x**</sup><sup>_i_=0.The</sup> orthogonal<sup>1</sup> projection _⇡U_ : R<sup>_p_</sup> _!_ R<sup>_p_</sup> onto a _k_ -dimensional subspace _U ⇢_ R<sup>_p_</sup> can be described with help of an orthogonal basis of _U_ . More precisely, if we denote this basis by the matrix **U** _k 2_ R<sup>_p⇥k_</sup> (i.e. with orthonormal columns **U**<sup>_>_</sup> _k_<sup>**U**</sup><sup>_k_=</sup><sup>**I**</sup><sup>_k_)theprojectionis</sup> given by 

> 1If nothing else is mentioned, _orthogonal_ is always with respect to the standard scalar product. 

25 

AMeAyR 4B Variance 

_Proof._ First, Let **Q** _k 2_ R<sup>_p⇥k_</sup> with **Q**<sup>_>_</sup> _k_<sup>**Q**</sup> _k_<sup>=</sup><sup>**I**</sup> _k_<sup>.Notethatsince</sup><sup>_k_</sup><sup>**x**</sup><sup>_i−_</sup><sup>**Q**</sup> _k_<sup>**Q**</sup><sup>_>_</sup> _k_<sup>**x**</sup><sup>_ik_</sup> 2<sup>2=</sup> _k_ **x** _ik_<sup>2</sup> 2<sup>_−_</sup><sup>**x**</sup><sup>_>_</sup> _i_<sup>**Q**</sup> _k_<sup>**Q**</sup><sup>_>_</sup> _k_<sup>**x**</sup><sup>_i_,minimizing</sup><sup>_J_isequivalenttomaximizingP</sup> _i_<sup>**x**</sup> _i_<sup>_>_</sup><sup>**Q**</sup> _k_<sup>**Q**</sup><sup>_>_</sup> _k_<sup>**x**</sup> _i_<sup>.Then</sup> 

The _n ⇥ n_ matrix **U**<sup>_>_</sup> **Q** _k_ **Q**<sup>_>_</sup> _k_<sup>**U**isarank</sup><sup>_k_projectionmatrix,henceitsdiagonalentries</sup> are all between 0 and 1. To see this, let **P** be an orthogonal projection matrix. Its _i_ -th diagonal entry is **e**<sup>_>_</sup> _i_<sup>**Pe**</sup><sup>_i_,where</sup><sup>**e**</sup><sup>_i_isthe</sup><sup>_i_-thstandardbasisvector.Since</sup><sup>**P**isan</sup> orthogonal projection matrix, we have 

Since **P** is an orthogonal projection, ( **I** _−_ **P** ) is the projection onto the orthogonal complement and every vector in R<sup>_n_</sup> can be written as a unique composition of elements from the projection spaces. In particular, this is true for standard basis vectors, i.e. 

The last equality holds since **Pe** _i_ and ( **I** _−_ **P** ) **e** _i_ are orthogonal to one another. The summands _k_ **Pe** _ik_<sup>2</sup> and _k_ ( **I** _−_ **P** ) **e** _ik_<sup>2</sup> are both positive and add up to 1. Therefore, _k_ **Pe** _ik_<sup>2</sup> lies in the interval [0 _,_ 1] for all _i_ . Besides, the trace of **P** , i.e. the sum of all diagonal elements is equal to _k_ . So the maximum value we can achieve is to set **Q** _k_ = **U** _k_ , for **I** _k_ 0 then this projector is **U**<sup>_>_</sup> **U** _k_ **U**<sup>_>_</sup> _k_<sup>**U**=</sup> 0 0 . " # 

To see that the reduced variables **S** := **U**<sup>_>_</sup> _k_<sup>**X**areuncorrelated,notefirstthatsince</sup><sup>**X**</sup> is centered, so is **S** , and therefore its empirical covariance matrix is given by 

The ( _i, j_ )-entry of the matrix **S** is called the _j_ - _th score_ for the _i_ - _th principal component_ , the complete matrix is called _score matrix_ . The matrix **U** is also often referred to as 

27 

_loadings matrix_ .<sup>3</sup> Note, that **S** can directly be obtained by the SVD of **X** , since 

where **V** _k_ stands for the first _k_ columns of **V** . 

While the reduced variables can be computed via **U**<sup>_>_</sup> _k_<sup>**X**this is not feasible in practical</sup> applications since both **U** _k_ and **X** are dense ((2 _p −_ 1) _nk_ operations required to compute **S** ). Using **D** _k_ **V** _k_<sup>_>_ismuchcheapersince</sup><sup>**D**</sup> _k_<sup>isdiagonaland</sup><sup>**V**</sup> _k_<sup>_>_onlyhas</sup><sup>_k_rows(</sup><sup>_nk_</sup> operations required to compute **S** ). 

## 4.2 Statistical Interpretation 

Other than minimizing the reconstruction error _J_ ( **U** _k_ ), as defined in (4.2), PCA can be interpreted from a statistical point of view. If _X 2_ R<sup>_p_</sup> denotes a multidimensional random variable, PCA seeks for an orthogonal transformation to a new coordinate system, i.e. 

with **U**<sup>_>_</sup> **U** = _Ip_ , such that the covariance matrix of _Y_ is diagonal and that the variances of the components decreases, i.e. var( _Y_ 1) _≥· · · ≥_ var( _Yp_ ). 

Since the covariance matrix var( _X_ ) = E[( _X−µ_ )( _X−µ_ )<sup>_>_</sup> ], with _µ_ = E[ _X_ ], is symmetric positive semidefinite, it can be diagonalized and ordered by an orthogonal matrix **U** , i.e. 

is diagonal with decreasing, positive entries. If we consider the _empirical_ covariance matrix instead, then **U** is given by the SVD of the centered observation matrix as the left-singular vectors, cf. (4.3). 

## 4.3 Error Model Interpretation 

Consider the observed data **X** as the superposition of some _clean_ data **L** that lies in some _k_ -dimensional subspace and some _additional noise_ **N** , i.e. 

> 3The definition of the loadings matrix is not consistent in the literature. Sometimes, the matrix **L** := **UD** is called the loading matrix, and **U** is denoted as the _matrix of principal axes/directions_ . 

28 

Formally, requiring that the data **L** lies in some _k_ -dimensional subspace is equivalent to requiring that the rank of **L** is lower or equal to _k_ . We shall see below that another way of looking at PCA is to recover **L** given that the noise **N** is (entry-wise) i.i.d. Gaussian, i.e. each entry of the matrix **N** is drawn independently according to the Gaussian distribution with mean 0 and variance 1. Under this model assumption, the maximum likelihood estimation of **L** given the observations **X** is 

Classical PCA provides the solution of this problem. 

**Theorem 4.2.** _Let_ **X** = **UDV**<sup>_>_</sup> _be the singular value decomposition of the observation matrix_ **X** _with singular values d_ 1 _≥· · · ≥ dp ≥_ 0 _. If we denote by_ **U** _k the first k columns of_ **U** _and by_ **V** _k the first k columns of_ **V** _, then_ **L**<sup>ˆ</sup> = **U** _k_ diag( _d_ 1 _, . . . , dk_ ) **V** _k_<sup>_>minimizes_</sup> (4.11) _._ 

_Proof._ Let **L** be a minimizer of (4.11). Observe that the rank of a matrix **L** is lower or equal to _k_ if and only if its columns **l** _i_ lie in a _k_ -dimensional subspace, say _L_ . Now each column **l** _i_ has to be the projection of **x** _i_ onto _L_ , since otherwise, one could replace the _i_ -th column of **L** without increasing _k_ **X** _−_ **L** _k_<sup>2</sup> _F_<sup>= P</sup> _i_<sup>_k_</sup><sup>**x**</sup><sup>_i −_</sup><sup>**l**</sup><sup>_ik_2.Thus,</sup> 

and we know from Theorem 4.1, that _L_ is spanned by the first _k_ singular vectors **U** _k_ of **X** . Thus the best rank- _k_ approximation is given by 

## 4.4 Relation to Autoencoders 

PCA is closely related to a particularly simple form of neural networks, the so-called _autoencoders_ . The aim of an autoencoder is to reconstruct the inputs as good as possible after they have been passed through a lower dimensional space, or, more formally: Realizations _x_ 1 _, . . . , xn_ of a _p_ -dimensional random variable are first mapped to a lower dimensional space R<sup>_k_</sup> using a function _f_ , and these images are then mapped to R<sup>_p_</sup> again, trying to best fit the original input 

Autoencoders try to find the best pair of functions ( _f, g_ ) for this job, and the idea is that a reasonable part of the interesting information is contained in the reduced data _f_ ( **x** _i_ ). 

29 

Let us now assume that _f_ and _g_ are linear and represented by matrices **V** _2_ R<sup>_k⇥p_</sup> and **W** _2_ R<sup>_p⇥k_</sup> . Then _g ◦ f_ ( **x** _i_ ) = **WVx** _i_ , and if we measure the reconstruction error via the sum of squared distances, i.e. 

one can show that the optimal **V** is just given by the first _k_ left singular vectors of the observation matrix **X** = [ **x** 1 _, . . . ,_ **x** _n_ ]. 

**Theorem 4.3.** _Let_ **U** _k be the first k left singular vectors of the observation matrix_ **X** _. Then_ **V** = **U**<sup>_>_</sup> _k_<sup>_and_</sup><sup>**W**=</sup><sup>**U**</sup><sup>_kminimizethereconstructionerrorofthelinearauotencoder_</sup> (4.15) _._ 

_Proof._ Due to the dimensions of **V** and **W** , the squared matrix **WV** can at most have a rank of _k_ . So all **WVx** _i_ lie in a _k_ -dimensional subspace and thus form a matrix of at most rank _k_ . We have seen in Theorem 4.2 that the minimum of _L_ is achieved by **V** = **U**<sup>_>_</sup> _k_<sup>and</sup><sup>**W**=</sup><sup>**U**</sup><sup>_k_.</sup> 

There are interesting extensions of this approach by allowing nonlinear functions. Very successful in practical applications are for example functions of the form _f_ ( **x** ) := _σ_ ( **Vx** ), where **V** is a matrix as in the above case and _σ_ is an activation function that operates component-wise and is zero for negative arguments while leaving positive arguments unchanged. In such nonlinear settings, there are no closed form solutions to the resulting optimization problem, and we need techniques from optimization, like e.g. gradient descent methods, to approximate an optimal solution. 

Figure 4.2: A simple autoencoder with **V** _2_ R<sup>_k⇥p_</sup> , **W** _2_ R<sup>_p⇥k_</sup> , _k < p_ . 

30 

# 5 (Deep) Feedforward Neural Networks 

## 5.1 and Motivation of FNNs 

Roughly speaking, Feedforward Neural Networks (FNN) are particular function classes that are very powerful in minimizing any kind of expected loss, at the price of training a huge amount of parameters. To be more precise, consider an input variable _X 2_ R<sup>_p_</sup> and a function class _F_ out of which we want to find a function _f_ that minimizes the expectation of a certain loss function _L_ . Consider for example the simple Loss function _L_ ( _f_ ( _X_ )) = _kf_ ( _X_ ) _−Xk_<sup>2</sup> 2<sup>forautoencodersthataimsatreconstructingsamplesof</sup><sup>_X_.</sup> Here, _f_ consists of the concatenation of an encoder function (that maps into a low dimensional space) and a decoder function (mapping back into the raw data space). 

Another example from supervised Learning is regression, where we have a joint distribution ( _X , Y_ ) of input and output variables and the aim is to find the best _f 2 F_ that minimizes the expectation of _L_ ( _f_ ( _X_ ) _, Y_ ) = _kf_ ( _X_ ) _−Yk_<sup>2</sup> 2<sup>.Wediscussmulticlass</sup> classification subsequent to this section. 

If all functions in _F_ can be described by a set of parameters, say ⇥ _2_ R<sup>_N_</sup> , and if some samples, say _n_ , for training are given, then these learning problems result in a minimization process 

A very important class of functions in many applications are the so-called _feed-forward neural networks (FNN)_ . FNNs are concatenations of linear functions<sup>1</sup> 

followed by so called _activation functions_ that operate component-wise on a vector. Examples for activation functions are the _Rectified Linear Unit ReLU_ 

and others, cf. here Wikipedia. By a slight abuse of notation, we also denote the vector activation function by 

A feedforward neural network is a function 

> 1This setting also includes _affine_ functions, since we may simply append the additional component 1 to our input vector. 

31 

~~oSHESHE ——~~ <u>pit Lower Le</u> ~~<u>aM = WfOA+ WfOA+OA++ .</u> 3 Input~~ ~~<u>| wo 8) B)</u> ay~~ ~~<u>Co.. | | Ve t 4 zZ 3 We Wa we a2 ? a —6 | a) i Woo |} ar</u>~~<sup>~~<u>:|</u>~~</sup> ~~<u>fale</u>~~<sup>~~<u>tet :</u>~~</sup> ~~<u>al RATA</u>~~ 

where as usual _◦_ denotes the concatenation of functions and di↵erent function classes _F_ are defined by di↵erent activation function, the number of the _layers l_ and the dimensions of the matrices **W** _i 2_ R<sup>_ni⇥mi_</sup> . One usually calls such a FNN _deep_ if the number of layers is greater than three. Note, that the output dimension _o_ is determined by the loss function, since the output of the FNN serves as input to the loss. 

## 5.2 Training FNNs 

Training FNNs in practice is an art in itself, and there are many tricks and regularization techniques that decide on failure or success of learning a powerful FNN. In this section, we focus on the fundamental method of finding the optimal FNN for the general problem (5.1), i.e. the optimal weights ⇥= (<sup>ˆ</sup> **W**<sup>ˆ</sup> 1 _, . . ._ **W**<sup>ˆ</sup> _L_ ) within a given class of FNNs. It is in fact, a gradient descent method that iteratively updates the weights and the method is known as _backpropagation_ in the literature. In the following, we describe how it works in principal. 

The most important tool that we need here from our undergrad math courses is _the Chain rule and Jacobi matrices._ Recall, that if _g_ : R<sup>_k_</sup> _!_ R<sup>_l_</sup> and _h_ : R<sup>_l_</sup> _!_ R<sup>_m_</sup> are two functions with _g_ being di↵erentiable at **x** and _h_ di↵erentiable at **y** = _g_ ( **x** ) with Jacobi matrices **J** _g_ ( **x** ) and **J** _h_ ( **y** ), then the function _h ◦ g_ : R<sup>_k_</sup> _!_ R<sup>_m_</sup> is di↵erentiable at **x** with Jacobi matrix **J** _h◦g_ ( **x** ) = **J** _h_ ( _g_ ( **x** )) _·_ **J** _g_ ( **x** ). 

#### **Examples.** 

- ( _Linearity of derivatives_ .) If in the above setting, _h_ is a simple linear transformation given by matrix multiplication with, say **W** , then 

- Since the _i_ -th output only depends on _xi_ , the Jacobi-matrix of _σ_ in (5.4) is a squared diagonal matrix of the form 

- In order to define the Jacobi matrix of the function _multiplication with a vector from the right_ 

we first need to embed the _mn_ variables, here given in matrix structure, into R<sup>_mn_</sup> . This can be done in various ways, but if we choose to embed one row after the 

32 

other, i.e. 

then the Jacobi matrix has the well-arranged form 

Note, that due to the linearity of mult( **x** ), the Jacobi matrix does not depend on **W** . 

In order to compute the gradient of the cost function (5.1) with respect to the weights ⇥= ( **W** _l, . . ._ **W** 1), we note that it is the average of the gradients of the loss function for one input data **x** , i.e. 

It is therefore sufficient to compute the gradient of _F_ , which depends on one input signal, and then average over all training data **x** _i_ . For convenience, we denote 

the output after the _j_ -th layer of the FNN. For 0 _< j < l_ , **h** _j_ is called the _j-th hidden layer of the FNN_ , **h** _l_ is called _the output layer_ and **h** 0 := **x** is the input to the FNN. _<u>@</u>_ We denote by _@_ **W** _j_<sup>_F2_R1</sup><sup>_⇥mjnj_theJacobi-Matrix2ofthefunction</sup> 

Using the chain rule and the examples from the last section, the derivatives with respect to the di↵erent weight matrices **W** _i_ are given by 

33 

In practice, the initial weights are often chosen at random from a normal distribution, and then individually updated with the above gradients and a step size _↵>_ 0. Algorithms and methods for step-size selection are beyond the scope of this lecture. Note, that we have to ”reshape” the gradients, i.e. the transpose of _@_ **W** _<u>@</u> j_<sup>_F_, in matrix form by inverting</sup> the matrix embeddings _⇡j_ from (5.9). We finally have the update rule 

## 5.3 Multiclass with FNNs 

Multiclass classification considers the problem of assigning an input _X 2_ R<sup>_p_</sup> to one out of multiple, say _C_ , classes. We model the problem via the random variable ( _X , Y_ ), where _X 2_ R<sup>_p_</sup> and _Y 2 {_ **e** 1 _, . . . ,_ **e** _C}_ is one of the _C_ standard basis vectors in R<sup>_C_</sup> . A realization **y** = **e** _c_ means that the event _belongs to class c_ is true. This modelling of the output variable is also known as _one-hot-encoding._ 

The idea behind multiclass classification with FNNs is that _X_ serves as input to the network and the output is a vector in R<sup>_C_</sup> that should reflect the probability of the class distributions given _X_ . More precisely, if **x** is a realization of _X_ , and if _f_ denotes the FNN, then the _c_ -th component of the output vector **h** _l_ := _f_ ( **x** ) should approximately be the probability that _Y_ belongs to class _c_ , given **x** , i.e. 

The motivation of the last activation function _σl_ is thus to output a probability distribution for the _C_ classes, meaning in practice that the entries of the output vector are between 0 and 1, and that they add up to 1. A prominent choice here is the so called _softmax_ , given by 

_Exercise:_ Compute the Jacobi-Matrix of the softmax function. 

The loss function that we need for training the network has to measure how well the predicted distribution corresponds to the distribution observed through our training set ( **x** _i,_ **y** _i_ ), _i_ = 1 _, . . . , n_ . Note, that these observed distributions are deterministic, meaning that _yc_ := _Pr_ ( _Y_ = **e** _c|X_ = **x** _i_ ) = 1 if **x** _i_ is labelled with class _c_ and 0 else. Generally, one common way of comparing two probability distributions P = ( _p_ 1 _, . . . pC_ ) 

34 

and Q = ( _q_ 1 _, . . . qC_ ) over the same underlying set of _C_ events is by using the so called _cross-entropy_ , cf. Wikipedia, 

In our case with one distribution being deterministic, this reduces to the loss function 

where _f_ is a FNN with softmax as output and _f_ ( _X_ ) _c_ denotes its _c_ ’s entry. For training, as usual, we use the empirical expectation of the loss on our training data, leading to the optimization problem 

35 

# 6 Kernels and the Kernel Trick 

The success of the machine learning algorithms that we have discussed so far all rely on assumption on the distribution of the input data. PCA for example works better, the more the data is distributed around a linear subspace. Or in Linear Discriminant Analysis, where we assume Gaussian distributions of the classes that even have the same Covariance matrix. 

One way of extending methods to better take into account other, more complicated, distribution of the input data is to employ the so-called _Kernel Trick_ . It allows to generalize all methods that essentially only have standard inner products as input data. 

More precisely, consider a ML-algorithm with input data that can either be unlabelled, i.e. **x** 1 _, ...,_ **x** _n_ or labelled, i.e. ( **x** 1 _,_ **y** 1) _, ...,_ ( **x** _n,_ **y** _n_ ). Assume moreover that the algorithm actually only uses inner products _h_ **x** _i,_ **x** _ji_ := **x**<sup>_>_</sup> _i_<sup>**x**</sup><sup>_j_of the input data.Then,by replacing</sup> _h_ **x** _i,_ **x** _ji_ with some function __ ( **x** _i,_ **x** _j_ ) that is _a suitable generalization of an inner product_ (namely a Kernel, the definition will follow right away!) is called the Kernel Trick, cf. Fig. 6.1. The resulting learning method is typically named by prefixing the term _Kernel_ . This trick usually allows to extend methods that are based on linearity assumptions on the distribution of the data to more complex, non-linear distribution. 

Figure 6.1: Illustration of the Kernel Trick: Replace the standard inner product in a Machine Learning algorithm by a kernel to obtain the ’Kernel’-version of the method. 

So here is the definition of a Kernel. It generalizes the standard inner product. **Definition 6.1.** Sollten wir hier nicht f¨ur allgemeine Mengen Kernel definieren, anstatt 

37 

nur R<sup>_p_</sup> ? 

A (positive semidefinite) Kernel is a function __ : R<sup>_p_</sup> _⇥_ R<sup>_p_</sup> _!_ R such that for all finite sets **X** = _{_ **x** 1 _, . . . ,_ **x** _n}_ the _n ⇥ n_ -matrix 

is symmetric and positive semidefinite. The matrix **K** is called _Gram-Matrix_ of __ and **X** . For a given function __ it is usually difficult to tell whether it is a kernel function or not. Necessary conditions, however, like symmetry and positivity are easily checked. 

**Example 6.2.** The function __ ( **x** _,_ **y** ) = _k_ **x** _kk_ **y** _k−_ 1 cannot be a Kernel, since there exists a finite set, namely _{_ **0** _} ⇢_ R<sup>_p_</sup> , such that the associated Gram-Matrix (which is 1 _⇥_ 1 in this case) **K** = _−_ 1 is negative definite. 

Or consider the function __ ( **x** _,_ **y** ) = e<sup>_−k_</sup><sup>**x**</sup><sup>_−_</sup><sup>**y**</sup><sup>_k−k_</sup><sup>**y**</sup><sup>_k_</sup> . It is easily seen that in general, __ ( **x** _,_ **y** ) _6_ = __ ( **y** _,_ **x** ), and thus it cannot be a Kernel function. 

A few common kernels are 

- the linear Kernel __ ( **x** _,_ **y** ) = **x**<sup>_>_</sup> **y** + _c_ , _c ≥_ 0; 

- the polynomial Kernel __ ( **x** _,_ **y** ) = ( _↵_ **x**<sup>_>_</sup> **y** + _c_ )<sup>_d_</sup> , _c, ↵, d ≥_ 0; 

It follows straightforwardly from the properties of positive semi-definite matrices that, if __ 1 and __ 2 are Kernels, and if _c >_ 0, then so are 

- _c_ 1 

- _c_ + __ 1 

Moreover, for any real valued function _f_ : R<sup>_p_</sup> _!_ R, we can construct a kernel via __ := _f_ ( **x** ) _· f_ ( **y** ). Note, that in this case, the corresponding Gram-Matrix has a rank of at most one. 

38 

*bew) a) (ye) =< dian, poppy = \ mM + Ap kelp + Az (xerke Hi P 

# 7 Kernel Principal Component Analysis 

Standard PCA reduces the dimensionality of the observed data by projecting it onto a linear subspace. The projection is chosen in a way that the error measured in the squared standard Euclidean norm is minimized, which can also be interpreted as a way to reduce white Gaussian noise. One very important application is to use PCA as a preprocess for classification, since classifiers perform better in the feature space where this noise is reduced. 

The major drawback of standard PCA is that it heavily relies on the approximate linear structure of the data. In many applications, this is a far too strict assumption. Kernel PCA (K-PCA) is an extension of standard PCA that does not have these shortcomings. The crucial idea behind K-PCA is that it implicitly assumes the existence of a nonlinear map 

where _H_ is a very high dimensional vector space (that could even be infinite dimensional) with an inner product<sup>1</sup> _h·, ·i_ . There is no harm for us to think of _H_ as some R<sup>_P_</sup> with very large _P_ . As a preliminary step, we reformulate the well known standard PCA in a way such that it only involves inner products of our data. 

## 7.1 Linear PCA expressed with inner products 

Let **X** _2_ R<sup>_p⇥n_</sup> be the centered data matrix (this will be a crucial assumption for the following derivations) and let **K** := **X**<sup>_>_</sup> **X** be the ( _n ⇥ n_ )-matrix consisting of all inner products of the data. More precisely, the ( _i, j_ ) entry of **K** is the inner product **x**<sup>_>_</sup> _i_<sup>**x**</sup><sup>_j_of</sup> the _i_ -th and the _j_ -th observation. 

Recall, that if **X** = **U⌃V**<sup>_>_</sup> is the SVD of the data matrix, then the first _k_ principle components of a data vector **y** _2_ R<sup>_p_</sup> are given by **U**<sup>_>_</sup> _k_<sup>**y**.Inthecaseathandonlythe</sup> inner products are given and it is thus not possible to obtain **U** _k_ directly. However, the eigenvalue decomposition of **K** allows for an elegant solution to this problem. Remember that we have **K** := **X**<sup>_>_</sup> **X** and substitute **X** by its SVD. This yields 

Remember that **V** is orthogonal and **⌃**<sup>_>_</sup> **⌃** is diagonal. Thus, Equation (7.2) is the eigenvalue decomposition of **K** , due to the uniqueness of the EVD, and by computing the _k_ largest eigenvalues of **K** with their respective eigenvectors, we obtain _σ_ 1<sup>2</sup><sup>_, · · ·, σ_</sup> _k_<sup>2</sup> 

> 1 Formally, a Hilbert space _H_ is a real or complex inner product space that is also a complete metric space with respect to the distance function induced by the inner product. 

41 

and **V** _k_ . We will assume that _σk >_ 0, since otherwise we could reduce the dimension of the targeted subspace without losing any information on our data. For simplicity, we define the diagonal matrix **⌃** _k_ = diag( _σ_ 1 _, . . . , σk_ ). 

This allows us to to compute the principal components **U**<sup>_>_</sup> _k_<sup>**y**foragivenobservation</sup> **y** by only using inner products. For a new measurement **y** we have 

By multiplying this equation with **⌃**<sup>_−_</sup> _k_<sup>1,weobtain</sup> 

Note, that the right hand side of this equation can be computed by data that only involves inner products of the data, namely the Gram matrix **K** and the inner products **x**<sup>_>_</sup> _n_<sup>**y**.</sup> 

**Centering the data.** So far, we have assumed that the data is centered, i.e. that the Gram matrix **K** arises from _centered_ data. This assumption was crucial, since otherwise the derivation for Equation (7.4) would not hold. Now let us assume that we have no centered data available, and that the Gram matrix **K** is built from non-centered data. The good news is that it is possible to deduce the Gram matrix that corresponds to the centered data, say **K**<sup>˜</sup> , directly from **K** via the formula 

_Proof._ Recall from Equation (1.3), that if **X** denotes the non-centered data matrix, the centered matrix is given by 

With the shorthand notation for the symmetric matrix **H** := **I** _n − n_<sup><u>1</u>1</sup><sup>_n_1</sup> _n_<sup>_>_weobtain</sup> 

42 

So, if we have the Gram matrix from non-centered data, we can easily compute the Gram matrix that corresponds to centered data (without explicitly centering our **X** ). From this, we can - as above - compute the **V** _k_ and the **⌃** _k_ . In order to compute the principal components for a new data sample **y** , we first have to center **y** w.r.t. the training samples. Concretely, this means that we have to subtract the empirical mean ˆ of the training data _µ_ = _n_<sup><u>1</u></sup> P _i_<sup>**x**</sup><sup>_i_=</sup> _n_<sup><u>1</u></sup><sup>**X**1</sup><sup>_n_from</sup><sup>**y**.Thenwehavetoreplace</sup><sup>**X**inEqs.</sup> (7.3) and (7.4) with **X** = **XH** . This yields 

with 

## 7.2 Transition to Kernel PCA 

It is now straightforward to extend classical PCA by simply replacing the inner product **x**<sup>_>_</sup> **y** by _hφ_ ( **x** ) _, φ_ ( **y** ) _i_ . The practical success of K-PCA is due to the fact that for computing _hφ_ ( **x** ) _, φ_ ( **y** ) _i_ , neither _φ_ nor the inner product _h·, ·i_ is explicitly needed. Instead, it is to have a function 

that reflects the properties of _hφ_ ( **x** ) _, φ_ ( **y** ) _i_ , i.e. that is symmetric and fulfills the positivity property __ ( **x** _,_ **x** ) _≥_ 0 for all **x** _2_ R<sup>_p_</sup> . Substituting _hφ_ ( **x** ) _, φ_ ( **y** ) _i_ by __ ( **x** _,_ **y** ), and therefore not needing to know the feature mapping _φ_ , is called the _Kernel trick_ . 

**Definition 7.1** (Positive Definite Kernel) **.** 

1. Let _S_ := _{_ **x** 1 _, . . . ,_ **x** _n} ⇢_ R<sup>_p_</sup> and __ ( _·_ ) as above. The ( _n ⇥ n_ )-matrix **K** with ( _i, j_ )- entries _kij_ = __ ( **x** _i,_ **x** _j_ ) is called a _Gram_ - or _Kernel-matrix_ of _S_ with respect to __ . 

2. The function __ ( _·_ ) is called a _Kernel (positive definite Kernel)_ , if for all finite, nonempty sets _S ⇢_ R<sup>_p_</sup> the corresponding Gram matrix is positive semidefinite (positive definite). 

In the previous subsection we discussed the issue of centering the data in the latent Hilbert space. We face the same problem here. When a new data point comes in, it 

43 

is straightforward to compute the _uncentered_ vector of inner products of the new data point w.r.t. the training data. We will refer to it as 

The _j_ -th entry of the centered data then is 

In order to find a more concise expression for this, we use the linearity of the scalar product to get 

and it can easily be see that 

In summary, K-PCA for a training set **X** = [ **x** 1 _, . . ._ **x** _n_ ] _,_ **x** _i 2_ R<sup>_p_</sup> consists of the following steps. 

1. Find a suitable Kernel function __ ( _·_ ) and compute the _Gram matrix_ 

2. Compute the Gram matrix **K**<sup>˜</sup> = **HKH** with **H** = **I** _n − n_<sup><u>1</u>1</sup><sup>_n_1</sup> _n_<sup>_>_correspondingto</sup> centered data. 

3. Compute the Eigenvalue Decomposition **K**<sup>˜</sup> = **V⇤V**<sup>_>_</sup> . Due to the definition of kernel functions, the matrix **K** is positive semi-definite, and therefore the diagonal entries of **⇤** are non-negative. Hence, we can write **⇤** = **⌃**<sup>2</sup> = diag( _σ_ 1<sup>2</sup><sup>_, . . . , σ_</sup> _n_<sup>2).</sup> 

44 

4. Define the reduced matrices **⌃** _k_ = diag( _σ_ 1 _, . . . , σk_ ) _2_ R<sup>_k⇥k_</sup> and **V** _k 2_ R<sup>_n⇥k_</sup> . 

5. The reduced training data then is given by 

6. For a new data point **y** _2_ R<sup>_p_</sup> , the _k_ first kernel principal components are then computed by 

# 8 Support Vector Machines 

The idea behind support vector machines is pretty easy. In the simplest case it is assumed that samples of two classes can be _linearly separated_ , i.e. one assumes that there exists an affine hyperplane of codimension one which can separate the two classes. SVMs are supervised learning algorithms which, given training samples of the two classes, find the ”best” separating hyperplane. Once it is found it is then easy to classify a new data point: Depending on which side of the hyperplane the incoming new data point lies, it is assigned to the respective class. 

## 8.1 Some Geometry 

For some **w** _2_ R<sup>_p_</sup> _\ {_ 0 _}_ and some non-negative _b ≥_ 0, an _affine hyperplane_ in R<sup>_p_</sup> is as the set 

The vector **w** is _normal_ , or perpendicular to _H_ **w** _,b_ , because whenever we have an arbitrary line segment **x2** _−_ **x1** in the hyperplane with **x1** _,_ **x2** _2 H_ **w** _,b_ , then 

˜ From this it also follows easily that two affine hyperplanes _H_ **w** _,b_ , _H_ ˜ **w** _,b_ are parallel if and only if **w** is a multiple of **w** ˜ . 

Any hyperplane separates R<sup>_p_</sup> in exactly two half-spaces<sup>1</sup> . The _signed distance_ from some point **x** to _H_ **w** _,b_ is defined as 

The justification of this definition is the following Lemma. 

**Lemma 8.1** (Signed distance to affine hyperplanes) **.** _The Euclidean distance from_ **x** _to H_ **w** _,b is given by |δ_ ( **x** _, H_ **w** _,b_ ) _|._ 

_Proof._ We use some geometric intuition for the proof. Assume that starting from **x** , we want to move towards the hyperplane in direction **r** . In order to find the shortest distance, we have to solve 

> 1This follows from the so-called _hyperplane separation theorem_ which essentially states that two disjoint convex sets can be separated by a hyperplane. The proof goes back to famous Hermann Minkowski (1864-1909). 

The constraint here is equivalent to ( **x** + **r** )<sup>_>_</sup> **w** = _b_ , or 

From this equation it follows that **r** that solves (8.4) has to be a multiple of **w** . We proof this in the following. Assume that **r** is not a multiple of **w** . Then we can always write it as **r** = _λ_ **w** +<sup>P</sup> _i_<sup>_µi_</sup><sup>**w**</sup> _i_<sup>_?_forsome</sup><sup>**w**</sup> _i_<sup>_?_areorthogonalto</sup><sup>**w**.Duetotheorthogonality,</sup> the equality ( _λ_ **w** +<sup>P</sup> _i_<sup>_µi_</sup><sup>**w**</sup> _i_<sup>_?_)</sup><sup>_>_</sup><sup>**w**=</sup><sup>_λ_</sup><sup>**w**</sup><sup>_>_</sup><sup>**w**holds,whileatthesametime</sup><sup>_k_</sup><sup>**r**</sup><sup>_k≥kλ_</sup><sup>**w**</sup><sup>_k_</sup> according to the triangle inequality. 

This observation yields the minimization problem 

which has as solution the absolute value of the signed distance 

It is readily seen that the signed distance to the hyperplane is positive in one half space, and negative in the other. We define the _margin_ of _H_ **w** _,b_ as the set of points that are close to _H_ **w** _,b_ . More precisely, let 

be two affine hyperplanes parallel to _H_ **w** _,b_ . Then the margin of _H_ **w** _,b_ is defined as the convex hull of _H_ + _[ H−_ . 

Using Lemma 8.1, we see that the thickness of this margin, i.e. the distance between <u>2</u> _H_ + and _H−_ , is given by _k_ **w** _k_<sup>.So if we want to find an affine hyperplane</sup><sup>_H_</sup><sup>**w**</sup><sup>_,b_that allows</sup> for “best” separating two classes of data points, and if we quantify “best” by allowing the largest margin while still preventing data points to fall within this margin, we will <u>2</u> have to maximize _k_ **w** _k_<sup>undersomeconstraints,whichleadsustolinearSVM.</sup> 

## 8.2 Basic Linear SVM 

As mentioned above, SVM is a supervised learning method, so we start with _N_ labeled training data ( **x** _i, yi_ ) _2_ R<sup>_p_</sup> _⇥{−_ 1 _,_ 1 _}, i_ = 1 _, . . . , N_ . Here, _yi_ is either 1 or _−_ 1 and indicates to which of the two classes the data belongs. For linear SVM, we have to assume that this data is linearly separable indeed, i.e. we have to assume that an affine hyperplane _H_ **w** _,b_ exists that separates the two classes. The constraint that no data point lies within 

48 

the margin of _H_ **w** _,b_ is equivalent to requiring that all points belonging to class _yi_ = 1 have a positive distance to _H_ +, and all data belonging to class _yi_ = _−_ 1 have a negative distance to _H−_ , i.e. 

This can be written more compactly as 

Therefore, the task of finding the affine hyperplane that allows for the largest margin, while still separating the two classes, is described by the optimization problem 

or, equivalently, to 

In order to tackle this constrained optimization problem we need to introduce optimality conditions for these types of problems. 

### 8.2.1 Karush-Kuhn-Tucker Conditions 

We refer to the text book _Numerical Optimization, 2nd Edition_ , Springer 2006, by J. Nocedal and S.J. Wright, for a more detailed insight into the topic of optimization. Consider the general problem 

with the smooth real valued functions _f, ci_ . Here _E_ stands for _equality constraints_ and _I_ stands for _inequality constraints_ . For some point **z** that satisfies the constraints, we define its _active set_ as _A_ ( **z** ) = _E [ {j_ : _cj_ ( **z** ) = 0 _}_ . In other words, _A_ ( **z** ) are all indices of the constraints where **z** _exactly_ satisfies the equality conditions. 

The Lagrange function of the optimization problem (8.15) is given by 

**Theorem 8.2** (Karush-Kuhn-Tucker (KKT) conditions) **.** _Let_ **z**<sup>_?_</sup> _be a solution of_ (8.15) _. Under certain conditions on the constraint functions_<sup>_2_</sup> _there exists a Lagrange multiplier_ **_λ_**<sup>_?_</sup> _such that_ 

Since the optimization problem for SVMs is convex (a convex objective function with constraints that define a convex feasible region) the KKT conditions are necessary and sufficient for **z**<sup>_?_</sup> _,_ **_λ_**<sup>_?_</sup> to be a solution. The last conditions imply that either constraint _i_ is in the active set, or that the _i_ -th component of the Lagrange multiplier is zero, or possibly both. 

### 8.2.2 Lagrangian Duality 

Like before, we will consider the general optimization problem 

This is also called the _primal_ problem. The corresponding Lagrange function is defined as 

with Lagrangian multipliers (also called _dual variables_ ) _λi ≥_ 0. Based on the Lagrange function we can create a new function that provides a lower bound on the objective function _f_ . Since the dual variables are all positive, i.e. _λi ≥_ 0, we know that _f_ ( **z** ) _≥ L_ ( **z** _,_ **_λ_** ) for all feasible **z** . This motivates the definition of the _Lagrange dual function_ as 

The dual function _g_ ( _·_ ) is concave, even if the original problem is not convex since it is the point-wise infimum of affine functions. The dual form yields lower bounds on the optimal value _p_<sup>_?_</sup> of the objective function _f_ . For any **_λ_** _≥_ 0 we have _g_ ( **_λ_** ) _ p_<sup>_?_</sup> . 

The Lagrangian dual problem is the maximization problem 

> 2which are fulfilled in case they are linear, so in particular for the case of linear SVM 

50 

Under certain conditions (that hold in the case of SVMs), the minimum of the primal problem coincides with the maximum of the dual problem, i.e. _d_<sup>_?_</sup> = max _g_ ( **_λ_** ) = inf **z** _f_ ( **z** ) = _p_<sup>_?_</sup> . This is called _strong duality_ . 

_Remark._ Duality allows us to compute a _lower bound_ on the optimal value for any problem, convex or not, using convex optimization. However, the dual function _g_ may not be easy to compute since it is defined as an optimization problem itself. Using duality works best when _g_ can be written in a closed form. Even then it might not be easy to find a solution to the dual problem since not all convex problems are easy to solve. 

### 8.2.3 Linear SVM: Primal and Dual Problem 

Following this motivation, for the problem at hand we have to solve the problem 

This problem is strictly convex, and therefore its solution is unique if it exists. The gradient of the Lagrange function w.r.t. the optimization parameters ( **w** _, b_ ) is given by 

Therefore, the KKT conditions (8.17) and (8.21) yield 

This implies that the solution **w**<sup>_?_</sup> is a linear combination of points that touch the boundary hyperplanes, i.e. those points that lie in _H_ + and _H−_ . These points are the name giving _support points_ or _vectors_ . 

We can now use these equations to derive an easy way for finding the desired optimal hyperplane parameters **w**<sup>_?_</sup> and _b_<sup>_?_</sup> . Substituting first (8.27) and then and (8.28) into 

51 

(8.25) yields (omitting<sup>_?_</sup> for improved readability) the equation 

This new formulation which only depends on **_λ_** is the dual form of the problem (cf. (8.22)) and we write it as 

where the entries of **H** are defined as the inner products _hij_ = _yiyj_ **x**<sup>_>_</sup> _i_<sup>**x**</sup><sup>_j_.Theoptimal</sup> Lagrangian multipliers are found by solving the maximization problem 

This is a convex quadratic optimization problem which can be solved by using a quadratic program (QP) solver (e.g. the function `quadprog` in Matlab). The resulting **_λ_**<sup>_?_</sup> then yields **w**<sup>_?_</sup> by plugging it into Equation (8.27) and _b_<sup>_?_</sup> is obtained via Equation (8.29). Note that the Lagrange multiplier _λi_ corresponds to the point **x** _i_ . If _λi_ is unequal to zero, then Equation (8.29) implies that _yi_ ( **w**<sup>_>_</sup> **x** _i − b_ ) = 1, i.e. the corresponding **x** _i_ is an element of either _H_ + or _H−_ . 

## 8.3 Soft Margin Linear SVM 

Clearly, the above method fails if the training set is not linearly separable, because in this case, there is no point that fulfills the constraints. To overcome this obvious drawback, the _soft margin SVM_ allows for some wrongly assigned data samples. It searches for a trade-o↵between a large margin and a degree of misclassification. This misclassification is quantified by a set of _N_ additional variables _⇠i_ that are assumed to be nonnegative, leading to the constraints 

So, in order to gain a large margin while at the same time keeping the misclassifiaction moderate, we consider the optimization problem 

52 

The free parameter _c >_ 0 weighs between a large margin and misclassification. The larger _c_ is chosen, the more the violation of the separation rule is punished. As before, this is a quadratic programming problem that can be solved by using a QP solver. The corresponding KKT conditions are discussed in the following. The Lagrangian for our soft margin SVM reads as 

where _µi_ are the Lagrange multipliers introduced to enforce positivity of _⇠i_ . The corresponding KKT conditions are 

While a QP solver could already find the solution to this problem, we will now derive the corresponding dual form since it has an appearance very similar to the separable problem. Also, the dual form is required in order to extend SVMs to work with kernels. First, note that Equation (8.37) implies that _µi_ = _c − λi_ . By plugging this into Equation (8.34) we can already remove the dependency of _µi_ . It can also easily be seen that by this substitution _⇠i_ is eliminated and we obtain 

Next, we use the fact from (8.35) that **w** =<sup>P</sup> _i_<sup>_λiyi_</sup><sup>**x**</sup><sup>_i_andsubstitutethisinto(8.44).</sup> By employing the property (8.36), we obtain the dual form 

with the matrix **H** with entries _hij_ = _yiyj_ **x**<sup>_>_</sup> _i_<sup>**x**</sup> _j_<sup>undertheconstraints0</sup><sup>_λic_and</sup> P _i_<sup>_λiyi_= 0.Hence,thedualproblemhastheform</sup> 

The solution is then given by **w**<sup>_?_</sup> =<sup>P</sup> _i_<sup>_λiyi_</sup><sup>**x**</sup><sup>_i_.Thus, the only di↵erence to the separable</sup> case is that the _λi_ have an upper bound _c_ . Equation (8.37) and (8.43) imply that _⇠i_ = 0 if _λi < c_ . Thus, any training sample **x** _i_ for which 0 _< λi < c_ is a support vector. Furthermore, for any support vector **x** _i_ Equation (8.42) reduces to _yi_ ( **w**<sup>_>_</sup> **x** _i − b_ ) + 1 = 0 and can be used to compute _b_<sup>_?_</sup> . To obtain a more stable solution it is recommended to use the average over all points in the support. Specifically, it is defined as 

where _Supp_ denotes the support indices and _NSupp_ the number of support vectors. 

54 

## 8.4 Kernel SVM 

As we have seen before the linear soft margin SVM problem for a set of trainings samples ( **x** _i, yi_ ) _, i_ = 1 _, . . . , N_ , **x** _i 2_ R<sup>_p_</sup> _, yi 2 {−_ 1 _,_ 1 _}_ can be rewritten in the dual form 

with the Gram matrix **H** the entries of which are defined as _hij_ = _yiyj_ **x**<sup>_>_</sup> _i_<sup>**x**</sup><sup>_j_.Wehave</sup> talked about the Kernel trick, i.e. using a nonlinear function _φ_ that maps the training samples to a high dimensional Hilbert space _H_ with inner product _h·, ·iH_ and expressing inner product directly with a kernel function __ , in the chapter about kernel PCA. That is, the kernel function __ maps two points to R via __ ( **x** _,_ **y** ) = _hφ_ ( **x** ) _, φ_ ( **y** ) _iH_ . The same holds true in this case. The inner product **x**<sup>_>_</sup> **y** can be replaced by an appropriate kernel function. Common choices are: 

With tanh( _x_ ) = ( _e_<sup>_x_</sup> _− e_<sup>_−x_</sup> ) _/_ ( _e_<sup>_x_</sup> + _e_<sup>_−x_</sup> ). The entries of the Gram matrix are then defined using one of these kernel functions via 

Some notes regarding the previously mentioned kernels: 

- In the case of the polynomial kernel the dimension of the Hilbert space _H_ is � _p_ + _d d_ � when the original signals are in R<sup>_p_</sup> . 

- The radial basis function kernel is often also referred to as Gauss kernel and describes a nonlinear function _φ_ that maps to an infinite dimensional Hilbert space _H_ . 

- The sigmoid kernel only produces a s.p.d. Kernel matrix for certain choices of _γ, δ_ and under specific conditions on the squared norm of the signals _k_ **x** _k_<sup>2</sup> . 

Solving the problem 

55 

with _hij_ = _yiyj_ ( **x** _i,_ **x** _j_ ) provides the Lagrange coefficients **_λ_**<sup>_?_</sup> , but how can we classify a new point using this results? First, note that according to the classification rule from the linear case (i.e. a vector **z** is classified using the rule sign( **w**<sup>_>_</sup> **z** _− b_ )), we can express the decision function by 

where _Supp_ denotes the support (i.e., all _i_ for which 0 _< λi  c_ ). We assign the label according to the sign of _f_ ( **z** ). 

The only remaining ingredient we need is the factor _b_ . From the KKT conditions of the original problem, we know that the equation 

has to hold for any _i 2 Supp_ . The vector **w** is an element of the high dimensional Hilbert space _H_ and can be rewritten as a sum of the mapped support vectors<sup>P</sup> _j2Supp_<sup>_λiyiφ_(</sup><sup>**x**</sup><sup>_i_).</sup> Plugging this into the previous equation yields 

for any _i_ in the support. To make this result more robust, we average over all possible indices _i_ and get 

where _NSupp_ is the number of support vectors. Hence, by replacing the inner products with the kernel function, we can compute _b_ as 

56

---

## 源文件

- [InfoRet_Lecture_Notes.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-86147004342c/InfoRet_Lecture_Notes.pdf)
