---
title: mlcomm_exam (2)
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-544538350872/mlcomm_exam
  (2).pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-544538350872/mlcomm_exam
    (2).pdf
  title: mlcomm_exam (2).pdf
---

# mlcomm_exam (2)

Technische Universit¨at M¨unchen Lehrstuhl f¨ur Nachrichtentechnik Prof. Dr. sc. techn. Gerhard Kramer 

# **Machine Learning for Communications** 

#### **Prof. Dr. sc. tech. Gerhard Kramer SS 2019 October 2nd, 2019** 

- Write your name and student ID on every sheet. Please have your student card ready for examination. 

- The exam duration is 90 minutes. 

- This exam has 5 questions on 15 pages (excluding the cover page). 

- Do not write with pencils or red pens. 

- You can get full credit if and only if you give reasons for your answer. 

- Problem parts that are marked with an = _)_ can be solved independently of previous parts of the same problem. 

- Please remain seated until the end of the exam. 

- You must hand in this problem set! 

- You are allowed to use a non-programmable calculator. 

### **Good Luck!** 

Name: 

Student ID: Course of Studies: **Please read and sign below:** I hereby confirm that I have been informed prior to the beginning of the examination that I have to notify the examination supervisors immediately if sudden illness occurs during the examination. This will be noted in the examination protocol. An application for exam withdrawal has to be filed immediately to the board of examiners being in charge. A medical certificate from one of the physicians acknowledged by the Technische Universit¨at M¨unchen issued on the same day as the examination must be forwarded without delay. In case the examination is completed despite of illness, a subsequent withdrawal due to illness cannot be accepted. In case the examination is ended due to illness it will not be <u>graded.</u> Signature: 

### **For internal use only:** 

|Aufgabe|Punktzahl|Davon erreicht|
|---|---|---|
|Short Questions|14||
|Neural Networks|23||
|Probabilistic Graphical Models|29||
|Expectation Maximization|12||
|Principal Component Analysis|8||
|**Summe**|86||

**(14 points)** 

### **Problem 1: Short Questions** 

Check the right box for the following problems. For each problem, _only one answer is correct_ . 

= _)_ 

a) Let _<u>X</u>_ and _<u>Y</u>_ be real-valued random variables in R<sup>_K_</sup> with mean _<u>µ</u>_ _~~<u>X</u>~~_ = **0** and _<u>µ</u>_ _~~<u>Y</u>~~_ and (1 pt) covariance matrices _CX_ and _CY_ <u>,</u> respectively. The joint covariance matrix is _CXY_ . We consider the optimization problem 

The solution of (1) for a concrete realization _<u>y</u>_ is given by 

<u>X</u> _f_ ( _<u>y</u>_ <u>) =</u> **_C_** _<u>XY</u>_ **_C_** _<u>Y</u>_<sup>_−_1</sup> ( _<u>y</u> −_ _<u>µ</u>_ _~~<u>Y</u>~~_ ), if _<u>X</u>_ and _<u>Y</u>_ are jointly Gaussian. _f_ ( _<u>y</u>_ <u>) =</u> **_C_** _<u>XY</u>_ **_C_** _<u>X</u>_<sup>_−_1</sup> _<u>Y</u>_ ( _<u>y</u> −_ _<u>µ</u>_ _~~<u>X</u>~~_ ), if _<u>X</u>_ and _<u>Y</u>_ are uncorrelated. _f_ ( _<u>y</u>_ <u>) =</u> **_C_** _<u>XY</u>_ **_C_** _<u>Y</u>_<sup>_−_1</sup> ( _<u>y</u> −_ _<u>µ</u>_ _~~<u>Y</u>~~_ ), if _<u>X</u>_ and _<u>Y</u>_ are stochastically independent. _f_ ( _<u>y</u>_ <u>) =</u> **_C_** _<u>XY</u>_ **_C_** _<u>X</u>_<sup>_−_1</sup> ( _<u>y</u> −_ _<u>µ</u>_ _~~<u>X</u>~~_ ), if _<u>X</u>_ and _<u>Y</u>_ are stochastically independent. T<sup>⇤</sup> = _)_ b) We consider the eigenvalue decomposition **_U_ ⇤** **_U_**<sup>_−_1</sup> of a covariance matrix **_C_** _<u>X</u>_ = E ⇥ _<u>XX</u>_ (1 pt) of a zero mean random variable _<u>X</u> 2_ R<sup>_K_</sup> . The columns of **_U_** are _<u>u</u>_ _~~i~~_<sup>_, i_= 1</sup><sup>_, . . . , K_.</sup> The covariance matrix **_C_** _<u>X</u>_ is positive definite. The covariance matrix **_C_** _<u>X</u>_ is negative semi-definite. <u>X</u> The covariance matrix **_C_** _<u>X</u>_ has real-valued eigenvalues and we have _<u>u</u>_ T _i_<sup>_<u>u</u>_</sup> _~~j~~_<sup>=</sup><sup>_δij_.</sup> The covariance matrix **_C_** _<u>X</u>_ has positive eigenvalues and we have _<u>u</u>_ T _~~i~~_<sup>_<u>u</u>_</sup> _~~j~~_<sup>=</sup><sup>_δij_.</sup> = _)_ c) Consider **_A_** _2_ R<sup>_K⇥K_</sup> . The eigenvalues of **_A_** are _λ_ 1 _≥ λ_ 2 _≥ . . . ≥ λK_ . Its singular values (1 pt) are _σ_ 1 _≥ σ_ 2 _≥ . . . ≥ σK_ It holds that _σk_<sup>2=</sup><sup>_λk_.</sup> It holds that _σk_ = _λ_<sup>2</sup> _k_<sup>.</sup> It holds that _σk_ = _λk_ . <u>X</u> In general, no statement can be made. = _)_ d) Consider a general matrix **_A_** _2_ R<sup>_K⇥K_</sup> . Its eigenvalues are _λ_ 1 _≥ λ_ 2 _≥ . . . ≥ λK_ . (1 pt) <u>X</u> det( **_A_** ) =<sup>Q</sup><sup>_K_</sup> _i_ =1<sup>_λi_.</sup> tr( **_A_** ) =<sup>Q</sup><sup>_K_</sup> _i_ =1<sup>_λi_.</sup> det( **_AA_**<sup>T</sup> ) =<sup>P</sup><sup>_K_</sup> _i_ =1<sup>_λ_</sup> _i_<sup>2.</sup> tr( **_AA_**<sup>T</sup> ) =<sup>P</sup><sup>_K_</sup> _i_ =1<sup>_λ_</sup> _i_<sup>2.</sup> = _)_ e) Consider a projector **_P_** _2_ R<sup>_K⇥K_</sup> . (1 pt) The eigenvalues of the projector **_P_** are all ones. The eigenvalues of the projector **_P_** are all zeros. <u>X</u> The eigenvalues of the projector **_P_** are either one or zero. In general, no statement can be made. 

_Page 1 - Please turn page_ 

<u>X</u> Any vector in range( **_A_** ) is an eigenvector of the projector **_P_** . The projector **_P_** is given as **_AA_**<sup>T</sup> . 

**_P_** has full rank. 

The projector **_P_** can be calculated from the eigenvalue decomposition of **_A_** . 

= _)_ h) We want to calculate E [ _f_ ( _X_ )] where _X ⇠ PX_ and use importance sampling (IS) to (1 pt) calculate an approximation of this expected value. Another distribution on the same alphabet as _PX_ is _QX_ . We further have _PX_ ( _x_ ) = _PX_<sup>_⇤_(</sup><sup>_x_)</sup><sup>_/Z_Pand</sup><sup>_QX_(</sup><sup>_x_)=</sup><sup>_Q⇤_</sup> _X_<sup>(</sup><sup>_x_)</sup><sup>_/Z_Q,</sup> where _Z_ P and _Z_ Q are normalization constants. 

_Page 2 - Please turn page_ 

- = _)_ i) We use rejection sampling to estimate the value of _⇡_ . As a proposal distribution, a uniform distribution over the two-dimensional region [ _−_ 1 _,_ 1] _⇥_ [ _−_ 1 _,_ 1] is used and the number of samples which fall into the circle/quadrat is counted. We get the following outcome. 

(2 pt) 

According to the picture above, an approximation of _⇡_ is 

2.92 

<u>X</u> 2.86 

3.14 

- 3.22 

= _)_ j) Markov Chain Monte Carlo methods approximate the desired distribution, from which (1 pt) sampling should take place, as the 

invariant distribution of a Markov chain. 

stable distribution of a Markov chain. 

reducible distribution of a Markov chain. 

<u>X</u> ergodic distribution of a Markov chain. 

= _)_ k) We now consider stochastic matrices. 

(2 pt) 

0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 3 ✓ ◆ imately) (0 _._ 46 _,_ 0 _._ 54)<sup>T</sup> . 0 _._ 3 0 _._ 6 <u>X</u> The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 4 ✓ ◆ imately) (0 _._ 46 _,_ 0 _._ 54)<sup>T</sup> . 0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. An invariant distribution is (approx0 _._ 7 0 _._ 4 ✓ ◆ imately) (0 _._ 76 _,_ 0 _._ 65)<sup>T</sup> . 0 _._ 3 0 _._ 6 The matrix is a stochastic matrix. The eigenvalues are 0.46 and 0.54. 0 _._ 7 0 _._ 4 ✓ ◆ 

_Page 3 - Please turn page_ 

**(23 points)** 

### **Problem 2: Neural Networks** 

In this task we want to build a classifier that is able to recognize the modulation format that has been used at the transmitter side. In particular, we want to be able to distinguish between 16-QAM and 64-QAM. The constellation samples are transmitted over a complex valued additive white Gaussian noise channel. 

- a) What dimension should the input vector have so that standard machine learning toolkits (2 pt) (e.g., Keras, Tensorflow) can be used? Why? 

**Solution:** The complex valued channel observation should be represented by a twodimensional vector with real valued components.<sup>_p_</sup> Standard frameworks assume real valued data.<sup>_p_</sup> 

You are asked to implement a classifier that fulfills the following specifications: 

   - Shallow, fully connected feedforward neural network (NN) architecture. 

   - Two hidden layers with _M_ 1 and _M_ 2 neurons. 

   - Output should indicate to which constellation format the input belongs. 

- = _)_ b) Sketch your proposed NN architecture and specify all of its parameters including their (6 pt) dimensions. 

#### **Solution:** 

_•_ First hidden layer: **_W_** 1 _2_ R<sup>_M_1</sup><sup>_⇥_2</sup> _, b_ ~~1~~<sup>_2_R</sup><sup>_M_</sup> 1 _p_ 

   - Second hidden layer: **_W_** 2 _2_ R<sup>_M_2</sup><sup>_⇥M_1</sup> _, b_ 2<sup>_2_R</sup><sup>_M_2</sup><sup>_p_</sup> 

- Output layer parameters: _<u>w</u>_ ~~3~~<sup>_2_R</sup><sup>_M_2</sup><sup>_, b_3</sup><sup>_2_R</sup><sup>_p_</sup> 

- Sketch:<sup>_pp_</sup> 

_Page 4 - Please turn page_ 

- c) Give an expression for the output _y_ model of the NN depending on the input _<u>x</u>_ and the (3 pt) NN parameters. Use the same notation as above. 

#### **Solution:** 

- d) What activation function do you propose for layer one and two? Which one for the (4 pt) output layer? Justify your choice. Provide the formulas. 

#### **Solution:** 

- Hidden layer 1/2: Relu (max(0 _, x_ ))/tanh (tanh( _x_ )).<sup>_p_</sup> 

- Output layer: sigmoid (1 _/_ (1 + exp( _−x_ ))).<sup>_pp_</sup> 

We need the sigmoid in the last layer to allow a binary classification based on the cross entropy.<sup>_p_</sup> 

_Page 5 - Please turn page_ 

- e) What loss function should be used during the training phase? Write down the formula (6 pt) using the parameters introduced in task b) and c). Assume _N_ feature and label pairs ( _<u>x</u>_ _~~i~~_<sup>_, yi_),</sup><sup>_i_= 1</sup><sup>_, . . . , N_.</sup> 

**Solution:** We should use a cross entropy based loss function.<sup>_p_</sup> The formula is 

- = _)_ f) Assume that in the next product release the NN should be able to recognize more (2 pt) constellation formats. What has to be changed? 

#### **Solution:** 

- We may need to add more hidden layers to increase the NN’s generalization abilities. _p_ 

- The sigmoid activation function needs to be replaced by a softmax function.<sup>_p_</sup> 

- The encoding of the labels has to be adapted (one-hot encoding of the modulation classes). 

_Page 6 - Please turn page_ 

**(29 points)** 

### **Problem 3: Probabilistic Graphical Models** 

We consider a binary linear block code _C_ defined by its parity-check matrix 

The code _C_ is used for transmission over a binary symmetric channel (BSC) with input _X 2 {_ 0 _,_ 1 _}_ and output _Y 2 {_ 0 _,_ 1 _}_ . The channel law is 

with _p_ = 0 _._ 2. 

= _)_ 

- a) Give a list of the codewords in _C_ . _Hint:_ A codeword fulfills _<u>c</u>_ **_<u>H</u>_**<sup>T</sup> = **0** . There are four (2 pt) codewords. 

� _<u>u p</u>_ <u>�,</u> we get _<u>p</u>_ = _<u>u</u>_ **_<u>P</u>_**<sup>T</sup> . The codewords are therefore: _C_ = _{_ 0000 _,_ 0110 _,_ 1011 _,_ 1101 _}_<sup>_pp_</sup> _._ 

- b) The distribution on the first two bits are _PX_ 1(0) = 0 _._ 2 and _PX_ 2(0) = 0 _._ 9. Further, (3 pt) _X_ 1 and _X_ 2 are independent. How does the distribution _PX_ of a codeword _<u>X</u>_ = ( _X_ 1 _, X_ 2 _, X_ 3 _, X_ 4) factorize? What is the probability of each codeword? 

**Solution:** The distribution factorizes as 

Hence, the codewords have the following distribution: 

_Page 7 - Please turn page_ 

- = _)_ c) We want to use a bit-MAP (maximum a posteriori) decoder. What is the corresponding (1 pt) decision rule for the _i_ -th bit and the channel observation _<u>y</u>_ = ( _y_ 1 _, y_ 2 _, y_ 3 _, y_ 4)? 

**Solution:** The decision rule is 

- = _)_ d) What problem can the sum-product algorithm solve? 

(1 pt) 

**Solution:** The sum-product algorithm solves an inference problem, e.g., how to calculate a marginalization efficiently. 

- = _)_ e) Draw a factor graph representation of the model imposed by _PX|Y_ and explain your (7 pt) derivations. Associate each variable and factor node with parts of the respective model. _Hint:_ Any proportionality constants can be omitted. 

#### **Solution:** We have: 

_PX|Y_ ( _<u>x|y</u>_ <u>)</u> _/ PXY_ ( _<u>x, y</u>_ <u>) =</u> _PY |X_ ( _<u>y|x</u>_ <u>)</u> _PX_ <u>(</u> _<u>x</u>_ ) 4 = Y _PY |X_ ( _yi|xi_ ) _PX_ 1( _x_ 1) _PX_ 2( _x_ 2) 1 ( _x_ 1 + _x_ 2 + _x_ 3 = 0) 1 ( _x_ 1 + _x_ 4 = 0)<sup>_pp_</sup> _i_ =1 The factor graph is therefore: 

Association<sup>_pp_</sup> : 

_Page 8 - Please turn page_ 

- f) Calculate _PX_ 1 _|y_ ( _x_ 1 _|y_ <u>)</u> numerically using the sum-product algorithm assuming that _<u>y</u>_ = (7 pt) ( _y_ 1 _, y_ 2 _, y_ 3 _, y_ 4) = (0 _,_ 0 _,_ 1 _,_ 0) was received. 

#### **Solution:** 

We now calculate the factors: 

We combine the expressions above: 

= _)_ g) Is the above result exact? Why? 

**Solution:** The solution is exact, because the factor graph is loop free.<sup>_p_</sup> 

_Page 9 - Please turn page_ 

For a simulation experiment, we need to create realizations of a truncated Gaussian random variable. A truncated Gaussian random variable with parameters ( _µ, σ_<sup>2</sup> _, a_ ) has the following probability density function (PDF) 

= _)_ h) Determine the constant _A_ such that _pX_ ( _x_ ) is a valid PDF. 

- = _)_ i) Write a Python function that returns `N` realizations of truncated Gaussian distribution (4 pt) with parameters ( _µ, σ_<sup>2</sup> _, a_ ). 

_Hint:_ Use the function signature: 

`def truncated` ~~`g`~~ `auss` ~~`s`~~ `amples(N, mu, sigma2, a): pass` . 

The function `np.random.randn(N)` returns _N_ realizations from the standard normal distribution. 

#### **Solution:** 

```
importnumpyasnp
#[...]
deftruncated_gauss_samples(N,mu,sigma2,a):
=
x_samples[]
while1:
x=np.sqrt(sigma2)*np.randn()+mu
ifx>a:
x_samples.append(x)
ifx_samples.len>=N:
break
```

_Page 10 - Please turn page_ 

**(12 points)** 

### **Problem 4: Expectation Maximization** 

The information theoretic model of a communication channel depends on the transmission and reception method as well as the physical properties of the respective transmission medium. For optical channels, i.e., when rather the particle nature (photon) of light plays a role instead of its nature as an electromagnetic wave, a common model is the Poisson distribution. We model the channel _PY |X_ ( _y|x_ ) as follows: In a given time slot, there is either a pulse, represented by _X_ = +1, or there is no pulse, represented by _X_ = _−_ 1. Further, let _λ_ s be the average number of received photons in a pulsed time slot (when no noise is present) and let _λ_ n be the average number of received noise photons per slot. We then have: 

The probability of sending a pulse or not pulse is iid and given as 

At the receiver, _N_ samples _y_ 1 _, y_ 2 _, . . . yN 2_ N0 were collected, but the model parameters _λ_ n, _λ_ s of (2) and _p_ 0 of (3) are unknown. 

- = _)_ a) Write down the log-likelihood function for a maximum-likelihood estimation of the (2 pt) model parameters _λ_ n _, λ_ s and _p_ 0. We collect the parameters in the vector _<u>✓</u>_ = ( _✓_ n _, ✓_ s _, ✓_ p) = ( _λ_ n _, λ_ s _, p_ 0). 

#### **Solution:** 

_Page 11 - Please turn page_ 

- = _)_ b) Formulate the E-step of the EM algorithm in the _t_ -th iteration. Simplify the expression (4 pt) as far as possible. 

#### **Solution:** 

_Page 12 - Please turn page_ 

- = _)_ c) Formulate the M-step of the expectation maximization (EM) algorithm in the _t_ -th (6 pt) iteration. Simplify the expression as far as possible. 

#### **Solution:** 

_Page 13 - Please turn page_ 

## | Bere 

4.432 ~~. _~~ 24 -[ FI) ble)FT, (27 cov (x:)=4 (In C=] | | ohe mel 7 > yy #(4-H 4417) BAF 4a? oe ene: - LA + Da- st(4 uv y]=2 =Te | yOFie AECP= —14 pele + BR] ADF 

- = _)_ b) Find the largest eigenvalue and corresponding eigenvector (of unit length) of the matrix (4 pt) **C** _X_ in the following cases: 

   - (i) **C** _X_ = _<u>xx</u>_ T for some non-zero vector _<u>x</u> 2_ R<sup>_M_</sup> . 

   - (ii) **C** _X_ is the _M ⇥ M_ all-ones matrix. 

#### **Solution:** 

- (i) For a unit length eigenvector _<u>v</u>_ of **C** _X_ , we have 

where (*) follows from the Cauchy-Schwartz inequality and is met with equality if _<u>x</u>_ and _<u>v</u>_ are colinear. Hence, the largest eigenvalue is _kxk_ which corresponds to the eigenvector _<u>v</u>_ = _<u>x/kxk</u>_ 

- (ii) This case corresponds to the case in (i) with _<u>x</u>_ = [1 _,_ 1 _, . . . ,_ 1]<sup>T</sup> . Thus, 

_Page 15 - End of exam_

---

## 源文件

- [[attachments/documents/AI_Machine-Learning-in-Communication-544538350872/mlcomm_exam (2).pdf|mlcomm_exam (2).pdf]]
