---
title: mlcomm_exam (1)
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-00fdbca9116c/mlcomm_exam
  (1).pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-00fdbca9116c/mlcomm_exam
    (1).pdf
  title: mlcomm_exam (1).pdf
---

# mlcomm_exam (1)

Technische Universit¨at M¨unchen Lehrstuhl f¨ur Nachrichtentechnik Prof. Dr. sc. techn. Gerhard Kramer 

# **Machine Learning for Communications** 

### **Prof. Dr. sc. tech. Gerhard Kramer WS 2018/2019 February 26th, 2019** 

- Write your name and student ID on every sheet. Please have your student card ready for examination. 

- The exam duration is 90 minutes. 

- This exam has 5 questions on 18 pages (excluding the cover page). 

- Do not write with pencils or red pens. 

- You can get full credit if and only if you give reasons for your answer. 

- Problem parts that are marked with an = _)_ can be solved independently of previous parts of the same problem. 

- Please remain seated until the end of the exam. 

- You must hand in this problem set! 

- You are allowed to use a non-programmable calculator. 

## **Good Luck!** 

|Name:||
|---|---|
|Student ID:|Course of Studies:|

### **Please read and sign below:** 

I hereby confirm that I have been informed prior to the beginning of the examination that I have to notify the examination supervisors immediately if sudden illness occurs during the examination. This will be noted in the examination protocol. An application for exam withdrawal has to be filed immediately to the board of examiners being in charge. A medical certificate from one of the physicians acknowledged by the Technische Universit¨at M¨unchen issued on the same day as the examination must be forwarded without delay. In case the examination is completed despite of illness, a subsequent withdrawal due to illness cannot be accepted. In case the examination is ended due to illness it will not be <u>graded.</u> Signature: 

## **For internal use only:** 

|Aufgabe|Punktzahl|Davon erreicht|
|---|---|---|
|Short Questions|14||
|Neural Networks|20||
|Probabilistic Graphical Models|26||
|Expectation Maximization|23||
|Dimensionality Reduction|20||
|**Summe**|103||

~~<u>ee ee</u> Py Lk FP) 7 -~~ Ge 2feals-a7 ~~<u>L_]</u> _~~ NX Cx=B=Cx ¥enPA ~~<u>[|</u>~~ Sem Cr2CxUAWKAA=U AUer ~~<u>o</u>~~ **~~<u>_</u>~~** ~~_~~ Glx=uaur-u aul ~~<u>_|</u> _ —~~ ~~<u>L_</u>~~ A=WUsy 2AAT= Wout ATA= zee" ~~<u>|</u>~~ ; aye ~~<u>7</u>~~ 56 emus sf ARIMA = (sila ve A) ~~L_|~~ ~~<u>[|</u> a~~ de\A)= det (QAG*) = oot (0°84) tA) ~~<u>[|</u>~~ = 419.497)=+(6794) =4(4)= det (A)= ZIfA ~~L_| m~~ C) ty (AA) = tra"ag gA g") ~~<u>L_| L_| [| [|</u>~~ 

Pgh CL= 9 AQ =2=Q AG" Wf genvalves owe one| 2eve 

<u>X</u> Any vector in range( **_A_** ) is an eigenvector of the projector **_P_** . The projector **_P_** is given as **_AA_**<sup>T</sup> . 

**_P_** has full rank. 

The projector **_P_** can be calculated from the eigenvalue decomposition of **_A_** . 

= _)_ h) We want to calculate E [ _f_ ( _X_ )] where _X ⇠ PX_ and use importance sampling (IS) to (1 pt) calculate an approximation of this expected value. Another distribution on the same alphabet as _PX_ is _QX_ . We further have _PX_ ( _x_ ) = _PX_<sup>_⇤_(</sup><sup>_x_)</sup><sup>_/Z_Pand</sup><sup>_QX_(</sup><sup>_x_)=</sup><sup>_Q⇤_</sup> _X_<sup>(</sup><sup>_x_)</sup><sup>_/Z_Q,</sup> where _Z_ P and _Z_ Q are normalization constants. 

_Page 2 - Please turn page_ 

- = _)_ i) We use rejection sampling to estimate the value of _⇡_ . As a proposal distribution, a uniform distribution over the two-dimensional region [ _−_ 1 _,_ 1] _⇥_ [ _−_ 1 _,_ 1] is used and the number of samples which fall into the circle/quadrat is counted. We get the following outcome. 

(2 pt) 

According to the picture above, an approximation of _⇡_ is 

2.92 

   - <u>X</u> 2.86 

      - 3.14 

      - 3.22 

- = _)_ j) Markov Chain Monte Carlo methods approximate the desired distribution, from which (1 pt) sampling should take place, as the 

invariant distribution of a Markov chain. 

stable distribution of a Markov chain. 

reducible distribution of a Markov chain. 

<u>X</u> ergodic distribution of a Markov chain. 

= _)_ k) What is the invariant distribution of the following Markov chain? 

_Page 3 - Please turn page_ 

**(20 points)** 

## **Problem 2: Neural Networks** 

In this task we want to build a classifier that is able to recognize the modulation format that has been used at the transmitter side. In particular, we want to be able to distinguish between 4-ASK and 8-ASK. The input to the classifier is a length _N_ vector _<u>x</u>_ = ( _x_ 1 _, x_ 2 _, . . . , xN_ ) that represents the channel output. 

You are asked to implement a classifier that fulfills the following specifications: 

- Shallow fully, connected feedforward neural network (NN) architecture. 

- One hidden layer with _M_ neurons. 

- Output should indicate to which constellation format the input belongs. 

= _)_ 

- a) Sketch your proposed NN architecture and specify all of its parameters including their (5 pt) dimensions. What activation functions should be used in each layer? What loss function do you propose for training? 

### **Solution:** 

- <sup>_p_</sup> 

- _•_ Hidden layer parameters: **_W_** 1 _2_ R<sup>_M⇥N_</sup> _, b_ ~~1~~<sup>_2_R</sup><sup>_M_</sup> 

- Output layer parameters: **_W_** 2 = _<u>w</u>_ ~~2~~<sup>_2_R</sup><sup>_M, b_</sup> ~~2~~<sup>=</sup><sup>_b_2</sup><sup>_2_R</sup><sup>_p_</sup> 

- Activation functions: ReLU ( _g_ 1) for hidden layer, sigmoid ( _g_ 2) for output layer.<sup>_p_</sup> 

- Loss function: cross-entropy.<sup>_p_</sup> 

= _)_ 

- b) Give an expression for the output of the NN depending on the input _<u>x</u>_ and the NN (3 pt) parameters. 

### **Solution:** 

_Page 4 - Please turn page_ 

= _)_ 

### **Solution:** 

- After hidden layer: 

_<u>z</u>_ ~~1~~<sup>=</sup><sup>**_W_**1</sup><sup>_<u>x</u>_</sup> + _<u>b</u>_ ~~1~~ _<u>a</u>_ 1<sup>=</sup><sup>_g_1(</sup><sup>_<u>z</u>_</sup> ~~1~~<sup>)</sup> 

- c) Assume that the loss function for _S_ feature/label samples <u>(</u> _<u>xi</u>_<sup>_, yi_)isgivenby</sup> 

The function _f_ : R<sup>_N_</sup> _!_ R<sup>_N_</sup> operates componentwise. 

Calculate the gradient of the above loss function with respect to the parameter _<u>c</u>_ . What is the name of the approach to do this systematically for larger NNs? 

### **Solution:** 

_Page 5 - Please turn page_ 

**Solution:** The derivative is: 

- = _)_ d) Assume that in the next product release the NN should be able to recognize more (2 pt) constellation formats. What has to be changed? 

### **Solution:** 

   - We may need to add more hidden layers to increase the NN’s generalization abilities. _p_ 

   - The sigmoid activation function needs to be replaced by a softmax function.<sup>_p_</sup> 

   - The encoding of the labels has to be adapted (one-hot encoding of the modulation classes). 

- = _)_ e) Why is a dataset for neural networks usually split into three parts? What are their (2 pt) names? Why are two not sufficient? 

**Solution:** A dataset for neural networks is usually split into three parts, the training, validation and test data set (<sup>_p_</sup> ). Having only two, i.e., training and test, is not sufficient, because we should not perform the hyperparameter tuning on the test data set, but should employ one, which is di↵erent from the test data to avoid<sup>_<u>p</u>_</sup> . overfitting 

_Page 6 - Please turn page_ 

**(26 points)** 

## **Problem 3: Probabilistic Graphical Models** 

We consider a binary linear block code _C_ defined by its parity-check matrix 

The code _C_ is used for transmission over a binary symmetric channel (BSC) with input _X 2 {_ 0 _,_ 1 _}_ and output _Y 2 {_ 0 _,_ 1 _}_ . The channel law is 

with _p_ = 0 _._ 2. 

= _)_ a) Give a list of the codewords in _C_ . _Hint:_ A codeword fulfills _<u>c</u>_ **_<u>H</u>_**<sup>T</sup> = **0** . 

(2 pt) 

**Solution:** We see that **_H_** has the form **_H_** = � **_P I_** � and for any codeword _<u>c</u>_ = 

� _<u>u p</u>_ <u>�,</u> we get _<u>p</u>_ = _<u>u</u>_ **_<u>P</u>_**<sup>T</sup> . The codewords are therefore: 

- b) The distribution on the first two bits are _PX_ 1(0) = 0 _._ 2 and _PX_ 2(0) = 0 _._ 9. Further, (3 pt) _X_ 1 and _X_ 2 are independent. How does the distribution _PX_ of a codeword _<u>X</u>_ = ( _X_ 1 _, X_ 2 _, X_ 3 _, X_ 4) factorize? What is the probability of each codeword? 

**Solution:** The distribution factorizes as 

Hence, the codewords have the following distribution: 

_Page 7 - Please turn page_ 

ae waymark Pity (714) = arg max 2 Pa ty (zy) = ogre 5 Sao 2 fs.n 4) Ke S01} Xi efor) Kieko} xi 

- f) Calculate _PX_ 1 _|y_ ( _x_ 1 _|y_ <u>)</u> numerically using the sum-product algorithm assuming that _<u>y</u>_ = (7 pt) ( _y_ 1 _, y_ 2 _, y_ 3 _, y_ 4) = (0 _,_ 0 _,_ 1 _,_ 0) was received. 

### **Solution:** 

_PX_ 1 _|Y_ (0 _|y_ ) _/ mf_ 1 _!x_ 1(0) _· mf_ 5 _!x_ 1(0) _· mf_ 6 _!x_ 1(0)<sup>_p_</sup> We now calculate the factors: 

We combine the expressions above: 

= _)_ g) Is the above result exact? Why? 

**Solution:** The solution is exact, because the factor graph is loop free.<sup>_p_</sup> 

_Page 9 - Please turn page_ 

For a simulation experiment, we need to create realizations of Bernoulli distributed variables like _X_ 1 or _X_ 2. Unfortunately, your Python library does not provide such a sampling function, but can only give samples from a continuous uniform distribution between zero and one. 

= _)_ 

- h) Write a Python function that returns `N` realizations of a Bernoulli random variable _X_ (4 pt) with parameter `p` , i.e., 

_Hint:_ Use the function signature: `def bernoulli` ~~`s`~~ `amples(N, p): pass` . The function `np.random.rand(N)` returns _N_ realizations from a uniform distribution between zero and one. 

### **Solution:** 

```
importnumpyasnp
```

```
#[...]
defbernoulli_samples(N,p):
return(np.random.rand(N)<p).astype(int)
```

_Page 10 - Please turn page_ 

**(23 points)** 

## **Problem 4: Expectation Maximization** 

The information theoretic model of a communication channel depends on the transmission and reception method as well as the physical properties of the respective transmission medium. For optical channels, i.e., when rather the particle nature (photon) of light plays a role instead of its nature as an electromagnetic wave, a common model is the Poisson distribution. We model the channel _PY |X_ ( _y|x_ ) as follows: In a given time slot, there is either a pulse, represented by _X_ = +1, or there is no pulse, represented by _X_ = _−_ 1. Further, let _λ_ s be the average number of received photons in a pulsed time slot (when no noise is present) and let _λ_ n be the average number of received noise photons per slot. We then have: 

The probability of sending a pulse or not pulse is iid and given as 

At the receiver, _N_ samples _y_ 1 _, y_ 2 _, . . . yN 2_ N0 were collected, but the model parameters _λ_ n, _λ_ s of (2) and _p_ 0 of (3) are unknown. 

= _)_ a) Write down the log-likelihood function for a maximum-likelihood estimation of the (2 pt) model parameters _λ_ n _, λ_ s and _p_ 0. We collect the parameters in the vector _<u>✓</u>_ = ( _✓_ n _, ✓_ s _, ✓_ p) = ( _λ_ n _, λ_ s _, p_ 0). 

### **Solution:** 

_Page 11 - Please turn page_ 

- = _)_ b) Formulate the E-step of the EM algorithm in the _t_ -th iteration. Simplify the expression (4 pt) as far as possible. 

### **Solution:** 

_Page 12 - Please turn page_ 

- = _)_ c) Formulate the M-step of the expectation maximization (EM) algorithm in the _t_ -th (6 pt) iteration. Simplify the expression as far as possible. 

### **Solution:** 

_✓_<sup>(</sup><sup>_t_+1)</sup> = argmax _L_ ˜( _<u>✓</u>_ <u>)</u> _<u>✓</u> N_ = argmax _<u>✓</u>_ X _i_ =1 _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(</sup><sup>_−_1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ln � _PY |X_ ( _yi| −_ 1; _<u>✓</u>_ <u>)</u> _· ✓_ p� + _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ln � _PY |X_ ( _yi|_ + 1; _<u>✓</u>_ <u>)</u> _· ✓_ p� _pp_ = argmax _<u>✓</u>_ X _i_ =1 _N Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(</sup><sup>_−_1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ln ✓ _✓y_ n _yii_ !<sup>e</sup><sup>_−✓_n</sup><sup>_· ✓_p</sup> ◆ + _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ln ✓ <u>(</u> _✓_ n + _yi ✓_ ! s) _yi_ e<sup>_−_(</sup><sup>_✓_s+</sup><sup>_✓_s)</sup> _·_ (1 _− ✓_ p)◆ _pp N_ = argmax _<u>✓</u>_ X _i_ =1 _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(</sup><sup>_−_1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ( _yi_ ln _✓_ n _−_ ln _yi_ ! _− ✓_ n + ln _✓_ p) + _Q_<sup>(</sup> _X_<sup>_t_)</sup> _i_<sup>(1;</sup><sup>_<u>✓</u>_</sup> ( _t_ )) ( _yi_ ln( _✓_ n + _✓_ s) _−_ ln _yi_ ! _− ✓_ n _− ✓_ s + ln(1 _− ✓_ p))<sup>_pp_</sup> 

_Page 13 - Please turn page_ 

- d) Calculate the derivative of the objective in the M-step with respect to the model pa(4 pt) rameter vector _<u>✓</u>_ <u>.</u> 

### **Solution:** 

_Page 14 - Please turn page_ 

- e) Solve the system of equations in d) and provide the expressions for the parameter (7 pt) ( _t_ +1) 

- updates. That is, give the expressions for _<u>✓</u>_ . 

### **Solution:** 

It’s easy to show from the third equation that 

Regarding _✓_ n<sup>(</sup><sup>_t_+1)</sup> and _✓_ s<sup>(</sup><sup>_t_+1)</sup> we start with the second equation 

and use this result in the one: 

_Page 15 - Please turn page_ 

**(20 points)** 

## **Problem 5: Dimensionality Reduction** 

In this problem we consider the problem of low-rank matrix approximation, i.e., we try to approximate a given full rank matrix **_A_** _2_ R<sup>_M⇥N_</sup> (with _M < N_ ) by a matrix **_A_**<sup>˜</sup> with rank( **_A_**<sup>˜</sup> ) = _K_ and _K <_ rank( **_A_** ). This can be formalized as 

The Frobenius norm _k·k_ F of a matrix **_A_** is defined as _k_ **_A_** _k_<sup>2</sup> F<sup>=P</sup><sup>_M_</sup> _i_ =1 P _Nj_ =1<sup>_a_2</sup> _ij_<sup>.Thesingular</sup> value decomposition (SVD) of **_A_** has the form **_A_** = **_U_ ⌃** **_V_**<sup>T</sup> , where **⌃** = [diag( _σ_ 1 _, . . . , σ_ rank( **_A_** )) **0** ] T and _σ_ 1 _≥ σ_ 2 _≥· · · ≥ σ_ rank( **_A_** ). The columns of **_U_** are _<u>ui</u>_<sup>,therowsof</sup><sup>**_V_**Tare</sup><sup>_<u>v</u>_</sup> _~~i~~_<sup>.</sup> 

= _)_ 

a) What is the rank of **_A_** ? 

**Solution:** We have rank( **_A_** ) = _M_ , as rank( **_A_** ) = min( _M, N_ ) and we know that **_A_** has full rank.<sup>_p_</sup> 

= _)_ 

- b) What are the dimensions of **_U_** , **⌃** and **_V_** ? 

(2 pt) 

**Solution:** The dimensions are 

= _)_ c) Write the SVD of **_A_** as a summation of outer vector products consisting of columns/rows (2 pt) of **_U_** and **_V_** . 

### **Solution:** 

_Page 16 - Please turn page_ 

~~-~~ 

| | 

vant LY)= K 

- f) Use the SVD of a rank _K_ matrix **_B_** = **_U_**<sup>˜</sup> **⌃**<sup>˜</sup> **_V_**<sup>˜T</sup> _2_ R<sup>_M⇥N_</sup> to calculate the approximation (6 pt) error _k_ **_A_** _−_ **_B_** _k_<sup>2</sup> F<sup>andarguethatthespecificchoiceofd)indeedminimizestheerror.</sup> _Hints:_ Use the fact that the product of orthonormal matrices is again an orthonormal matrix and _k_ **_A_** _k_<sup>2</sup> F<sup>= tr(</sup><sup>**_AA_**T).</sup> 

**Solution:** We use the hint to write **_U_**<sup>˜</sup> = **_R_** 1 **_U_** and **_V_**<sup>˜</sup> = **_R_** 2 **_V_** , where **_R_** 1 and **_R_** 2 are orthonormal matrices.<sup>_p_</sup> 

- Choose **_R_** 1 = **_I_** _M_ , **_R_** 2 = **_I_** _N_ , i.e., **_A_** and **_B_** should have the same orthonormal matrices **_U_** and **_V_** . As a result we have: 

- ˜ 

- _•_ The first term above disappears when _σi_ = _σi, i_ = 1 _, . . . , K_ , i.e., **_A_** and **_B_** should share the same singular values. 

= _)_ 

- g) Write a Python function (including its complete signature) that takes a matrix **_A_** and (3 pt) the desired rank _K_ as argument and returns a Python tuple consisting of the rank _K_ approximation as well as the approximation error. _Hint:_ The `numpy` function `[U,S,V] = numpy.linalg.svd(A)` returns the SVD of **_A_** , where `U` and `V` are matrices of respective size and `S` is a row vector. 

### **Solution:** 

```
defrank_approx(A,K):
=
[U,S,V]np.linalg.svd(A)
return(np.matmul(U[:,0:K]*S[0:K],V.T[0:K,:]),np.sum(S[0:K]**2))
```

---

## 源文件

- [[attachments/documents/AI_Machine-Learning-in-Communication-00fdbca9116c/mlcomm_exam (1).pdf|mlcomm_exam (1).pdf]]
