---
title: tutorial_Sampling
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-53756914bd5e/tutorial_Sampling.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-53756914bd5e/tutorial_Sampling.pdf
  title: tutorial_Sampling.pdf
---

# Machine Learning for Communications Sampling Methods Exercises 

Fabian Steiner 

January 24, 2020 

Institute for Communications Engineering Technical University of Munich Prof. Dr. sc. techn. Gerhard Kramer 

The following lecture notes are part of the course “Machine Learning for Communications” o↵ered by the Institute for Communications Engineering at the Technical University of Munich. All content is subject to copyright restrictions. If you are planning to use any of the material, please contact Prof. Dr. sc. techn. Gerhard Kramer ( `gerhard.kramer@tum.de` ). 

2 

_1 Stochastic Matrices_ 

## **1 Stochastic Matrices** 

In the following, we consider _stochastic matrices_<sup>1</sup> **_T_** = <u>�</u> _<u>t</u>_ ~~1~~ _<u>t</u>_ 2 _. . ._ _<u>tK</u>_ � _2_ [0 _,_ 1]<sup>_K⇥K_</sup> which have the property that the entries of each column _<u>t</u>_ _~~i~~_<sup>_, i_= 1</sup><sup>_, . . . , K_sumuptoone,</sup> i.e., 

Similarly, a vector _<u>x</u> 2_ [0 _,_ 1]<sup>_K_</sup> is called _stochastic_ , if<sup>P</sup><sup>_K_</sup> _i_ =1<sup>_xi_=1.Animportantrepre-</sup> sentative of stochastic matrices is the transition matrix of a Markov chain. 

1. Assume that **_A_** is a stochastic matrix and _<u>p</u>_ is a stochastic vector. Show that **_A_** _<u>p</u>_ is again a stochastic vector. 

2. Show that a stochastic matrix **_A_** has always an eigenvalue of 1. _Hint:_ Consider the properties of **_A_**<sup>T</sup> . 

3. Assume now that you are given a transition matrix **_T_** of a Markov chain with eigenvalues _λ_ 1 = 1 _, λ_ 2 _, . . . , λK_ , _λ_ 1 _≥|λ_ 2 _| ≥ . . . ≥|λK|_ . Express the distribution in the _n_ -th state _<u>p</u>_ _~~n~~_<sup>dependingontheeigenvaluesandeigenvectorsof</sup><sup>**_T_**andthe</sup> initial vector _<u>p</u>_ ~~1~~<sup>.</sup><sup>_Hint:_Assumethattheinitialstateallowsarepresentationas</sup> _<u>p</u>_ ~~1~~<sup>=P</sup><sup>_K_</sup> _i_ =1<sup>_aiu_</sup> _~~i~~_<sup>.</sup> The eigenvectors _<u>ui</u>_<sup>_, i_=1</sup><sup>_, . . . , K_areassumedtobelinearly</sup> independent and provide a basis for the initial probability space. 

If you want to find out more about this special class of matrices, we recommend to read related literature about the Perron-Frobenius theorem. 

## **2 Sampling Techniques** 

In this task, we want to review importance and rejection sampling for a simple 1D example. For that purpose, we assume that we want to sample from the following PDF: 

1. Determine the value of _Z_ P by means of numerical integration (see `scipy.integrate` ). 

2. Determine the mean E [ _X_ ] by numerical integration. 

3. Verify your numerical solution by estimating the mean E [ _X_ ] via importance sampling. Think of reasonable choices for the auxiliary distribution _QX_ . The function signature should have the form: 

> 1We note that there are many definitions around. Some people also distinguish between left-stochastic and right-stochastic to di↵erentiate between matrices whose rows or colums sum up to one. You should pay attention to this, when you read related literature. 

3 

~~<u>“Lo”</u>~~ <u>nt</u> ~~<u>esl aana</u> At | Law~~ <u>Fs</u> ~~<u>Cu</u>~~ <u>toe Uy |</u> ~~<u>l</u> Ow Ae|~~ 

~~can be omitted because alk [Del~ Vel< |~~ 

<u>= ais</u> 

_3 Gibbs Sampling_ 

```
defimportance_sampling(Px,Qx,Qx_samples):
pass
```

Hereby, `Px` , `Qx` are function handles: 

- `def Px(x): pass` (implements the target density _PX_ ( _x_ )). 

- `def Qx(x): pass` (implements the auxiliary density _QX_ ( _x_ )). 

The parameter `Qx samples` is a an array containing samples from the auxiliary _QX_ . 

4. Verify your numerical solution by estimating the mean E [ _X_ ] via rejection sampling. Think of reasonable choices for the auxiliary distribution _QX_ . The function signature should have the form: 

   - `def rejection_sampling(Px_star, Qx_star, Qx_sampler, c, num_samples): pass` 

Hereby, `Px` ~~`s`~~ `tar` , `Qx` ~~`s`~~ `tar` and `Qx` ~~`s`~~ `ampler` are function handles. 

   - `def Px` ~~`s`~~ `tar(x): pass` (implements _PX_<sup>_⇤_(</sup><sup>_x_)).</sup> 

   - `def Qx` ~~`s`~~ `tar(x): pass` (implements _Q_<sup>_⇤_</sup> _X_<sup>(</sup><sup>_x_)).</sup> 

   - `def Qx` ~~`s`~~ `ampler(num): pass` (returns a number of `num` samples from the auxiliary distribution _QX_ . 

   - `c` : constant for the rejection procedure. 

   - `num` ~~`s`~~ `amples` : number of samples that should be returned. 

5. Compare the quality of the di↵erent Monte Carlo estimators depending on the number of employed samples. 

## **3 Gibbs Sampling** 

We consider a point-to-point MIMO system with _N_ t transmit and _N_ r receive antennas over a frequency flat fading channel **_H_** _2_ R<sup>_N_r</sup><sup>_⇥N_t</sup> . The corresponding channel model is 

where _<u>X</u> ⇠N_ ( **0** _, σx_<sup>2</sup><sup>**_I_**) and</sup><sup>_<u>N</u>_</sup> _⇠N_ ( **0** _, σn_<sup>2</sup><sup>**_I_**).The entries of</sup><sup>**_H_**are assumed iid zero-mean</sup> and unit-variance Gaussian. 

In this task, we want to estimate the mean-square error estimate, i.e., the conditional mean E ⇥ _<u>X|Y</u>_ = _<u>y</u>_ <u>⇤,</u> of the sent symbol _<u>x</u>_ given the observation _<u>y</u>_ . For the considered scenario, a Markov chain Monto Carlo approach isn’t actually necessary, as the estimate even allows a closed form expression as 

_3 Gibbs Sampling_ 

However, it helps us to get familiar with the new MCMC approach and provides a solution that we can reproduce. 

1. Implement the system model so that you can obtain received samples _<u>y</u>_ . 

2. Derive an expression for _pXk|Y_ _<u>,X</u>_ _~~⇠~~ k_<sup>(</sup><sup>_xk|y_</sup> _<u>, x</u>_ _~~⇠~~ k_<sup>).</sup><sup>_Hint:_The RV</sup><sup>_Xk|y_</sup> _<u>, x</u>_ _~~⇠~~ k_<sup>is Gaussian</sup> (check!) with appropriate mean and variance. Hence, you simply need to derive T 

these two parameters. To do so, proceed as follows. Define _<u>Z</u>_ = � _<u>Y X</u>_ _~~⇠~~ k_ � and calculate E [ _Xk|Z_ = _<u>z</u>_ <u>]</u> and Var [ _Xk|Z_ = _<u>z</u>_ <u>],</u> where 

Here, we have E ⇥ _XkZ_ T<sup>⇤</sup> = **_C_** _XkZ_ and **_C_** _<u>Z</u>_ = E ⇥ _<u>ZZ</u>_ T<sup>⇤</sup> . 

3. Implement a Gibbs sampler using the previously derived expressions. 

4. Compare the quality of the Gibbs sampler for di↵erent sampling strategies (using a long/short chain). 

5

---

## 源文件

- [tutorial_Sampling.pdf](attachments/documents/AI_Machine-Learning-in-Communication-53756914bd5e/tutorial_Sampling.pdf)
