---
title: tutorial_em
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-8365f17eabec/tutorial_em.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-8365f17eabec/tutorial_em.pdf
  title: tutorial_em.pdf
---

# tutorial_em

_1 Derviation of EM_ 

# **1 Derviation of EM** 

In the derivation of the EM algorithm we employed the non-negativity of the KullbackLeibler divergence to show that the best lower bound can be obtained by choosing the auxiliary _QX_ ( _x_ ) as the posterior of the latent variable, i.e., 

Now, let’s do the same with the help of the Karush-Kuhn-Tucker (KKT) necessary conditions. The optimization problem reads as: 

1. Formulate the Lagrangian function _f_ ( _<u>q, µ, λ</u>_ ), where _<u>q</u>_ = ( _q_ 1 _, . . . , qM_ )<sup>T</sup> is the probability vector, _µ_ is the Lagrangian multiplier corresponding to the equality constraint<sup>P</sup><sup>_M_</sup> _j_ =1<sup>_pj_=1and</sup><sup>_λj, j_=1</sup><sup>_, . . . , M_aretheLagrangianmultiplierscorre-</sup> sponding to the non-negativity constraints _qj ≥_ 0. 

2. Calculate the derivative of the Lagrangian with respect to _qj_ . 

3. Formulate the primal, dual feasibility and complementary slackness constraints. 

4. Determine the stationary point for _<u>q</u>_ . 

# **2 EM for Amplitude Shift Keying** 

We consider the following additive noise communication channel 

_Y_ = ∆ _X_ + _Z, Z ⇠N_ (0 _, σ_<sup>2</sup> ) 

where 

- the transmit signal _X_ is taken from a normalized 8-ASK set, i.e., _X_ = _{−_ 7 _, −_ 5 _, . . . ,_ 5 _,_ 7 _}_ and ∆ _>_ 0 is a constellation scaling parameter, 

- _Z ⇠N_ (0 _, σ_<sup>2</sup> ) is Gaussian noise that is independent across channel uses, 

- _Y_ is the received signal. 

Suppose that we are given _N_ samples _yi, i_ = 1 _, . . . , N_ . Using the EM algorithm, we wish to estimate _<u>✓</u>_ = ( _PX ,_ ∆ _, σ_<sup>2</sup> ) where we know that _PX_ ( _a_ ) = _PX_ ( _−a_ ) for _a 2 {_ 1 _,_ 3 _,_ 5 _,_ 7 _}_ . **Tasks:** 

1. Determine the distribution _pY_ of _Y_ for given _<u>✓</u>_ 

2. Determine the E-Step. 

3. Determine the M-Step. 

4. In the jupyter notebook _mlcomm_ _~~e~~ m_ _~~A~~ SK.ipynb_ , implement the EM algorithm and estimate _<u>✓</u>_ . 

1 

_3 Binary Asymmetric Channel_ 

# **3 Binary Asymmetric Channel** 

In this task we consider a binary asymmetric channel, which has a channel transition PMF of the form 

with _X_ = _{_ 0 _,_ 1 _}_ and input PMF _PX_ . We observe a number of _N_ channel realizations _Y_ 1 _, Y_ 2 _, . . . , YN_ , from which the parameters _p_ 1 and _p_ 2 should be estimated blindly by means of the EM algorithm. 

1. Find a closed form representation of _PY |X_ for the cases _x_ = 0 and _x_ = 1, respectively. 

2. Formulate the maximum likelihood objective. 

3. Formulate the E-step of the EM algorithm in the _t_ -th iteration. 

4. Formulate the M-step of the EM algorithm in the _t_ -th iteration. 

5. Determine the solution of the optimization problem in the previous subtask. 

6. Create a notebook, implement the source and channel model and estimate _<u>✓</u>_ = ( _p_ 1 _, p_ 2) for _PX_ (0) = 0 _._ 2, _PX_ (1) = 0 _._ 8, _p_ 1 = 0 _._ 1, _p_ 2 = 0 _._ 4 and _N_ = 10000. 

# **4 K-Means** 

In this task, we ask you to implement the K-Means algorithm to find the clusters of a given set of observed data points _{yi}_<sup>_N_</sup> _i_ =1<sup>.Usethenotebook</sup><sup>_mlcomm_</sup> _kmeans.ipynb_ for implementation and test your algorithm with the data provided there. 

The K-Means algorithm works as follows. We denote the coordinates of the initial clusters as _{Cj}_<sup>_K_</sup> _j_ =1<sup>.</sup> 

1. Initialize the _N ⇥ K_ matrix _<u>Z</u>_ with zeros. 

2. Assign each observed data point _i_ = 1 _, . . . , N_ to cluster _j 2_ 1 _, . . . , K_ via the following rule: 

We store this assignment via _zij_ = 1. 

2 

_4 K-Means_ 

3. Update the cluster centers by minimizing the objective 

with respect to _Cj_ . The resulting expression for _Cj_ can be provided in closed form. What is it? 

4. Reset _<u>Z</u>_ = **0** and repeat steps 2 – 4 until an appropriate convergence criterion has been met. 

Your function should have the following signature: 

```
defkmeans(y,centers,iters):
```

```
pass
```

Here, `y` is a `numpy` double array and centers defines the the cluster centroids which are used in the first iteration of the algorithm. Vectorize your code to speed it up. 

3

---

## 源文件

- [tutorial_em.pdf](attachments/documents/AI_Machine-Learning-in-Communication-8365f17eabec/tutorial_em.pdf)
