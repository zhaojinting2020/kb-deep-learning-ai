---
title: tutorial_em_solution
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-f6ef6133481c/tutorial_em_solution.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-f6ef6133481c/tutorial_em_solution.pdf
  title: tutorial_em_solution.pdf
---

# tutorial_em_solution

_Machine Learning for Communications 2019_ 

# **Solution: Derviation of EM: KKT conditions** 

Here, we want to show an alternative approach by using the Karush-Kuhn Tucker conditions for convex optimization problems. As pointed out during class we do not expect a full proficiency in convex optimization, but would like to encourage you to understand the basic concepts behind the following derivations. Feel free to step by, if you have specific questions. 

To illustrate the principle we assume the special case of having one sample only, i.e., _N_ = 1. 

1. The Lagrangian function is 

Be carefull with the signs of the Langrangian multipliers for inequality constraints. Depending on whether you want to maximize or minimize the objective, the inequality constraints imposing a “larger or equal than 0” have to be added with a “+” or “-”, respectively. As the elements of _<u>q</u>_ must be a valid PMF, the components _qj_ have to satisfy 0 _ qj _ 1 _, j_ = 1 _, . . . , M_ . The constraint _qj _ 1 is automatically included by demanding that<sup>P</sup> _j_<sup>_qj_=1and</sup><sup>_qj≥_0sothatitisnot</sup> added to the Lagrangian function separately. 

2. The derivative of _f_ ( _<u>q, µ, λ</u>_ <u>)</u> is: 

3. _•_ PF 

4. To find the stationary point, we first have to determine which inequality constraints are not active (i.e., _pj >_ 0) as we must have _λj_ = 0 for those (following from CS). Because of the PF condition<sup>P</sup><sup>_M_</sup> _j_ =1<sup>_qj_= 1,weknowforsurethatnotall</sup><sup>_Mpj_can</sup> be zero. So, for the moment let us assume that we have for some specific _pj0 >_ 0. 

1 

_Machine Learning for Communications 2019_ 

We solve the DF constraint for _qj0_ : 

and plug this into the PF constraint 

to obtain _µ_ = 1 _−_ ln( _pY_ ( _y_ ; _<u>✓</u>_ <u>)).</u> This yields: 

# **Solution: EM for Amplitude Shift Keying** 

You can find the implemention of the EM algorithm in the jupyter notebook `mlcomm` ~~`e`~~ `m.ipynb` . We use the parameter vector _<u>✓</u>_ = ( _<u>p,</u>_ ∆ _, σ_<sup>2</sup> ). 

For the _E_ -step, the auxiliary distributions are calculated for _j_ = 1 _, . . . , M_ as: 

In the M-step, the objective for the _t_ -th iteration 

is optimized regarding our parameters _<u>p, x</u>_ und _σ_<sup>2</sup> . Calculating the derivates yields: 

2 

_Machine Learning for Communications 2019_ 

For the optimization with respect to _pj_ , we have a constrained optimization and formulate the corresponding Langrangian function. Have a look at the Jupyter notebook to play around with the parameters, inspect the convergence and see how the reconstructed looks compared to the real density _pY_ . 

# **Solution: Binary Asymmetric Channel** 

1. We can combine the di↵erent cases into one expression as follows: 

2. The maximum likelihood objective is 

3. In the E-step, we need to find the optimal auxiliary distribution. As derived and shown in the lecture notes by means of the log-sum identity and the properties of the Kullback-Leibler divergence, it is given as 

where 

4. The objective _f_ ( _<u>✓</u>_ <u>)</u> for the maximization in the M-step of the _t_ -th iteration is 

such that we have to 

_Machine Learning for Communications 2019_ 

We can calculate the partial derivatives as 

We compute the stationary points by setting the expressions above to zero and obtain: 

# **Solution: K-Means** 

You can find the implemention of the K-Means algorithm in the Jupyter notebook `mlcomm` ~~`k`~~ `means.ipynb` . 

The expression for finding the updated cluster centers can be derived by calculating the gradient: 

This yields: 

4

---

## 源文件

- [tutorial_em_solution.pdf](attachments/documents/AI_Machine-Learning-in-Communication-f6ef6133481c/tutorial_em_solution.pdf)
