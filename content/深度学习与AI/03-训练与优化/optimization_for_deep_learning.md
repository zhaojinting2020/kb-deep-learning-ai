---
title: optimization_for_deep_learning
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-b6b8d2d51b41/optimization_for_deep_learning.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-b6b8d2d51b41/optimization_for_deep_learning.pdf
  title: optimization_for_deep_learning.pdf
---

# optimization_for_deep_learning

~~a~~ 

Insight AYLIEN 

<mark>a</mark> @ ~~=~~ = = Hac 

~~eee~~ 

° e e ° e @ \"(@) J(@) local) minimum g* 7] <mark>o></mark> «#8 <u>=></u> <=> EB Hace 

Gradient descent variants 

# Gradient descent variants 

> 1 Batch gradient descent 

> 2 Stochastic gradient descent 

> 3 Mini-batch gradient descent 

Di↵erence: Amount of data used per update 

<mark>4 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Gradient descent variants Batch gradient descent 

Batch gradient descent 

Computes gradient with the **entire** dataset. Update equation: _✓_ = _✓ − ⌘ · r✓J_ ( _✓_ ) `for i in range (nb_epochs ): params_grad = evaluate_gradient ( loss_function , data , params) params = params - learning_rate * params_grad` Listing 1: Code for batch gradient descent update 

<mark>24.11.17 5 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent variants Batch gradient descent 

## Pros: 

Guaranteed to converge to **global** minimum for **convex** error surfaces and to a **local** minimum for **non-convex** surfaces. 

Cons: 

**Very slow** . 

Intractable for datasets that **do not fit in memory** . **No online learning** . 

<mark>6 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Gradient descent variants Stochastic gradient descent 

# Stochastic gradient descent 

Computes update for **each** example _x_<sup>(</sup><sup>_i_)</sup> _y_<sup>(</sup><sup>_i_)</sup> . Update equation: _✓_ = _✓ − ⌘ · r✓J_ ( _✓_ ; _x_<sup>(</sup><sup>_i_)</sup> ; _y_<sup>(</sup><sup>_i_)</sup> ) `for i in range (nb_epochs ): np.random.shuffle(data) for example in data: params_grad = evaluate_gradient ( loss_function , example , params) params = params - learning_rate * params_grad` Listing 2: Code for stochastic gradient descent update 

<mark>24.11.17 7 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

e 

e 

e 

e 

@ 

~~=~~ 

= 

= 

<mark>o</mark> a 

DAG 

e 

DAS 

<mark>o</mark> 

a 

~~=~~ 

= 

Ee 

Gradient descent variants Mini-batch gradient descent 

# Mini-batch gradient descent 

Performs update for every **mini-batch** of _n_ examples. Update equation: _✓_ = _✓ − ⌘ · r✓J_ ( _✓_ ; _x_<sup>(</sup><sup>_i_:</sup><sup>_i_+</sup><sup>_n_)</sup> ; _y_<sup>(</sup><sup>_i_:</sup><sup>_i_+</sup><sup>_n_)</sup> ) `for i in range (nb_epochs ): np.random.shuffle(data) for batch in get_batches(data , batch_size =50): params_grad = evaluate_gradient ( loss_function , batch , params) params = params - learning_rate * params_grad` Listing 3: Code for mini-batch gradient descent update 

<mark>24.11.17 10 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent variants 

Mini-batch gradient descent 

## Pros 

**Reduces variance** of updates. Can exploit **matrix multiplication** primitives. 

## Cons 

**Mini-batch size** is a hyperparameter. Common sizes are 50-256. Typically the algorithm of choice. 

Usually referred to as SGD even when mini-batches are used. 

<mark>11 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Gradient descent variants Mini-batch gradient descent 

|**Method**|**Accuracy**|**Update**<br>**Speed**|**Memory**<br>**Usage**|**Online**<br>**Learning**|
|---|---|---|---|---|
|**Batch**<br>gradient descent|Good|Slow|High|No|
|**Stochastic**<br>gradient descent|Good (with<br>annealing)|High|Low|Yes|
|**Mini-batch**<br>gradient descent|Good|Medium|Medium|Yes|

Table: Comparison of trade-o↵s of gradient descent variants 

<mark>24.11.17 12 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Challenges 

# Challenges 

Choosing a **learning rate** . Defining an **annealing schedule** . Updating features to **di↵erent extent** . **Avoiding suboptimal minima** . 

<mark>24.11.17 13 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms 

# Gradient descent optimization algorithms 

> 1 Momentum 

> 2 Nesterov accelerated gradient 

> 3 Adagrad 

> 4 Adadelta 

> 5 RMSprop 

> 6 Adam 

> 7 Adam extensions 

<mark>14 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

~~ee~~ 

e@ e @ befre Ve= Va.J (Oss) 

“8 

t tA 

@ 

e@ 

<mark>o</mark> a ~~=~~ = =z DNAS 

~~eee~~ 

° 

° 

<mark>o</mark> a ~~=~~ = = DAG 

Gradient descent optimization algorithms Adagrad 

# Adagrad 

Previous methods: **Same learning rate** _⌘_ for all parameters _✓_ . Adagrad [Duchi et al., 2011] **adapts** the learning rate to the parameters ( **large** updates for **infrequent** parameters, **small** updates for **frequent** parameters). 

SGD update: _✓t_ +1 = _✓t − ⌘ · gt_ 

_gt_ = _r✓t J_ ( _✓t_ ) 

Adagrad divides the learning rate by the **square root of the sum of squares of historic gradients** . Adagrad update: 

_Gt 2_ R<sup>_d⇥d_</sup> : diagonal matrix where each diagonal element _i, i_ is the sum of the squares of the gradients w.r.t. _✓i_ up to time step _t ✏_ : smoothing term to avoid division by zero _⊙_ : element-wise multiplication 

<mark>24.11.17 18 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms 

Adagrad 

## Pros 

Well-suited for dealing with **sparse data** . Significantly **improves robustness** of SGD. Lesser need to manually tune learning rate. 

## Cons 

**Accumulates squared gradients** in denominator. Causes the learning rate to **shrink** and become **infinitesimally small** . 

<mark>19 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Gradient descent optimization algorithms Adadelta 

# Adadelta 

Adadelta [Zeiler, 2012] restricts the window of accumulated past gradients to a **fixed size** . SGD update: 

Defines **running average** of squared gradients _E_ [ _g_<sup>2</sup> ] _t_ at time _t_ : 

Adagrad update: 

Preliminary Adadelta update: 

<mark>24.11.17 20 / 49</mark> 

<mark>Sebastian Ruder</mark> 

Gradient descent optimization algorithms Adadelta 

Denominator is just root mean squared (RMS) error of gradient: 

Note: **Hypothetical units do not match** . Define **running average of squared parameter updates** and RMS: 

<mark>24.11.17 21 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms RMSprop 

# RMSprop 

Developed independently from Adadelta around the same time by Geo↵Hinton. 

Also divides learning rate by a **running average of squared gradients** . 

RMSprop update: 

(12) 

_γ_ : decay parameter; typically set to 0 _._ 9 _⌘_ : learning rate; a good default value is 0 _._ 001 

<mark>24.11.17 22 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms 

Adam 

# Adam 

Adaptive Moment Estimation (Adam) [Kingma and Ba, 2015] also stores **running average of past squared gradients** _vt_ like Adadelta and RMSprop. Like Momentum, stores **running average of past gradients** _mt_ . 

_mt_ = _β_ 1 _mt−_ 1 + (1 _− β_ 1) _gt_ (13) _vt_ = _β_ 2 _vt−_ 1 + (1 _− β_ 2) _gt_<sup>2</sup> 

_mt_ : first moment (mean) of gradients _vt_ : second moment (uncentered variance) of gradients _β_ 1 _, β_ 2: decay rates 

<mark>24.11.17 23 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms Adam 

_mt_ and _vt_ are initialized as 0-vectors. For this reason, they are biased towards 0. 

Compute bias-corrected first and second moment estimates: 

(14) 

Adam update rule: 

_✓t_ +1 = _✓t −_ ˆ _<u>⌘</u>_ _~~p~~ vt_ + _✏_<sup>_m_ˆ</sup><sup>_t_</sup> 

(15) 

<mark>24.11.17 24 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms Adam extensions 

# Adam extensions 

## 1 AdaMax [Kingma and Ba, 2015] Adam with _`1_ norm 

> 2 Nadam [Dozat, 2016] Adam with Nesterov accelerated gradient 

<mark>24.11.17 25 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Gradient descent optimization algorithms Update equations 

# Update equations 

Method Update equation _gt_ = _r✓t J_ ( _✓t_ ) SGD ∆ _✓t_ = _−⌘ · gt ✓t_ = _✓t_ + ∆ _✓t_ Momentum ∆ _✓t_ = _−γ vt−_ 1 _− ⌘gt_ NAG ∆ _✓t_ = _−γ vt−_ 1 _− ⌘r✓J_ ( _✓ − γvt−_ 1) _<u>⌘</u>_ Adagrad ∆ _✓t_ = _−_ _~~p~~ Gt_ + _✏_<sup>_⊙gt_</sup> Adadelta ∆ _✓t_ = _−_<sup>_RMS_</sup><sup><u>[∆</u></sup><sup>_✓_</sup><sup><u>]</u></sup><sup>_t−_1</sup> _gt RMS_ [ _g_ ] _t_ _<u>⌘</u>_ RMSprop ∆ _✓t_ = _− gt_ ~~<u>p</u>~~ _E_ [ _g_<sup>2</sup> ] _t_ + _✏_ _<u>⌘</u>_ Adam ∆ _✓t_ = _−_ _~~<u>p</u>~~ v_ ˆ _t_ + _✏_<sup>_m_ˆ</sup><sup>_t_</sup> 

### Table: Update equations for the gradient descent optimization algorithms. 

<mark>24.11.17 26 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

~~=~~ 

= 

= 

DAG 

<mark>o</mark> a 

Gradient descent optimization algorithms Comparison of optimizers 

Which optimizer to choose? 

Adaptive learning rate methods (Adagrad, Adadelta, RMSprop, Adam) are **particularly useful for sparse features** . 

Adagrad, Adadelta, RMSprop, and Adam work well in similar circumstances. 

- [Kingma and Ba, 2015] show that bias-correction helps Adam **slightly outperform RMSprop** . 

<mark>28 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Parallelizing and distributing SGD 

# Parallelizing and distributing SGD 

- 1 Hogwild! [Niu et al., 2011] 

Parallel SGD updates on CPU 

Shared memory access **without parameter lock** Only works for **sparse input data** 

- 2 Downpour SGD [Dean et al., 2012] 

**Multiple replicas** of model on subsets of training data run in parallel Updates sent to parameter server; **updates fraction of model parameters** 

- 3 Delay-tolerant Algorithms for SGD [Mcmahan and Streeter, 2014] Methods also adapt to **update delays** 

> 4 TensorFlow [Abadi et al., 2015] 

Computation graph is split into a **subgraph for every device** Communication takes place using Send/Receive node pairs 

- 5 Elastic Averaging SGD [Zhang et al., 2015] 

**Links parameters elastically** to a center variable stored by parameter server 

<mark>24.11.17 29 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

@ 

e 

e 

@ 

e 

e 

@ 

e 

e@ 

e 

e 

<mark>o</mark> a ~~=~~ = = DAG 

~~<mark>a</mark>~~ 

a ~~=~~ = Ee DAS 

Outlook Tuned SGD vs. Adam 

# Tuned SGD vs. Adam 

Many recent papers use **SGD with learning rate annealing** . SGD with tuned learning rate and momentum is **competitive with Adam** [Zhang et al., 2017b]. 

Adam **converges faster** , but **underperforms SGD** on some tasks, e.g. Machine Translation [Wu et al., 2016]. Adam with **2 restarts and SGD-style annealing** converges faster and outperforms SGD [Denkowski and Neubig, 2017]. **Increasing the batch size** may have the same e↵ect as decaying the learning rate [Smith et al., 2017]. 

<mark>32 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

e@ 

e@ 

<mark>o</mark> a 

~~=~~ = =z DNAS 

Outlook 

Learning to optimize 

(16) 

# Discovered update rules 

**PowerSign** : 

_↵_ : often _e_ or 2 

_f_ : either 1 or a decay function of the training step _t_ 

_m_ : moving average of gradients 

Scales update by _↵_<sup>_f_(</sup><sup>_t_)</sup> or 1 _/↵_<sup>_f_(</sup><sup>_t_)</sup> depending on whether the direction of the gradient and its running average agree. 

**AddSign** : 

(17) 

_↵_ : often 1 or 2 

Scales update by _↵_ + _f_ ( _t_ ) or _↵ − f_ ( _t_ ). 

<mark>36 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Outlook Understanding generalization in Deep Learning 

# Understanding generalization in Deep Learning 

**Optimization** is closely tied to **generalization** . 

The number of possible local minima **grows exponentially with the number of parameters** [Kawaguchi, 2016]. 

Di↵erent local minima **generalize to di↵erent extents** . Recent insights in understanding generalization: 

Neural networks can **completely memorize random inputs** [Zhang et al., 2017a]. 

Sharp minima found by batch gradient descent have **high generalization error** [Keskar et al., 2017]. 

Local minima that generalize well can be **made arbitrarily sharp** [Dinh et al., 2017]. 

Several submissions at ICLR 2018 on understanding generalization. 

<mark>37 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Outlook 

Case studies 

# Case studies 

Deep Biaffine Attention for Neural Dependency Parsing [Dozat and Manning, 2017] 

Adam with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 9 Report large positive impact on final performance of lowering _β_ 2 Attention is All You Need [Vaswani et al., 2017] 

Adam with _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 98, _✏_ = 10<sup>_−_9</sup> , learning rate _⌘ ⌘_ = _d_ model<sup>_−_0</sup><sup>_._5</sup><sup>_·_min(</sup><sup>_step_</sup> _~~n~~ um_<sup>_−_0</sup><sup>_._5</sup> _, step_ _~~n~~ um · warmup_ _~~s~~ teps_<sup>_−_1</sup><sup>_._5</sup> ) _warmup steps_ = 4000 

<mark>38 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Outlook Case studies 

Thank you for attention! For more details and derivations of the gradient descent optimization algorithms, refer to [Ruder, 2016]. 

<mark>24.11.17 39 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Bibliography 

# Bibliography I 

[Abadi et al., 2015] Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Kaiser, L., Kudlur, M., Levenberg, J., Man, D., Monga, R., Moore, S., Murray, D., Shlens, J., Steiner, B., Sutskever, I., Tucker, P., Vanhoucke, V., Vasudevan, V., Vinyals, O., Warden, P., Wicke, M., Yu, Y., and Zheng, X. (2015). 

TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. 

- [Bello et al., 2017] Bello, I., Zoph, B., Vasudevan, V., and Le, Q. V. (2017). 

Neural Optimizer Search with Reinforcement Learning. In _Proceedings of the 34th International Conference on Machine Learning_ . 

<mark>40 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography II 

[Bengio et al., 2009] Bengio, Y., Louradour, J., Collobert, R., and Weston, J. (2009). Curriculum learning. 

_Proceedings of the 26th annual international conference on machine learning_ , pages 41–48. 

[Dean et al., 2012] Dean, J., Corrado, G. S., Monga, R., Chen, K., Devin, M., Le, Q. V., Mao, M. Z., Ranzato, M. A., Senior, A., Tucker, P., Yang, K., and Ng, A. Y. (2012). Large Scale Distributed Deep Networks. 

_NIPS 2012: Neural Information Processing Systems_ , pages 1–11. 

[Denkowski and Neubig, 2017] Denkowski, M. and Neubig, G. (2017). Stronger Baselines for Trustable Results in Neural Machine Translation. In _Workshop on Neural Machine Translation (WNMT)_ . 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17 41 / 49</mark> 

<mark>Sebastian Ruder</mark> 

Bibliography 

# Bibliography III 

- [Dinh et al., 2017] Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. (2017). 

Sharp Minima Can Generalize For Deep Nets. In _Proceedings of the 34 th International Conference on Machine Learning_ . 

- [Dozat, 2016] Dozat, T. (2016). 

Incorporating Nesterov Momentum into Adam. _ICLR Workshop_ , (1):2013–2016. 

[Dozat and Manning, 2017] Dozat, T. and Manning, C. D. (2017). Deep Biaffine Attention for Neural Dependency Parsing. In _ICLR 2017_ . 

<mark>42 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography IV 

[Duchi et al., 2011] Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. 

_Journal of Machine Learning Research_ , 12:2121–2159. 

[Huang et al., 2017] Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., and Weinberger, K. Q. (2017). Snapshot Ensembles: Train 1, get M for free. In _Proceedings of ICLR 2017_ . 

[Io↵e and Szegedy, 2015] Io↵e, S. and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. _arXiv preprint arXiv:1502.03167v3_ . 

<mark>24.11.17 43 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

Bibliography 

# Bibliography V 

[Kawaguchi, 2016] Kawaguchi, K. (2016). Deep Learning without Poor Local Minima. In _Advances in Neural Information Processing Systems 29 (NIPS 2016)_ . [Keskar et al., 2017] Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. 

In _Proceedings of ICLR 2017_ . 

[Kingma and Ba, 2015] Kingma, D. P. and Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. _International Conference on Learning Representations_ , pages 1–13. 

<mark>44 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography VI 

[Loshchilov and Hutter, 2017] Loshchilov, I. and Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. In _Proceedings of ICLR 2017_ . 

[Mcmahan and Streeter, 2014] Mcmahan, H. B. and Streeter, M. (2014). Delay-Tolerant Algorithms for Asynchronous Distributed Online Learning. 

_Advances in Neural Information Processing Systems (Proceedings of NIPS)_ , pages 1–9. 

[Neelakantan et al., 2015] Neelakantan, A., Vilnis, L., Le, Q. V., Sutskever, I., Kaiser, L., Kurach, K., and Martens, J. (2015). Adding Gradient Noise Improves Learning for Very Deep Networks. pages 1–11. 

<mark>45 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography VII 

- [Nesterov, 1983] Nesterov, Y. (1983). 

A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). _Doklady ANSSSR (translated as Soviet.Math.Docl.)_ , 269:543–547. 

[Niu et al., 2011] Niu, F., Recht, B., Christopher, R., and Wright, S. J. (2011). Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent. 

pages 1–22. 

[Qian, 1999] Qian, N. (1999). On the momentum term in gradient descent learning algorithms. _Neural networks : the official journal of the International Neural Network Society_ , 12(1):145–151. 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17 46 / 49</mark> 

<mark>Sebastian Ruder</mark> 

Bibliography 

# Bibliography VIII 

[Ruder, 2016] Ruder, S. (2016). An overview of gradient descent optimization algorithms. _arXiv preprint arXiv:1609.04747_ . 

[Smith et al., 2017] Smith, S. L., Kindermans, P.-J., and Le, Q. V. (2017). Don’t Decay the Learning Rate, Increase the Batch Size. In _arXiv preprint arXiv:1711.00489_ . 

[Vaswani et al., 2017] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention Is All You Need. 

In _Advances in Neural Information Processing Systems_ . 

<mark>47 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography IX 

[Wu et al., 2016] Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., Klingner, J., Shah, A., Johnson, M., Liu, X., Kaiser, L., Gouws, S., Kato, Y., Kudo, T., Kazawa, H., Stevens, K., Kurian, G., Patil, N., Wang, W., Young, C., Smith, J., Riesa, J., Rudnick, A., Vinyals, O., Corrado, G., Hughes, M., and Dean, J. (2016). Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. _arXiv preprint arXiv:1609.08144_ . 

[Zeiler, 2012] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. _arXiv preprint arXiv:1212.5701_ . 

<mark>48 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark> 

<mark>24.11.17</mark> 

Bibliography 

# Bibliography X 

- [Zhang et al., 2017a] Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017a). 

Understanding deep learning requires rethinking generalization. In _Proceedings of ICLR 2017_ . 

[Zhang et al., 2017b] Zhang, J., Mitliagkas, I., and R´e, C. (2017b). YellowFin and the Art of Momentum Tuning. In _arXiv preprint arXiv:1706.03471_ . 

[Zhang et al., 2015] Zhang, S., Choromanska, A., and LeCun, Y. (2015). Deep learning with Elastic Averaging SGD. _Neural Information Processing Systems Conference (NIPS 2015)_ , pages 1–24. 

<mark>24.11.17 49 / 49</mark> 

<mark>Sebastian Ruder</mark> 

<mark>Optimization for Deep Learning</mark>

---

## 源文件

- [optimization_for_deep_learning.pdf](attachments/documents/AI_Machine-Learning-in-Communication-b6b8d2d51b41/optimization_for_deep_learning.pdf)
