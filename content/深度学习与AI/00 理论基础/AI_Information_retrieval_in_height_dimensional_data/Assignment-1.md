---
title: Assignment 1
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a041642c8f5b/Assignment
  1.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a041642c8f5b/Assignment
    1.pdf
  title: Assignment 1.pdf
---

# Assignment 1

TECHNISCHE UNIVERSITÄT MÜNCHEN Fakultät für Elektrotechnik und Informationstechnik Lehrstuhl für Datenverarbeitung PD Dr. Martin Kleinsteuber 

`Information Retrieval in High Dimensional Data Assignment #1` , 31.10.2019 

Due date: 17.11.2019, 10 P.M. 

Please hand in your solutions via Moodle as an IPYTHON (Jupyter) notebook. 

Solutions can be handed in by groups of **four** to **five** people. Please state the names of your group members at a prominent place in your submission. (For example, at the beginning of your provided notebook or in a separate text file.) 

# **Curse of Dimensionality** 

**Task 1:** [2 Points] Let _Cd_ = _{_ **x** _2_ R<sup>_p_</sup> _|k_ **x** _k1 _<sup>_<u>d</u>_</sup> 2<sup>_}_denotethe</sup><sup>_p_-dimensionalhypercubeofedge</sup> length _d_ , centered at the origin. 

- Assume _X_ to be uniformly distributed in _C_ 1. Determine _d_ in dependence of _p_ and _q 2_ [0 _,_ 1], such that 

   - Pr( _X 2 Cd_ ) = _q_ 

holds. 

   - Let the components of the _p_ -dimensional random variable _X_<sup>_p_</sup> be independent and have the standard normal distribution. It is known that Pr( _|X_<sup>1</sup> _| _ 2 _._ 576) = 0 _._ 99. For an arbitrary _p_ , determine the probability Pr( _kX_<sup>_p_</sup> _k1 >_ 2 _._ 576) for any of the components of _X_<sup>_p_</sup> to lie outside of the interval [ _−_ 2 _._ 576 _,_ 2 _._ 576]. Evaluate the value for _p_ = 2, _p_ = 3 and _p_ = 500. 

- **Task 2:** [10 Points] Provide the PYTHON code to the following tasks (the code needs to be commented properly): 

   - Sample 100 uniformly distributed random vectors from the box [ _−_ 1 _,_ 1]<sup>_d_</sup> for _d_ = 2. 

   - For each of the 100 vectors determine the minimum angle to all other vectors. Then compute the average of these minimum angles. Note that for two vectors _x, y_ the cosine of the angle between the two vectors is defined as 

~~<u>P(xeCa)=P( new ett)9 = Baeble 9 plod =p P(xeG)=P( xe 3)</u>~~ 

- Repeat the above for dimensions _d_ = 1 _, . . . ,_ 1000 and use the results to plot the average minimum angle against the dimension. 

- Give an interpretation of the result. What conclusions can you draw for 2 randomly sampled vectors in a _d_ -dimensional space? 

- Does the result change if the sample size increases? 

# **Statistical Decision Making** 

**Task 3:** [10 Points] Answer the following questions. All answers must be 

- The numbers in Figure 1 show the probability of the respective event to happen (e.g. the probability for the event _X_ = 1 and _Y_ = 1 is 0 _._ 02). Is this table a probability table? If so, why? 

- Based on Figure 1 give the conditional expectation E _Y |X_ =2[ _Y_ ] and the probability of the event _X_ = 1 under the condition that _Y_ = 3. 

- Is the function _p_ ( _x, y_ ) given by 

a joint density function for two random variables? 

- For two random variables _X_ and _Y_ the joint density function is given by 

What are the marginal density functions for _X_ and _Y_ respectively? 

- Let the joint density function of two random variables _X_ and _Y_ be given by 

Determine the probability for _X _ 2 under the condition that _Y_ =<sup><u>1</u></sup> 2<sup>.</sup> 

- **Task 4:** [3 Points] Show that the covariance matrix **C** of any random variable _X 2_ R<sup>_p_</sup> is symmetric positive semidefinite, i.e. **C** = **C**<sup>_>_</sup> and **x**<sup>_>_</sup> **Cx** _≥_ 0 for any covariance matrix **C** _2_ R<sup>_p⇥p_</sup> and any **x** _2_ R<sup>_p_</sup> . 

<mark>FLL</mark> <u><mark>Ft IL</mark></u> 

|~~A~~<br>~~o4 4+ 044-24 008°3~~<br><br>|
|---|
|~~2.~~<br>~~Eyy-olt)= = PY l2)-¥ =~~<br>~~O4 + 0.4~~<br>~~+ 0k~~<br>~~= 449}~~|
|~~potsa|ye3) = Pica~~ ~~y23) oy,~~<br>P( 4-3)<br>~<br>~~0-40-06~~|
|~~° ry Po~~ p ~~ax dy = ale 4 ly x =3 F!~~|
|~~mt~~ a ~~ant~~ olensity function<br><br><br>|
|~~4.~~<br>~~pu) “f° perry) dy~~<br>~~P by? =|. poe) Ax~~|
|-<br>~~(* 5 ,-(xt )~~<br>~~=~~<br>~~q~~<br>~~-UtY))~~<br>~~J 2g~~<br>~~ty ly~~<br>~~| 20~~<br>~~ohx~~|
|~~L ref" et oy~~<br>~~= 204 fi o* Ax~~<br>~~Fae ,~~<br>~~= 294 Cox)|~~<br>~~= 2e”~~<br>~~X 7~~<br>i<br>~~=aK:(4-Y‘)(479)~~|

~~5~~ ~~<u>Plxe|y-e) = prea</u> 53)~~ ~~<u>Ply=>.)</u> =~~ ~~<u>J</u> rooted) dy~~ ~~<u>ik</u> ae [ax-) ox~~ ~~<u>4 | ypxt c) |.</u> “t = 083~~

---

## 源文件

- [Assignment 1.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a041642c8f5b/Assignment 1.pdf)
