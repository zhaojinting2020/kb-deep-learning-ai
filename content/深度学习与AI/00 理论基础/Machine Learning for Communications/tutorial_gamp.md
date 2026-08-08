---
title: tutorial_gamp
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-1b9a65c986c9/tutorial_gamp.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-1b9a65c986c9/tutorial_gamp.pdf
  title: tutorial_gamp.pdf
---

## **Application of Approximate Message Passing: Detection for Quantized Massive MIMO** 

### **Derviation of the Nonlinear Variable Node Functions for BPSK** 

Calculate E [ _X|Y_ = _y_ ] and Var [ _X|Y_ = _y_ ] for _pY X_ ( _y, x_ ) = _pY |X_ ( _y|x_ ) _PX_ ( _x_ ) and _Y |_ ( _X_ = _x_ ) _⇠N_ ( _x_ ; _σ_<sup>2</sup> ) and BPSK input _X_ , i.e., _X 2 X_ = _{−A,_ + _A}_ and _PX_ ( _−A_ ) = _PX_ ( _A_ ) = 0 _._ 5. 

## **Solution** 

We have 

For the variance, we have 

Now suppose that _X_ is a complex-valued QPSK symbol. We obtain the results in the lecture notes can by adopting the previous two expressions for the complex case and treating the real and imaginary parts independently from each other. Note that if _Y |_ ( _X_ = _x_ ) _⇠CN_ ( _x_ ; _σ_<sup>2</sup> ), then R( _Y_ ) _|_ (R( _X_ ) = R( _x_ )) _⇠N_ (R( _x_ ) _, σ_<sup>2</sup> _/_ 2) and similarly for the imaginary part. 

where the second line follows since the R( _X_ ) is independent of I( _Y_ ) and I( _X_ ) is independent of R( _Y_ ). For the variance of the complex random variable _X_ , we have 

| | : ! 

c 

# a ee EM -tutorial 

### **Implementation of GAMP for a Massive MIMO Uplink scenario** 

1. Consider the model _Y_ = Q( _Z_ + _N_ ), where _N_ is additive white Gaussian noise with zero mean and variance _σ_<sup>2</sup> . The function Q : R _! {−_ 1 _,_ +1 _}_ is the one-bit quantization function introduced in the lecture notes in Eq. (3.53). Show that the conditional PMF _PY |Z_ ( _y|z_ ) is given by 

where Φ( _·_ ) is the CDF of a standard normal random variable. _Hint: Distinguish between the cases y_ = +1 _and y_ = _−_ 1 _first and then combine both cases into a single expression._ 

2. Implement an auxiliary function `help` ~~`f`~~ `un` ~~`m`~~ `ean(y, muz, varz, varn)` to calculate value of 

Hereby, we have that 

and 

As indicated in the lecture notes, a closed form expression is possible by following the steps in derivation of `http://www.gaussianprocess.org/gpml/chapters/ RW.pdf` , Chapter 3.9. 

3. Implement an auxiliary function `help` ~~`f`~~ `un` ~~`v`~~ `ar(y, muz, varz, varn)` to calculate value of Var [ _Z|Y_ = _y_ ], where the involved random variables are distributed as shown above. 

   - Again, a closed form solution is possible following the steps in the above mentioned book. 

4. Implement the function `one` ~~`b`~~ `it quant(X)` which implements Q( _·_ ) (Eq. (3.53)). Use vectorization as far as possible. The argument `X` is a (potentially complex) `np.array` . 

2 

5. Implement the function `do` ~~`d`~~ `emod(Xhat, cstll)` , which returns the index of the constellation point in the `np.array cstll[’X’]` that is closest to `Xhat` . Use vectorization and broadcasting as far as possible. 

Hint: We will use the dictionary `cstll` to represent a constellation. For instance, a QPSK constellation may be represented as 

```
cstll=dict()
cstll[’X’]=1/np.sqrt(2)*np.array([-1+1j,-1-1j,+1+1j,+1-1j])
=
cstll[’pX’]1/4.0
cstll[’M’]=4
cstll[’m’]=2
```

6. Write the function `one` ~~`b`~~ `it gamp(H, y, sigma2, cstll, num` ~~`i`~~ `t)` , which implements Algorithm 2 of the AMP chapter in the lecture notes. The meaning of the parameters is as follows: 

   - `H` : Matrix of complex channel coefficients. 

   - `y` : Quantized receive vector. 

   - `sigma2` : Variance of the additive white Gaussian noise. 

   - `cstll` : Dictionary representing the constellation. 

   - `num` ~~`i`~~ `t` : Number of iterations performed. 

It should return the tuple `(xhat, varxhat)` representing the (approximate) MMSE estimate of the transmitted symbol and the corresponding variance after `num` ~~`i`~~ `t` iterations. 

In the Jupyter notebook, you will find some further skeleton code and see how the functions are supposed to interact with each other. 

## **Solution** 

To obtain a single expression, we can use _Q_ ( _x_ ) = Φ( _−x_ ) such that _PY |Z_ (1 _|z_ ) = Φ � _σz_ �. Both cases can now be combined by multiplying the argument of the Φ( _·_ ) 

function with the sign of _y_ . For the complex case, note that the real and imaginary parts of _Z_ are independent and also quantized independently. Thus, we have 

2. See Jupyter notebook. You may also compare this solution to a straightforward approach using numerical integration techniques, e.g., by using `scipy.integrate.quad` . This is also helpful for cross checking. Let _Z ⇠CN_ ( _p, ⌧_ ). We get 

where _φ_ is the standard Gaussian pdf and 

3. See Jupyter notebook. You may also compare this solution to a straightforward approach using numerical integration techniques, e.g., by using `scipy.integrate.quad` . This is also helpful for cross checking. We get 

— To compute the variance of the complex variable, we note that 

Var [ _Z|Y_ = _y_ ] = Var [R( _Z_ ) _|Y_ = _y_ ] + Var [I( _Z_ ) _|Y_ = _y_ ] 

4. See Jupyter notebook. 

5. See Jupyter notebook. 

6. See Jupyter notebook. 

4

---

## 源文件

- [tutorial_gamp.pdf](attachments/documents/AI_Machine-Learning-in-Communication-1b9a65c986c9/tutorial_gamp.pdf)
