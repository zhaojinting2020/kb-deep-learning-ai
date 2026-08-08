---
title: cs231n_2017_lecture3
source: converted:attachments/documents/AI_CNN-8b037c56d733/cs231n_2017_lecture3.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-8b037c56d733/cs231n_2017_lecture3.pdf
  title: cs231n_2017_lecture3.pdf
---

# Lecture 3: Loss Functions and Optimization 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 1 

April 11, 2017 

#### Administrative 

**Assignment 1** is released: <u>http://cs231n.github.io/assignments2017/assignment1/</u> 

Due **Thursday April 20** , 11:59pm on Canvas (Extending due date since it was released late) 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 2 

April 11, 2017 

#### Administrative 

Check out **Project Ideas** on Piazza 

Schedule for **Office hours** is on the course website TA specialties are posted on Piazza 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 3 

April 11, 2017 

#### Administrative 

Details about redeeming **Google Cloud Credits** should go out today; will be posted on Piazza 

$100 per student to use for homeworks and projects 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 4 

April 11, 2017 

airplane = Pea |— RS automobile fig = FE aa SS Se _ -_ | a bird =a Qs | Cl bal FS a oe x ot E <mark>IEN</mark> deer Ped wt ae ae El SF Bt aoe ES er Sune | rae SERPS 7 a roe ISOC Oe Oe - 7 horse jag (Sed EM ps FE Pa Pa hp FeeeeeSseeeSsSs . tuck = gS a ee of ty 

|airplane|-3.45|=0)51|3.42|
|---|---|---|---|
|automobile|-8:.87|6.04|4.64|
|bird|0.09|Bed|2.65|
|cat<br>|2.9<br>|-4.22<br>|5.1<br>|
|deer|4.48|-4.19|2.64|
|dog<br>|8.02<br>|3.58<br>|5.55<br>|
|frog|3.78|4.49|-4.34|
|horse|1.06|-4,.37|-1.5|
|ship<br>|-0.36<br>|-2.09<br>|-4.79<br>|
|truck|-0.72|-2.93|6.14|

<mark>f(z,W) =W2</mark> 

#1 <mark>U</mark> ~ ~ : J Yi + <mark>n</mark> =!JFYiJFYi S$; —S,. +1 ifotherwisesy,otherwisesy,sy, > 3; +1 = Ss” max(0, max(0, 8; — Sy, + 1) JFY: 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

(71,4) Ly. 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

(71,4) 

Ly. <mark>Li = di</mark> jzy, <mark>max(0, 8; —</mark> Sy, <mark>+ 1)</mark> 

<mark>Li =</mark> do j4y, <mark>max(0,</mark> 8; <mark>—</mark> Sy, <mark>+ 1)</mark> 

Li <mark>= ></mark> j¢y, <mark>Max(0,</mark> 85 <mark>—</mark> Sy, <mark>+ 1)</mark> 

<mark>Li = ></mark> jty, <mark>max(0,</mark> 83 <mark>—</mark> Sy, <mark>+</mark> 1) 

def L_ivectorized(x, y, W): scores = W.dot(x) margins = np.maximum(0, scores - scores[y] + 1) margins[y] = 0 lossi = np.sum(margins) return Lossi 

<mark>f(z,W) =We L= o it diy, max(0, it diy, max(0, diy, max(0, max(0,</mark> f(xi; <mark>W)5 — f(s W)y, f(s W)y, W)y, +</mark> 1) 

<mark>f(z,W) =We L= o it diy, max(0, it diy, max(0, diy, max(0, max(0,</mark> f(xi; <mark>W)5 — f(s W)y, f(s W)y, W)y, +</mark> 1) 

L(W) = ND = ND ND (las). \ 

L(W) = = <mark>al</mark> Fl0 Wm) \ 

A <mark>.</mark> 

<mark>L= = vist dj4y, max(0, f(ri;W); = vist dj4y, max(0, f(ri;W); vist dj4y, max(0, f(ri;W); dj4y, max(0, f(ri;W); max(0, f(ri;W); — f(eis W)y, + W)y, + + 1) +|AR(W) +|AR(W)</mark> 

RW) =d, Wi, RCW) = dog 2a 2a |We| 

2 <mark>R(W) = don da Wa, da Wa, Wa,</mark> 

<mark>x = [1,1,1,1 w1 —[1,0,0,0 we — (00 (000</mark> **<mark>.</mark>** <mark>25, 0.222</mark> **<mark>5</mark>** <mark>, 0.25],,</mark> 

<mark>wi £ = Ws x</mark> —s <mark>|</mark> 

2 <mark>R(W) = don da Wa, da Wa, Wa,</mark> 

<mark>x = [1,1,1,1 w1 —[1,0,0,0 we — (00 (000</mark> **<mark>.</mark>** <mark>25, 0.222</mark> **<mark>5</mark>** <mark>, 0.25],,</mark> 

<mark>wi £ = Ws x</mark> —s <mark>|</mark> 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

> cat **<mark>3.2</mark>** car <mark>5.1 -1.7</mark> frog 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 37 

April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

**scores = unnormalized log probabilities of the classes.** 

> cat **<mark>3.2</mark>** car <mark>5.1 -1.7</mark> frog 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 38 

April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

|cat<br>**3.2**|
|---|
|car<br>5.1|
|frog<br>-1.7|
|unnormalized log probabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 43 

April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

|||unnormalized probabilities|
|---|---|---|
|cat|**3.2**|**24.5**|
|car|5.1|164.0<br>exp|
|frog|-1.7|0.18|
|unnor|malized log p|robabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 44 

April 11, 2017 

|**Soft**|**max Class**|**ifier**(Multinomial Logistic Regression)<br>unnormalized probabilities|
|---|---|---|
|cat<br>car|**3.2**<br>5.1|**24.5**<br>164.0<br>exp<br>normalize<br>**0.13**<br>0.87|
|frog<br>unno|-1.7<br>rmalized log pr|obabilities<br>0.18<br>0.00<br>probabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 45 April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

|||unnormalized prob|abilities<br><br>|
|---|---|---|---|
|cat<br>car|**3.2**<br>5.1|**24.5**<br>164.0<br>exp|normalize<br>**0.13**<br>0.87<br>L_i = -log(0.13)<br>=**0.89**|
|frog<br>unnor|-1.7<br>malized log p|robabilities<br>0.18|0.00<br>probabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 46 April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

Q: What is the min/max possible loss L_i? 

|||unnormalized prob|abilities<br><br>|
|---|---|---|---|
|cat<br>car|**3.2**<br>5.1|**24.5**<br>164.0<br>exp|normalize<br>**0.13**<br>0.87<br>L_i = -log(0.13)<br>=**0.89**|
|frog<br>unnor|-1.7<br>malized log p|robabilities<br>0.18|0.00<br>probabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 47 April 11, 2017 

##### **Softmax Classifier** (Multinomial Logistic Regression) 

Q2: Usually at initialization W is small so all s ≈ 0. <mark>Wh</mark> at is the loss? 

|||unnormalized prob|abilities|What is the loss?|
|---|---|---|---|---|
|cat<br>car|**3.2**<br>5.1|**24.5**<br>164.0<br>exp|normalize|**0.13**<br>0.87<br>L_i = -log(0.13)<br>=**0.89**|
|frog<br>unnor|-1.7<br>malized log p|robabilities<br>0.18||0.00<br>probabilities|

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 48 April 11, 2017 

<mark>= log(<—) Li = do j4y, max(0, max(0, 8; — — Sy, + 1)</mark> 

<mark>ES</mark> —leef <mark>== => — Sy. + + me a ” max(0,</mark> 8; 

<mark>=0</mark> 

s=f(z;W)=We 

s=f(z;W)=We 

# Optimization 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 54 

April 11, 2017 

bestloss = float("inf") high j ‘Loa l for num in xrange(1000): W = np.random.randn(10, 3073) * 0.0001 enerate ; ara E loss = L(X_train, Y_<sup>train,</sup> W) get t t g t if loss < bestloss: # ick of est solt bestloss = loss bestW = W print ‘in attempt %d the loss was %f, best %f' % (num, loss, bestloss) 

scores = Wbest.dot(Xtecols) # 10 x 10000, the class score: rr all test examples Yte predict = np.argmax(scores, axis = 0) np.mean(Yte predict == Yte) 

df(z) _, f(e+h)— fle) dx h 0 h 

##### **<mark>current W:</mark>** 

##### **<mark>gradient dW:</mark>** 

|[0.34,|[?,|
|---|---|
|-1.11,|?,|
|0.78,|?,|
|0.12,|?,|
|0.55,|?,|
|2.81,|?,|
|-3.1,|?,|
|-1.5,|?,|
|0.33,…]|?,…]|
|**loss 1.25347**||

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 61 

April 11, 2017 

##### **<mark>gradient dW:</mark>** 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 62 

April 11, 2017 

##### **<mark>current W:</mark>** 

##### **<mark>W + h :</mark>** <mark>(second dim)</mark> 

##### **<mark>gradient dW:</mark>** 

|[0.34,|[0.34,|[-2.5,|
|---|---|---|
|-1.11,|-1.11 +**0.0001**,|?,|
|0.78,|0.78,|?,|
|0.12,|0.12,|?,|
|0.55,|0.55,|?,|
|2.81,|2.81,|?,|
|-3.1,|-3.1,|?,|
|-1.5,|-1.5,|?,|
|0.33,…]|0.33,…]|?,…]|
|**loss 1.25347**|**loss 1.25353**||

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 64 

April 11, 2017 

##### **<mark>current W:</mark>** 

##### **<mark>W + h :</mark>** <mark>(third dim)</mark> 

##### **<mark>gradient dW:</mark>** 

|[0.34,<br>[0.34,|[-2.5,|
|---|---|
|-1.11,<br>-1.11,|0.6,|
|0.78,<br>0.78 +**0.0001**,|?,|
|0.12,<br>0.12,|?,|
|0.55,<br>0.55,|?,|
|2.81,<br>2.81,|?,|
|-3.1,<br>-3.1,|?,|
|-1.5,<br>-1.5,|?,|
|0.33,…]<br>0.33,…]|?,…]|
|**loss 1.25347**<br>**loss 1.25347**||

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 66 

April 11, 2017 

L= 5 ialitN ialitNN dD, We Li = dij4y, max(0,j4y, max(0, max(0, 8; — Sy, + 1) s=f(tz;W)=We2 

VwL 

> L= = eaNN Li +>>,W? L; = splay max(0,s; max(0,s; — sy, + 1) 

##### **<mark>current W:</mark>** 

##### **<mark>gradient dW:</mark>** 

[0.34, -1.11, 0.78, 0.12, 0.55, 2.81, -3.1, -1.5, 0.33,…] **loss 1.25347** 

[-2.5, dW = ... 0.6, (some function 0, data and W) 0.2, 0.7, -0.5, 1.1, 1.3, -2.1,…] 

Fei-Fei Li & Justin Johnson & Serena Yeung 

71 

Lecture 3 - 

April 11, 2017 

### In summary: 

- Numerical gradient: approximate, slow, easy to write 

- Analytic gradient: exact, fast, error-prone 

=> 

<u>In practice:</u> Always use analytic gradient, but check implementation with numerical gradient. This is called a **gradient check.** 

Fei-Fei Li & Justin Johnson & Serena Yeung 

72 

Lecture 3 - 

April 11, 2017 

while [rue: 

weights grad = evaluate gradient(loss fun, data, weights) weights += - step size * weights grad # perform parameter update 

<mark>original W W 1</mark> _ <mark>negative gradient direction</mark> 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 74 

April 11, 2017 

Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 3 - 75 April 11, 2017 

1 N 71 1 N a=] 

while ss data batch = sample training data(data, 256) # nple 256 example weights grad = evaluate gradient(loss fun, data batch, weights) weights += - step size * weights grad # perform parameter update 

### Interactive Web Demo time.... 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 78 

April 11, 2017 

#### Image Features: Motivation 

Fei-Fei Li & Justin Johnson & Serena Yeung 

## <mark>Next time:</mark> 

##### Introduction to neural networks 

Backpropagation 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 3 - 85 

April 11, 2017

---

## 源文件

- [cs231n_2017_lecture3.pdf](attachments/documents/AI_CNN-8b037c56d733/cs231n_2017_lecture3.pdf)
