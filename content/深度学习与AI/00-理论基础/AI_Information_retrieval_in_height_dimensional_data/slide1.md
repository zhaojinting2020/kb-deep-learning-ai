---
title: slide1
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a617a9e5f764/slide1.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a617a9e5f764/slide1.pdf
  title: slide1.pdf
---

###### **Information Retrieval in High Dimensional Data** Winter Term 2019/20 

PD Dr. rer. nat. Martin Kleinsteuber, Mercateo Group Umer Anwaar, MSc., Mercateo Group Rayyan Khan, MSc., Mercateo Group 

Slide 1/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Outline 

- Introduction 

- About this Course 

- Statistical Decision Making 

- The Curse of Dimensionality 

Slide 2/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### ~~——_—_-—---S SS~~ 

###### What is Machine Learning? 

Slide 4/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### What is Machine Learning? 

_The field of machine learning is concerned with the question of how to construct computer programs that automatically improve with experience_ (Tom Mitchell) _Vast amounts of data are being generated in many fields, and the statisticians’ job is to make sense of it all: to extract important patterns and trends, and to understand “what the data says”. We call this learning from data._ (Trevor Hastie et al.) _Pattern recognition has its origins in engineering, whereas machine learning grew out of computer science. However, these activities can be viewed as two facets of the same field_ (C. Bishop) 

Slide 5/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

# ~~a~~ 

~~es~~ 

How do we extract/store the picture‘s information? What are the chosen "features"? Can we even name them? 

Observation: 

”Features” carry information of the data What are those features? Can we compute them? 

Slide 9/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

<u>|</u> 

> 

> 

~~—_—]|——-Ta, SS.<7~~ 

> 

> 

~~£§_|§~~ 

## a7 <u>_—_-</u> 

PD Dr. Martin Kleinsteuber 

> 

> 

> 

> 

> 

> 

> 

~~—_—_rrSTD~~ 

~~£§_|§~~ 

~~re~~ 

###### Outline 

- Introduction 

- About this course 

- Statistical Decision Making 

- The Curse of Dimensionality 

Slide 18/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

~~—EEEEOSSOSSSSSSSSSSSS~~ 

~~ccc~~ 

~~ccc~~ 

~~ccc‘~~ 

~~—EEEEOSSOSSSSSSSSSSSS~~ 

~~ccc~~ 

~~ccc~~ 

~~ccc‘~~ 

###### Decision Making 

Apple? Classifier Pear? 

• Input data lies in a high-dimensional space • 1 1 _y 2 {− , }_ Discrete label output 

• Task: real-valued _decision function_ , s.t. _y ⇡ f_ ( **x** ) 

Slide 21/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Decision Making 

~~SSS"~~ 

<u>Ts-|::|::.:L:L:L:L:L:L.:.:.LDLD..S</u> 

###### Decision Making 

- **Loss:** nonnegative, real-valued measure for the deviation of the output and decision function. 

- _L_ ( _Y, f_ ( _X_ )) is real Random Variable 

- **Expected Loss:** EPE _L_ ( _f_ ) = E[ _L_ ( _Y, f_ ( _X_ ))] 

- **Examples:** 

- Squared Loss 

- • And others. 

Slide 23/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Decision Making 

- Global Methods 

Find best decision function out of a class _explicit_ , _global_ ˆ _f_ = argmin EPE( _f_ ) _f 2F_ 

- Local Methods 

Find best, _local_ value of decision for a given realization of _c_ ˆ = argmin E _Y X_ = **x** _L_ ( _Y, c_ ) _| c2_ R 

Slide 24/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### – Decision Making Global Methods 

- Predefine a class of functions that is described by finitely 

many parameters: 

- Best decision function is with 

- Estimating EPE via samples of the distribution: 

- Finding an optimal global decision function based on samples: 

ˆ _✓ L_ **x** = argmin ( _yi, f✓_ ( _i_ )) <u>X</u> _✓2_ R<sup>_d_</sup> _i_ 

Slide 25/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

( yi) ) OL IG) xX a, G G 

###### – Decision Making Local Methods 

• Solve _c_ ˆ = argmin E _Y X_ = **x** _L_ ( _Y, c_ ) _| c2_ R 

_−_ E _Y c_ E _Y_<sup>2</sup> _−_ 2 _c_ E _Y c_<sup>2</sup> [( )<sup>2</sup> ] = [ ] [ ] + 

• The best prediction of given a realization of is the conditional mean, if best is measured with respect : to the squared loss 

ˆ _c_ = E _Y X_ = **x** [ _Y_ ] _|_ 

Slide 27/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Decision Making — Local Methods 

(X1, 41) --- (XN, yn) T(x) = {i | x; is close to x} 

Las - -0.5 an 5 H 2 3 4 >A =X <u>E[Y|X =x] © wag DVier(x)</u> Yi 

### ~~a~~ 

Decision Making — Local Methods 

A227224A2232 22Z229d9)N2242 ADQA2BAAAARA 3333%3333% 33233333333 ZFFRBAGIZ334°3 

AXZ2?2BZZAzZB2 2Z22Z2gI\\22\4\2 <> AVDQAIADAP 2AARA 333333 3)3\)3 9 33233 3\3\3 333 SF FRZBAGIG 328 

~~_~~ 

~~_~~ 

~~a~~ 

###### Decision Making in High Dimensions 

###### **Best prediction is based on** 

1. Estimation of probability distributions 

2. Nearest neighbors **BUT in high dimensions...** 

• ... probability estimations become less accurate 

• ... there are no nearest neighbors 

Slide 31/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Summary Decision Making 

###### **Take Home Messages** 

- Two classes of models for Decision Making 

- Global Methods: 

   - Ø incorporate all points in a data set 

Ø reduce complexity by learning parametrized decision function 

- Local Methods: 

   - Ø incorporate only data points in a region of interest 

   - Ø more flexible 

   - Ø causes challenges in high dimensions 

Slide 32/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Outline 

- Introduction 

- Statistical Decision Making 

- The Curse of Dimensionality 

Slide 33/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### The Curse of Dimensionality 

The Curse of Dimensionality 

logi9 N(p) © 0.6(p — 0.25) — N(140) + 7-10°° 

~ 10°? 

~~_~~ 

###### The Curse of Dimensionality 

**Samples are in tails** 

- High dimensional samples tend to lie in distribution tail 

• Example: random vector with           i.i.d. such that _Pr_ ( _Xi_<sup>2</sup><sup>_< β_)</sup><sup>_<_1</sup> for some                 . Then 

Slide 38/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

Fag Aa? Aa? reves “ba AZZ, FEE SD : ea tend Ww bie s° 

~~’''“-_;,,,,,,_|"7-,,,___|7|)sS~~ 

###### The Curse of Dimensionality 

4 4G 12) JB Soom la. , LEH ayy Dyin(p) Er PEE = mini<i<n {dist,(Q, X')} X')} Dymax(p) = maxi<i<n{dist,(Q, X')} X')} p7ooim, ( Path) —IDeaticol(B) Sanh <e) =1 o> 0 

#### ~~TT~~ 

The Curse of Dimensionality 

##### ~~eee~~ 

###### The Curse of Dimensionality 

- **Supervised Learning** 

ØRequires labeled training data ØInformation to be maintained is given by labels Ø same labels to similar feature-values “Map data with , and with different labels to different feature-values” 

- **Unsupervised Learning** 

Øno labels required 

- ØInformation to be maintained is in the intrinsic data structure (e.g. pairwise distances, clusters) 

Ø“Reduce data volume while maintaining intrinsic data structure” 

- **Note:** Hybrid approaches exist (semi-supervised learning) 

Slide 43/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### - Summary The Curse of Dimensionality 

- Unexpected phenomena occur in high dimensions 

   - ØIt is difficult to estimate the underlying distribution. ØSamples tend to lie in the tail of the underlying distribution. 

   - ØSamples tend to be equidistant. 

- Use dimensionality reduction to circumvent these issues 

Slide 44/44 

Information Retrieval in High Dimensional Data 

PD Dr. Martin Kleinsteuber 

###### Summary 

~~$e~~

---

## 源文件

- [slide1.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-a617a9e5f764/slide1.pdf)
