---
title: Assignment 3
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-b9f59207a693/Assignment
  3.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-b9f59207a693/Assignment
    3.pdf
  title: Assignment 3.pdf
---

# Assignment 3

TECHNISCHE UNIVERSITAT<sup>¨</sup> MUNCHEN<sup>¨</sup> Fakult¨at f¨ur Elektrotechnik und Informationstechnik Lehrstuhl f¨ur Datenverarbeitung Prof. Dr. Martin Kleinsteuber 

`Information Retrieval in High Dimensional Data Lab #3: Theoretical Exercises` , 26.04.2018 

# **Logistic Regression** 

Task 1. Consider the binary classification problem of assigning a label _y 2 {−_ 1 _,_ 1 _}_ to a data sample **x** _2_ R<sup>_p_</sup> by means of Logistic Regression. You are given a training set _{_ ( **x** 1 _, y_ 1) _, . . . ,_ ( **x** _N , yN_ ) _}_ of labeled data. Recall that the loss function is given by 

a) Compute the gradient _r_ **w** _,bL_ . 

**Solution** : Applying the chain rule, we get 

Accordingly, we get 

b) Assume that the two classes of the training set are linearly separable, i.e. there is a weight vector **w** _s 2_ R<sup>_p_</sup> and a bias _bs 2_ R such that 

holds. Show that, under this assumption, the loss function has no global minimum ( **w**<sup>_⇤_</sup> _, b_<sup>_⇤_</sup> ) _2_ R<sup>_p_+1</sup> . 

**Solution** : A global minimum of _L_ is a pair ( **w**<sup>_⇤_</sup> _, b_<sup>_⇤_</sup> ) _2_ R<sup>_p_+1</sup> such that 

holds. Furthermore, for non-empty training sets, _L_ is strictly positive, so that we can conclude 

Assume that such a point exist. Let us define 

Observe that _zi_ is strictly positive for every _i_ . Consider the function 

Since every summand approaches 0 as _h_ approaches _1_ , so does _f_ ( _h_ ), i.e. 

Observing the equality 

this means that for any _" >_ 0, we can find an _h 2_ R and set ( **w** _, b_ ) = ( _h_ **ws** _, hbs_ ), such that 

holds, which contradicts the assumption of ( **w**<sup>_⇤_</sup> _, b_<sup>_⇤_</sup> ) with _L_ ( **w**<sup>_⇤_</sup> _, b_<sup>_⇤_</sup> ) = _"_ being a global minimum. 

Note that the hyperplane described by ( **w** _s, bs_ ) does not have to be optimal in any sense. Depending on the algorithm this can lead to a perpetual increase of the norm of ”non-ideal” hyperplane descriptors. 

- c) To avoid the scenario in b), the norm of ( **w** _, b_ ) can be penalized by adding a squared norm regularizer. Consider the modified loss function 

where _λ >_ 0 is a real-valued constant. Compute the gradient _r_ **w** _,bL_<sup>˜</sup> . 

**Solution** : Due to the linearity of the derivative, we have 

and

---

## 源文件

- [Assignment 3.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-b9f59207a693/Assignment 3.pdf)
