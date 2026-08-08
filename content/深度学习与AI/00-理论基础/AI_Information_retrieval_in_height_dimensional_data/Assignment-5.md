---
title: Assignment 5
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-229720ad3541/Assignment
  5.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-229720ad3541/Assignment
    5.pdf
  title: Assignment 5.pdf
---

# Assignment 5

TECHNISCHE UNIVERSITAT<sup>¨</sup> MUNCHEN<sup>¨</sup> Fakult¨at f¨ur Elektrotechnik und Informationstechnik Lehrstuhl f¨ur Datenverarbeitung PD Martin Kleinsteuber 

## `Information Retrieval in High Dimensional Data` 

## `Lab #7, Theoretical Exercises` , 10.01.2019 

# **Kernel PCA** 

Task 1. Let _φ_ : R<sup>_p_</sup> _!_ R<sup>_q_</sup> be a function which maps a vector from the observation space to a feature space. When applied to matrices, let us assume it operates column-wise. The respective kernel shall be defined as 

a) Let **X** _2_ R<sup>_p⇥N_</sup> be a training data matrix. Give an expression for centering _φ_ ( **X** ). Show that it can be written as _φ_ ( **X** ) **H** , where **H** is a square matrix. 

**Solution** : 

b) Define **K** = _φ_ ( **X** )<sup>_>_</sup> _φ_ ( **X** ) and **K**<sup>˜</sup> = **HKH** = ( _φ_ ( **X** ) **H** )<sup>_>_</sup> ( _φ_ ( **X** ) **H** ) with the sorted EVD **K**<sup>˜</sup> = **V⇤V**<sup>_>_</sup> . Let **V** _k 2_ R<sup>_N,k_</sup> denote the first _k_ columns of **V** . Express the leading _k_ left singular vectors **U** _k 2_ R<sup>_q,k_</sup> of _φ_ ( **X** ) **H** in terms of the previously defined matrices. 

### **Solution:** 

c) Let **Y** _2_ R<sup>_p⇥N_test</sup> be a test data matrix. Express its Kernel PCA scores **S** _k_ by using the result from b) and describe how you can compute them, given that __ is known, but _φ_ is not. 

**Solution:** The feature matrix _φ_ ( **Y** ) needs to be centered: 

The scores are given by 

If we want to compute the score of our original training data, i.e. **X** = **Y** , then the expression simplifies to

---

## 源文件

- [Assignment 5.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-229720ad3541/Assignment 5.pdf)
