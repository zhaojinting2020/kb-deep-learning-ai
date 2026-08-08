---
title: tutorial_pca
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-41d644115a41/tutorial_pca.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-41d644115a41/tutorial_pca.pdf
  title: tutorial_pca.pdf
---

# Machine Learning for Communications Principal Component Analysis Programming Exercises 

## Lars Palzer, Tobias Prinz 

January 28, 2020 

Institute for Communications Engineering Technical University of Munich Prof. Dr. sc. techn. Gerhard Kramer 

The following lecture notes are part of the course “Machine Learning for Communications” o↵ered by the Institute for Communications Engineering at the Technical University of Munich. All content is subject to copyright restrictions. If you are planning to use any of the material, please contact Prof. Dr. sc. techn. Gerhard Kramer ( `gerhard.kramer@tum.de` ). 

2 

_1 PCA_ 

## **1 PCA** 

1. Generate data according to the following code snippet: 

```
importnumpyasnp
```

```
fromnumpy.randomimportrandn
P=np.array([[0.5,-0.866],[0.866,0.5]])
```

```
X=P.dot(np.array([[1],[4]])*randn(2,1000)+np.array([[1],[0.5]]))
```

2. Create a scatter plot of the generated data using `matplotlib.pyplot.scatter` . 

3. Implement PCA as introduced in the lecture. Your function should have the following signature: 

   - `def pca(x,param,mode="preserve_var"): return (Y,W)` 

The function should support di↵erent modes. If `mode` is set to `preserve_var` , the target dimension _K_ should be computed as 

where the value of _↵_ is given by `param` . If mode is set to `fixed_k` , we use a fixed _K_ which is then given by `param` . 

Use SVD ( `numpy.linalg.svd` ) on the empirical correlation matrix for computing the eigenvectors and eigenvalues. 

4. Extend the function by adding normalization possibilities. Your function should then have the following signature: 

   - `def pca(x,param,mode="preserve_var",norm_mode="f_wise", scale_var=True): return (Y,W,mu,sigma)` 

`norm_mode` denotes the mode of the normalization. The three modes are: 

- `f_wise` : feature wise mean shift/variance normalization _)_ each feature should have zero mean/unit variance 

- `s_wise` : sample wise mean shift/variance normalization _)_ each sample should have zero mean/unit variance 

- `none` : no mean shift/variance normalization 

`scale_var` is a boolean variable that determines whether the features/samples are scaled to unit variance. 

5. Choose an appropriate mode for mean shift and variance normalization and perform PCA on the given dataset. 

3 

b 

_3 K-Means with the MNIST dataset_ 

3. Assign a label to each cluster and compute the resulting error rate for clustering the data. 

4. Compare the results of K-Means with and without PCA for di↵erent values of _K_ . 

5

---

## 源文件

- [tutorial_pca.pdf](attachments/documents/AI_Machine-Learning-in-Communication-41d644115a41/tutorial_pca.pdf)
