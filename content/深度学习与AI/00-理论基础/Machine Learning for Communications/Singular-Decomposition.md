---
title: Singular Decomposition
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-78b10208f7b9/Singular
  Decomposition.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-78b10208f7b9/Singular
    Decomposition.pdf
  title: Singular Decomposition.pdf
---

# Singular Decomposition

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

# COM521500 

Math. Methods for SP I Lecture 4: Singular Value Decomposition & Orthogonal Projection 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

1 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Theorem 4.1 Every A ∈ C<sup>m×n</sup> can be decomposed as 

where U ∈ C<sup>m×m</sup> and V ∈ C<sup>n×n</sup> are unitary, and 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

2 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

The values σi are called the singular values of A. The columns ui & vi of U & V are called the left and right singular vectors of A. 

Outer product representation of SVD: 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

3 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Relationship with the 2-norm: 

Recall ∥A∥2 =<sup>√</sup> λmax, where λmax is the max. eigenvalue of A<sup>H</sup> A. 

By SVD A = UΣV<sup>H</sup> , 

It follows that the eigenvalues of A<sup>H</sup> A are σi<sup>2,andthatthe</sup> eigenvector matrix of A<sup>H</sup> A is V. Thus, 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

4 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Relationship with eigendecomposition: 

Consider a Hermitian A ∈ C<sup>n×n</sup> . Eigendcomposition: 

SVD: 

Hence, for Hermitian A we have U = V = Q & Λ = Σ. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 5 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Partitioning the SVD 

Suppose that the number of nonzero singular values is r ≤ p; i.e., σr+1 = σr+2 = . . . σp = 0. 

The SVD can be rewritten as 

where Σ<sup>˜</sup> = Diag(σ1, . . . , σr) ∈ R<sup>r×r</sup> , U1 ∈ C<sup>m×r</sup> , U2 ∈ C<sup>m×m−r</sup> , V1 ∈ C<sup>n×r</sup> , and V2 ∈ C<sup>n×m−r</sup> . 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

6 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 7 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Inverse 

## Consider a square, nonsingular A. 

An alternate form of the inverse: 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

8 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Linear System of Equations 

Given A ∈ C<sup>m×n</sup> , b ∈ C<sup>n</sup> , the problem of the linear system of eqns. is find an x ∈ C<sup>m</sup> (or multiple x’s) such that 

We have learnt that for m = n, Ax = b is always satisfied if A is nonsingular. 

Can Ax = b be satisfied when m ⇐= n, and/or when A is rank 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

9 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

where 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

10 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Case A: m > n, and r = n. 

Ax = b can only be satisfied if b ∈ R⊥(U2) = R(U1). 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 11 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Case B: m > n, and r = n. 

Ax = b can always be satisfied, but x is not unique. If xo is a solution to Ax = b, then xo + V2c2, for any c2 ∈ C<sup>n−r</sup> is also a solution. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 12 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Case C: r < min(m, n). 

Ax = b can only be satisfied if b ∈ R(U1). 

If xo is a solution to Ax = b, then xo + V2c2, for any c2 ∈ C<sup>n−r</sup> is also a solution. 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Theorem 4.2 Let UΣV<sup>H</sup> be the SVD of A. For k < r = rank(A), the solution to the problem 

is 

Moreover, the minimal objective function value is 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

14 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Theorem 4.3 Let UΣV<sup>H</sup> be the SVD of A. For k < r = rank(A), the solution to the problem 

is 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 15 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Recall the KL transform in Lecture 3. 

ˆ The vector xn, formed from truncating N ⇒ r KL coefficients, has the covariance matrix given by 

From Theorems 4.2 & 4.3 we know that Rxˆ is the closest rank-r matrix to the true signal covariance matrix Rx, in the 2-norm and Frobenius-norm senses. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

16 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

The idea: An arbitrary vector y can be expressed as 

where ys ∈S, & yc ∈S⊥. 

We are interested in obtaining a matrix P, called the orthogonal projection, such that 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

17 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Application: noise reduction 

Consider a received signal that consists of a signal vector s ∈S and noise w: 

We don’t know s, but we do know S. 

We can enhance the signal by performing a projection 

where ws = Pw is a residual noise vector. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

18 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

A matrix P ∈ C<sup>n×n</sup> is an orthogonal projection onto S if 

Note that a matrix having the property P<sup>2</sup> = P is called an idempotent matrix. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

19 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

We have learnt that for a subspace S with a dimension m, there is a full rank matrix X ∈ C<sup>n×m</sup> , such that S = R(X). An orthogonal projection onto S = R(X) is 

Exercise: Verify that (∗) satisfies the 3 properties for an orthogonal projection matrix. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

20 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Theorem 4.4 The orthogonal projection matrix in (∗) is unique (i.e., there does not exist P1 such that P1 is an orthogonal projection onto S and P1 = P). 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

21 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

The orthogonal complement projection: 

By observing that 

we obtain 

and that (I ⇒ P) is the orthogonal projection onto the orthogonal complement subspace S⊥. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

22 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Property 4.6 

- V1V1<sup>HistheorthogonalprojectionontoR(AH).</sup> 

• U1U<sup>H</sup> 1<sup>istheorthogonalprojectionontoR(A).</sup> 

• U2U<sup>H</sup> 2<sup>istheorthogonalprojectionontoR⊥(A).</sup> 

Property 4.7 The eigenvalues of a projection matrix is either 1 or 0. The number of nonzero eigenvalues is the dimension of the associated subspace. 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

23 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

## Distance between subspaces: 

Let S1 & S2 be two subspaces with dim S1 = dim S2. 

Let P1 & P2 be the orthogonal projection matrices of S1 & S2, respectively. 

The distance between S1 & S2 is defined as 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

24 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Theorem 4.5 Suppose 

are unitary, where W1, Z1 ∈ C<sup>n×k</sup> . If S1 = R(W1) & S2 = R(Z1), then 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 25 

COM521500 Math. Methods for Signal Processing I 

Lecture 4: SVD & Orthogonal Projection 

Property 4.8 0 ≤ dist(S1, S2) ≤ 1. 

Property 4.9 If S1 = S2, then dist(S1, S2) = 0. 

Property 4.10 If S1 ∪S2<sup>⊥= {0},thendist(S1, S2) = 1.</sup> 

Institute Comm. Eng. & Dept. Elect. Eng., National Tsing Hua University 

26

---

## 源文件

- [Singular Decomposition.pdf](attachments/documents/AI_Machine-Learning-in-Communication-78b10208f7b9/Singular Decomposition.pdf)
