---
title: Old and New Matrix Algebra Useful for Statistics
source: converted:attachments/documents/root-5adb07d37ef4/Old and New Matrix Algebra
  Useful for Statistics.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/root-5adb07d37ef4/Old and New Matrix Algebra Useful
    for Statistics.pdf
  title: Old and New Matrix Algebra Useful for Statistics.pdf
---

# Old and New Matrix Algebra Useful for Statistics Thomas P. Minka 

December 28, 2000 

# Contents 

- 1 Derivatives 1 2 Kronecker product and vec 6 3 Vec-transpose 7 4 Multilinear forms 8 5 Hadamard product and diag 10 6 Inverting partitioned matrices 12 7 Polar decomposition 14 8 Hessians 15 

Warning: This paper contains a large number of matrix identities which cannot be absorbed by mere reading. The reader is encouraged to take time and check each equation by hand and work out the examples. This is advanced material; see Searle (1982) for basic results. 

# 1 Derivatives 

Maximum-likelihood problems almost always require derivatives. There are six kinds of derivatives that can be expressed as matrices: 

||S|calar|Ve|ctor|M|atrix|
|---|---|---|---|---|---|---|
|Scalar||dy<br>dx|dy<br>dx <sup>=</sup>|�∂yi<br>∂x<br>�|dY<br>dx <sup>=</sup>|�<br>∂yij<br>∂x<br>�|
|Vector|dy<br>dx|<sup>=</sup><br>�<br>∂y<br>∂xj<br>�|dy<br>dx <sup>=</sup>|�<br>∂yi<br>∂xj<br>�|||
|Matrix|dy<br>dX|<sup>=</sup><br>�<br>∂y<br>∂xji<br>�|||||

The partials with respect to the numerator are laid out according to the shape of Y while the partials with respect to the denominator are laid out according to the transpose of X. For example, dy/dx is a column vector while dy/dx is a row vector (assuming x and y are column vectors—otherwise it is flipped). Each of these derivatives can be tediously computed via partials, but this section shows how they instead can be computed with matrix manipulations. The material is based on Magnus and Neudecker (1988). 

Define the differential dy(x) to be that part of y(x +dx) − y(x) which is linear in dx. Unlike the classical definition in terms of limits, this definition applies even when x or y are not scalars. 

1 

For example, this equation: 

is well-defined for any y satisfying certain continuity properties. The matrix A is the derivative, as you can check by setting all but one component of dx to zero and making it small. The matrix A is also called the Jacobian matrix Jx→y. Its transpose is the gradient of y, denoted ∇y. The Jacobian is useful in calculus while the gradient is useful in optimization. 

Therefore, the derivative of any expression involving matrices can be computed in two steps: 

1. compute the differential 

2. massage the result into canonical form 

after which the derivative is immediately read off as the coefficient of dx, dx, or dX. 

The differential of an expression can be computed by iteratively applying the following rules: 

where<sup>⋆</sup> is any operator that rearranges elements, e.g. transpose, vec, and vec-transpose (section 3). The rules can be iteratively applied because of the chain rule, e.g. d(AX + Y) = d(AX) + dY = AdX + (dA)X + dY = AdX + dY. Most of these rules can be derived by subtracting F(X + dX) − F(X) and taking the linear part. For example, 

from which (6) follows. 

To derive dX<sup>−1</sup> , note that 

2 

from which (9) follows. 

The next step is to massage the differential into one of the six canonical forms (assuming x and y are column vectors): 

This is where the operators and identities developed in the following sections are useful. For example, since the derivative of Y with respect to X cannot be represented by a matrix, it is customary to use dvec(Y)/dvec(X) instead (vec is defined in section 2). If the purpose of differentiation is to equate the derivative to zero, then this transformation doesn’t affect the result. So after expanding the differential, just take vec of both sides and use the identities in sections 2 and 3 to get it into canonical form. 

One particularly helpful identity is: 

Examples: 

Constraints Sometimes we want to take the derivative of a function whose argument must be symmetric. In this case, dX must be symmetric, so we get 

where A ◦ I is simply A with off-diagonal elements set to zero. The reader can check this by expanding tr(AdX) and merging identical elements of dX. An example of this rule is: 

when Σ must be symmetric. This is usually easier than taking an unconstrained derivative and then using Lagrange multipliers to enforce symmetry. 

Similarly, if X must be diagonal, then so must dX, and we get 

Example: Principal Component Analysis Suppose we want to represent the zero-mean random vector x as one random variable a times a constant unit vector v. This is useful for compression or noise removal. Once we choose v, the optimal choice for a is v<sup>′</sup> x, but what is the best v? In other words, what v minimizes E[(x − av)<sup>′</sup> (x − av)], when a is chosen optimally for each x? 

Let Σ = E[xx<sup>′</sup> ]. We want to maximize 

4 

where λ is a Lagrange multiplier. Taking derivatives gives 

so the gradient is zero at any eigenvector of Σ. (Recall that the gradient is the transpose of the derivative.) If v is an eigenvector then f (v) = λ so the maximum is attained when v has the largest eigenvalue. 

Example: Blind source separation Suppose we have k microphones listening to k overlapped sound sources. Can we recover the individual sources? More generally, suppose we’ve observed data x generated by the function x = A<sup>−1</sup> u where u is a set of independent hidden causes and A is an unknown square mixing matrix. Assume each ui is distributed according to some density fi(wi) with unknown parameter wi. We want to find the mixing matrix which maximizes the likelihood of the data, so that we can then recover the hidden causes. 

These equations can be used in a gradient-based optimization to find A and w. This approach comes from Pearlmutter and Parra (1996). 

Example: Gaussian covariance Suppose we’ve observed vectors xi independently sampled from a zero-mean Gaussian distribution, i.e. 

We want to determine the most likely covariance matrix Σ, keeping in mind that the solution must be symmetric. Maximizing the log-likelihood gives: 

5 

# 2 Kronecker and vec product 

The Kronecker product (Lancaster and Tismenetsky, 1985) (Horn and Johnson, 1991) is 

which, like ordinary matrix product, is associative and distributive but not commutative. 

which implies (A ⊗ B)<sup>−1</sup> = A<sup>−1</sup> ⊗ B<sup>−1</sup> . 

If A and B are square, then the eigenvectors and eigenvalues of (A ⊗ B) are given by 

which implies 

Define vec(A) to be the stacked columns of A: 

Then the main result is 

Example The Lyapunov equation is 

which can be solved for vec(X). 

The other properties of vec will be presented in the context of vec-transpose. 

6 

# 3 Vec-transpose 

Vec-transpose is a new operator that generalizes vec and transpose. It is essential for expressing derivatives of Kronecker products and is also useful for expressing multilinear forms. It was called “vector transposition” by Marimont and Wandell (1992). 

and similarly A<sup>(p)</sup> for any integer p dividing rows(A). The basic properties are: 

We can freely apply vec-transpose inside of a trace expression: 

assuming conformability. This generalizes tr(A<sup>′</sup> B) = vec(A)<sup>′</sup> vec(B) as well as tr(A<sup>′</sup> B) = tr(AB<sup>′</sup> ). In fact, 

for any operator<sup>⋆</sup> that rearranges elements. 

7 

We can generalize equations 33 and 40: 

We now have the tools to express the derivative of a Kronecker product: 

where p is uniquely defined by conformability to be cols(B). 

Equation 51 gives us the following rule for pulling a matrix out of nested vec-transposes: 

This formula is useful in fitting multilinear forms (see the next section). Unlike regular transpose, it is not true in general that (B<sup>(p)</sup> C)(p) = C(p)B, as we can see by setting A = I in (53). 

# 4 Multilinear forms 

Multilinear statistical models are more expressive than linear models yet still easy to use. A multilinear form f (x, y, ..., z) is linear in each component separately, i.e. 

For example, face images can be modeled as linear in identity and linear in lighting conditions, yielding a bilinear model (Tenenbaum and Freeman, 1997). Another example is colored objects under colored light. 

Just as every bilinear form can be written as x<sup>′</sup> Gy = (y<sup>′</sup> ⊗ x<sup>′</sup> )vec(G), every multilinear form can be written as (z<sup>′</sup> ⊗ ... ⊗ y<sup>′</sup> ⊗ ... ⊗ x<sup>′</sup> )vec(G) (Magnus and Neudecker, 1988) (Prasolov, 1991) (Dodson and Poston, 1991). G is the tensor defining the form. 

For example, the trilinear form 

8 

Thus we can express a multilinear form either with tensor products or with vec-transpose. Matlab users have often used this kind of reshaping to manipulate higher-dimensional objects. 

which is a linear combination of matrices. 

Therefore, we can think of G in the trilinear form as a three-dimensional stack of matrices. The formula x<sup>′</sup> (Gz)<sup>(p)</sup> y says to first combine the stack according to z, then combine the columns according to y, and finally to combine the elements according to x. 

The vector-valued bilinear form is: 

which is the same as the scalar-valued trilinear form, except that the three-dimensional tensor G is only being summed in two dimensions. This form was used in Marimont and Wandell (1992) and subsequently by Tenenbaum and Freeman (1997). To fit this model to data, note that by (53) we have 

so given an observation and the value of y we can solve for x and vice-versa, by applying vectranspose to the observation. Therefore we can iterate from an initial guess until we reach a fixed point. This method generalizes to any multilinear form. Compare this to the complex algorithm without vec-transpose given in Magnus and Neudecker (1988). 

What if G must be learned as well as x and y? In this case, we need an entire observation matrix D = (GY)<sup>(p)</sup> X. Without loss of generality, we can assume that X and Y are orthogonal matrices, since G can always be chosen to make this so. (This can be proven with a polar decomposition of X and Y.) Therefore, if we know Y, we can solve for X by singular-value decomposition of D, and vice-versa with D<sup>(p)</sup> . Once X and Y have settled, it is easy to solve for vec(G). This algorithm comes from Marimont and Wandell (1992). 

9 

# 5 Hadamard product and diag 

The Hadamard product is simply the product of corresponding elements: 

Schur’s product theorem (Horn and Johnson, 1991) says 

This is also true for Kronecker product (by (35)), but not for regular matrix product. To prove it for Hadamard product, define a random vector z = x ◦ y, where x and y are independent random vectors with covariance Σx and Σy. Then the covariance of z can be shown to be Σx ◦Σy. Since every covariance matrix is nonnegative definite, the theorem follows. 

diag<sup>−1</sup> is a kind of pseudoinverse because 

but 

only for diagonal D. 

The basic properties are: 

Equation 66 can be used to remove all Hadamard products from an expression. 

10 

Another way to remove Hadamard products is with the diag<sup>−1</sup> matrix, which is the unique matrix Rn satisfying 

where A is n by n. Rn is an n × n<sup>2</sup> matrix with orthogonal rows (each row picks out one element of A). Some useful properties are: 

These properties cause Hadamard product and diag to have a kind of duality with Kronecker product and vec, as seen in the following table: 

A useful special case of (71) is 

If we factor A = VΛV<sup>−1</sup> then by (71), 

which relates the diagonal of a matrix to its eigenvalues. Many facts about the matrix V ◦ V<sup>−T</sup> can be found in Horn and Johnson (1991). 

Many identities for diag<sup>−1</sup> (A) also apply to tr(A), because 

where 1 is a column vector of ones. For example, tr(Adiag(x)C) = 1<sup>′</sup> (C<sup>′</sup> ◦A)x = diag<sup>−1</sup> (CA)x. Similarly to (48) we have 

which allows us to compute 

cf (52). 

11 

# 6 Inverting partitioned matrices 

A B If we partition P into then the Schur complement of A in P (Prasolov, 1991) is C D � � (P|A) = D − CA<sup>−1</sup> B (78) 

Similarly, 

Then the main result is 

This formula still holds if all inverses are replaced by pseudo-inverses. The reader may want to check this formula when P is a 2 × 2 matrix. 

Since PP<sup>−1</sup> = I and P<sup>−1</sup> P = I, we know 

These identities define (B|P) and (C|P) when B and C are singular. Since (A|P)(D − CA<sup>−1</sup> B) = I, we know 

which are handy for rewriting (83) solely in terms of (D|P) or (A|P). 

The Schur complement has the flavor of a division. Equation 84, for example, tells us that (B|P)(P|A) = A<sup>−1</sup> B which is a kind of cancellation of P. The clearest example of this is the formula for the determinant of P: 

- Pt | | ~~H~~ b VL 

| 

# 7 Polar decomposition 

Suppose a set of points B has been subjected to an unknown rotation and then jittered by white Gaussian noise to give a new set of points A. What is the most likely rotation? More generally, what unitary matrix minimizes f (U) = tr((A − UB)<sup>′</sup> (A − UB))? 

Expanding f (U) gives 

so the problem reduces to maximizing tr(A<sup>′</sup> UB) = tr(UBA<sup>′</sup> ). Define unitary V and W and positive diagonal S so that BA<sup>′</sup> = VSW<sup>′</sup> . This is the singular value decomposition of BA<sup>′</sup> . Then 

def tr(UBA<sup>′</sup> ) = tr(UVSW<sup>′</sup> ) = tr(W<sup>′</sup> UVS) = tr(XS) 

where X is also unitary. Since S is diagonal, 

which is maximum when xii = 1; that is, X = I. Therefore U = WV<sup>′</sup> is the desired solution. 

If B = I, this solution minimizes (A − U), i.e. it is the closest unitary matrix to an arbitrary matrix A. This U has the property that there exists a positive definite P such that A = PU. These two matrices are called the polar decomposition of A: U is the rotation and P is the magnitude, exactly analogous to the decomposition of a complex number. 

Scott and Longuet-Higgins (1991) used this technique to match columns in A with those in B. Matching assumes that U is a permutation matrix, but finding U in this case is difficult. So they first found the optimal unitary matrix and obtained a permutation from it. 

14 

# 8 Hessians 

The Hessian matrix is a matrix of second derivatives. The Hessian of a scalar function with respect to a vector argument is 

This section is based on Magnus and Neudecker (1988). 

The Hessian is the derivative of the first derivative. The first derivative a(x) = � ∂x∂yj � is a row vector function of x, and the derivative of this function with respect to x<sup>′</sup> is a matrix 

The Hessian can also be defined by the Taylor expansion of y: 

The matrix H is the Hessian, as you can check by setting all but two components of dx to zero. The Hessian can be computed in three steps: 

1. Compute the first differential 

2. Compute the differential of the first differential 

3. Massage the result into canonical form 

The only new differential rule we need is: 

because dx is not a function of x. 

The second has three canonical forms: 

where H must be symmetric. 

Some of these forms require rewriting the differential in terms of dvec(X), which can be tricky. Equation 48 is particularly helpful for introducing vec into an expression. To eliminate terms 

15 

like vec(X<sup>′</sup> ), Magnus and Neudecker (1988) define the commutation matrix Knm, which is the permutation matrix satisfying 

where X is n by m. Also, 

Examples: 

Constraints Sometimes we want to compute the Hessian of a function whose argument must be symmetric. In this case, the conversion from canonical form is 

where Knn is the commutation matrix discussed earlier and Dn is defined below. Since ∂xij and ∂xji are identical, this formula adds together ∂x∂y<sup>2</sup> ij<sup>,</sup> ∂xij∂∂xy ji<sup>,</sup> ∂xji∂∂xy ij<sup>,and</sup> ∂x∂y<sup>2</sup> ji<sup>,wheni=j.To</sup> derive it, we make dX symmetric by substituting dX + dX<sup>′</sup> − (dX ◦ I) (cf (21)) and get 

by (99) and (66). 

However, we may not want the full Hessian, but only the submatrix corresponding to unique elements of X. That is, we want dvech(Xd)dvech(<sup>2</sup> <u>y</u> X)<sup>′,wherevech(X)(Searle,1982)isvec(X)with</sup> elements above the diagonal deleted. For example, 

To convert between vec(X) and vech(X), Magnus and Neudecker (1988) define the duplication matrix Dn, which is the permutation matrix satisfying 

where X is n by n. This leads to the rule 

Furthermore, it can be shown that the matrix In2 + Knn − diag(vec(In)) above is equal to DnD<sup>′</sup> n<sup>.</sup> 

17 

Example: Hessian of a Gaussian likelihood Let l(m, V) be the logarithm of a Gaussian likelihood at x: 

By (107), the first term has Hessian 

where the symmetry of V has been invoked. 

The of the second term is 

and the second is 

So the full Hessian is 

And the reduced Hessian involving unique elements of V is 

18 

# Acknowledgements 

Tony Jebara and Martin Szummer helped clarify the presentation. 

# References 

- [1] C. T. J. Dodson and T. Poston. Tensor geometry: the geometric viewpoint and its uses. Springer-Verlag, 1991. 

- [2] Roger A. Horn and Charles R. Johnson. Topics in Matrix Analysis. Cambridge University Press, 1991. 

- [3] Harold Jeffreys. Theory of Probability. Clarendon Press, Oxford, third edition, 1961. 

- [4] Peter Lancaster and Miron Tismenetsky. The theory of matrices. Academic Press, 1985. 

- [5] Jan R. Magnus and Heinz Neudecker. Matrix differential calculus with applications in statistics and econometrics. John Wiley & Sons, 1988. 

- [6] David H. Marimont and Brian A. Wandell. Linear models of surface and illuminant spectra. Journal of the Optical Society of America, 9(11):1905–1913, November 1992. 

- [7] Barak A. Pearlmutter and Lucas C. Parra. Maximum likelihood blind source separation: A context-sensitive generalization of ICA. In Michael C. Mozer, Michael Jordan, and Thomas Petsche, editors, Advances in Neural Information Processing Systems 9, Cambridge, MA, 1997. MIT Press. http://www.cs.unm.edu/~bap/publications.html. 

- [8] Viktor V. Prasolov. Problems and Theorems in Linear Algebra. American Mathematical Society, 1991. 

- [9] G. Scott and H. Longuet-Higgins. An algorithm for associating the features of two images. Proc. Royal Society of London, B(244):21–26, 1991. 

- [10] Shayle R. Searle. Matrix Algebra Useful for Statistics. John Wiley & Sons, New York, NY, 1982. 

- [11] Joshua B. Tenenbaum and William T. Freeman. Separating style and content. In Michael C. Mozer, Michael Jordan, and Thomas Petsche, editors, Advances in Neural Information Processing Systems 9, Cambridge, MA, 1997. MIT Press. 

19

---

## 源文件

- [Old and New Matrix Algebra Useful for Statistics.pdf](attachments/documents/root-5adb07d37ef4/Old and New Matrix Algebra Useful for Statistics.pdf)
