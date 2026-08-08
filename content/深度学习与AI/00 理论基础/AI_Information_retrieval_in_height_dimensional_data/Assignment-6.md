---
title: Assignment 6
source: converted:attachments/documents/AI_Information-retrieval-in-height-dimensional-data-d7c64bc0e092/Assignment
  6.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Information-retrieval-in-height-dimensional-data-d7c64bc0e092/Assignment
    6.pdf
  title: Assignment 6.pdf
---

# Assignment 6

TECHNISCHE UNIVERSITAT<sup>¨</sup> MUNCHEN<sup>¨</sup> Fakult¨at f¨ur Elektrotechnik und Informationstechnik Lehrstuhl f¨ur Datenverarbeitung PD Martin Kleinsteuber 

`Information Retrieval in High Dimensional Data Lab #9, Theoretical Exercises` , 24.01.2019 

# **Convex Optimization** 

Task 1. Consider the optimization problem 

minimize _x_<sup>2</sup> + 1 

subject to ( _x −_ 2)( _x −_ 4) __ 0 _._ 

- a) Provide the feasible set, the optimal value, and the optimal solution. 

**Solution** : Feasible set: _S_ = [2 _,_ 4], solution: _x_<sup>_⇤_</sup> = 2, because due to monotonicity, the solution has to be the left limit of the feasible set, optimal value: _p_<sup>_⇤_</sup> = 5. 

- b) Rewrite the problem as a set of KKT conditions. 

**Solution** : The Lagrange function is given by 

The KKT conditions are 

- c) State the dual problem. Find the dual optimal value and dual optimal solution. Does strong duality hold? 

**Solution** : The Lagrange function is a squared function and thus its infimum can be easily determined by setting its derivative to 0: 

The _Lagrange dual function_ is given by 

The dual problem is thus given by 

Di↵erentiating with respect to _λ_ yields 

which, together with the constraint _λ ≥_ 0, yields the two candidates _λ_ 1 = 0 _, λ_ 2 = 2. Furthermore, _LD_ (2) _≥ LD_ (0), so we conclude _λ_<sup>_⇤_</sup> = 2 and _LD_ ( _λ_<sup>_⇤_</sup> ) = 5 = _p_<sup>_⇤_</sup> . Thus, strong duality holds. 

- d) Determine the solution of the original problem by substituting the dual solution into the KKT conditions. 

## **Solution** : We get 

These conditions are satisfied by _x_<sup>_⇤_</sup> = 2. Note: This works, because strong duality holds. Otherwise, the dual solution would not lead to a unique point that the KKT conditions.

---

## 源文件

- [Assignment 6.pdf](attachments/documents/AI_Information-retrieval-in-height-dimensional-data-d7c64bc0e092/Assignment 6.pdf)
