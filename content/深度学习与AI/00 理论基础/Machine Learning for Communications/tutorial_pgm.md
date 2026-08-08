---
title: tutorial_pgm
source: converted:attachments/documents/AI_Machine-Learning-in-Communication-3c60a4e40e83/tutorial_pgm.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Machine-Learning-in-Communication-3c60a4e40e83/tutorial_pgm.pdf
  title: tutorial_pgm.pdf
---

# tutorial_pgm

_Exercises: Probabilistic Graphical Models_ 

# **Factor graphs, Sum-product rules** 

Consider the follow factor graph: 

1. Write down the overall function _f_ ( _x_ 1 _, x_ 2 _, . . . , x_ 11). 

2. What is the marginalization _f_ ( _x_ 1) using the sum-product rules? 

3. How many operations (summations) does a naive calculation of _f_ ( _x_ 1) require? Assume that all _xi, i_ = 1 _, . . . ,_ 11 are binary. How many summations are needed with the sum-product algorithm and the result of subtask 2 (without further sophisticated simplifications)? 

# **Factor graphs, Sum-product rules: Solution** 

1. We have 

2. We have 

We need to find the expression for _mx_ 2 _!f_ 1( _x_ 2) first: 

1 

_Exercises: Probabilistic Graphical Models_ 

We now address _mx_ 3 _!f_ 1(˜ _x_ 3): 

Putting everything together, we obtain: 

3. If we perform a naive marginalization of _f_ ( _x_ 1 _, x_ 2 _, . . . , x_ 11) without exploiting any information about the factorization, we have a summation over 2<sup>10</sup> summands and perform 2<sup>10</sup> _−_ 1 additions. Using the sum-product algorithm and the previous result, we have: 

   - 2<sup>2</sup> = 4 values of ( _x_ 2 _, x_ 3) and for each pair we need 

- 2<sup>2</sup> _−_ 1) = 3 additions to add the above four terms together. 

Therefore, we have a total number of 4 _·_ (7 + 3 + 1 + 3) + 3 = 59 additions. 

2 

_Exercises: Probabilistic Graphical Models_ 

# **Markov chain** 

1. Consider the joint probability mass function (PMF) 

which represents a so called Markov chain. Draw the corresponding factor graph having _x_ 3 as the root. 

2. Calculate the marginalization _f_ ( _x_ 3) = _PX_ 3( _x_ 3) using the sum-product rules. 

3. Assume now that you are given the following numerical values: 

# **Markov chain: Solution** 

1. The factor graph of the Markov chain looks as follows: We have the correspondence: 

2. Use the graph from the previous sub task, we have 

So let’s calculate each factor one after the other: 

Regarding the second factor, we have: 

3 

_Exercises: Probabilistic Graphical Models_ 

where 

and 

Putting all together, we arrive at 

3. We have 

4 

_Exercises: Probabilistic Graphical Models_ 

# **Exam Task WS17/18: Solution** 

1. The random variable _X_ 1 _X_ 2 _X_ 3 _X_ 4 can take on values in the set 

The RV _X_ 3 is a deterministic function of the _X_ 1 and _X_ 2, i.e., the realizations of _X_ 1 and _X_ 2 determine _X_ 3. The same holds true for _X_ 4 if _X_ 1 **xor** _X_ 2 = 0. 

2. The distribution _PX_ <u>(</u> _<u>x</u>_ <u>)</u> factorizes as 

where 

Therefore, we have: 

3. We use the definition of a conditional distribution and Bayes’ rule: 

We have for _PY_ ( _<u>y</u>_ <u>):</u> 

Exemplarily, for _PX_ 1 _<u>Y</u>_ (0 _, y_ <u>),</u> we obtain 

6 

_Exercises: Probabilistic Graphical Models_ 

## 4. The model function is 

where 

5. The factor graph looks like: 

6. We note that 

7 

_Exercises: Probabilistic Graphical Models_ 

We have for the variable to factor node messages: 

We have for the factor node messages to the variable node messages (we limit the detailed derivations to messages involving _x_ 1): 

8 

_Exercises: Probabilistic Graphical Models_ 

Summary: 

After variable node update of _x_ 1 _, x_ 2 _, x_ 3 _, x_ 4 we have _mx_ 1 _!f_ 5( _x_ 1) = _mx_ 1 _!f_ 6( _x_ 1) = _mf_ 1 _!x_ 1( _x_ 1) = [0 _._ 2 _,_ 0 _._ 8] _mx_ 2 _!f_ 5( _x_ 2) = _mx_ 2 _!f_ 6( _x_ 2) = _mf_ 2 _!x_ 2( _x_ 2) = [0 _._ 9 _,_ 0 _._ 1] _mx_ 3 _!f_ 5( _x_ 3) = _mf_ 3 _!x_ 3( _x_ 3) = [0 _._ 3 _,_ 0 _._ 7] _mx_ 4 _!f_ 6( _x_ 4) = _mf_ 4 _!x_ 4( _x_ 4) = [0 _._ 7 _,_ 0 _._ 3] After factor node update of _f_ 5: _mf_ 5 _!x_ 1( _x_ 1) _⇡_ [0 _._ 3269 _,_ 0 _._ 6731] _mf_ 5 _!x_ 2( _x_ 2) _⇡_ [0 _._ 4697 _,_ 0 _._ 5303] _mf_ 5 _!x_ 3( _x_ 3) = [0 _._ 18 _,_ 0 _._ 82] After factor node update of _f_ 6: _mf_ 6 _!x_ 1( _x_ 1) _⇡_ [0 _._ 5667 _,_ 0 _._ 4333] _mf_ 6 _!x_ 2( _x_ 2) = [0 _._ 45 _,_ 0 _._ 55] _mf_ 6 _!x_ 4( _x_ 4) = [0 _._ 63 _,_ 0 _._ 37] Compute _PPPP_ ˜˜˜˜ _XXXX_ 1234 _||||YYYY_ (((( _xxxx_ 1234 _||||yyyy_ the)))) = = = =(approximate) _m m m mffff_ 1234 _!!!!xxxx_ 1234(((( _xxxx_ 1234)))) _mmmmffff_ marginals:5556 _!!!!xxxx_ 1234(((( _xxxx_ 1234)))) = [0 _⇡mmff_ 66[0 _!!.._ 08607990 _xx_ 12(( _xx_ 12 _,,_ )) 0 0 _⇡ ⇡.._ 9140]2010][0[0 _.._ 13708671 _,,_ 0 0 _.._ 8630]1329] 

9

---

## 源文件

- [tutorial_pgm.pdf](attachments/documents/AI_Machine-Learning-in-Communication-3c60a4e40e83/tutorial_pgm.pdf)
