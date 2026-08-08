---
title: 02 MDP
source: converted:attachments/documents/AI_Reinforcement-Learning-fb784b0b934a/02
  MDP.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Reinforcement-Learning-fb784b0b934a/02 MDP.pdf
  title: 02 MDP.pdf
---

# 02 MDP

Lecture 2: Markov Decision Processes 

Lecture 2: Markov Decision Processes 

David Silver 

Lecture 2: Markov Decision Processes 

<mark>1</mark> Markov Processes 

<mark>2</mark> Markov Reward Processes 

3 Markov Decision Processes 

<mark>4</mark> Extensions to MDPs 

Lecture 2: Markov Decision Processes 

Markov Processes Introduction 

# Introduction to MDPs 

_Markov decision processes_ formally describe an environment for reinforcement learning 

Where the environment is _fully observable_ 

i.e. The current _state_ completely characterises the process Almost all RL problems can be formalised as MDPs, e.g. Optimal control primarily deals with continuous MDPs Partially observable problems can be converted into MDPs Bandits are MDPs with one state 

Lecture 2: Markov Decision Processes 

Markov Processes 

Markov Property 

# Markov Property 

“The future is independent of the past given the present” 

<mark>Defnition</mark> 

A state _St_ is _Markov_ if and only if 

P [ _St_ +1 _| St_ ] = P [ _St_ +1 _| S_ 1 _, ..., St_ ] 

The state captures all relevant information from the history Once the state is known, the history may be thrown away i.e. The state is a statistic of the future 

Lecture 2: Markov Decision Processes 

Markov Processes 

Markov Property 

# State Transition Matrix 

For a Markov state _s_ and successor state _s_<sup>_0_</sup> , the _state transition probability_ is defined by 

State transition matrix _P_ defines transition probabilities from all states _s_ to all successor states _s_<sup>_0_</sup> , 

where each row of the matrix sums to 1. 

Lecture 2: Markov Decision Processes 

Markov Processes Markov Chains 

# Markov Process 

A Markov process is a memoryless random process, i.e. a sequence of random states _S_ 1 _, S_ 2 _, ..._ with the Markov property. 

<mark>Defnition</mark> 

A _Markov Process_ (or _Markov Chain_ ) is a tuple _hS, Pi_ 

_S_ is a (finite) set of states 

_P_ is a state transition probability matrix, _Pss 0_ = P [ _St_ +1 = _s_<sup>_0_</sup> _| St_ = _s_ ] 

Lecture 2: Markov Decision Processes 

Markov Processes 

Markov Chains 

# Example: Student Markov Chain 

Lecture 2: Markov Decision Processes 

Markov Processes 

Markov Chains 

# Example: Student Markov Chain Episodes 

Sample episodes for Student Markov Chain starting from _S_ 1 = C1 

_S_ 1 _, S_ 2 _, ..., ST_ 

C1 C2 C3 Pass Sleep C1 FB FB C1 C2 Sleep C1 C2 C3 Pub C2 C3 Pass Sleep C1 FB FB C1 C2 C3 Pub C1 FB FB FB C1 C2 C3 Pub C2 Sleep 

Lecture 2: Markov Decision Processes 

Markov Processes Markov Chains 

# Example: Student Markov Chain Transition Matrix 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

MRP 

# Markov Reward Process 

A Markov reward process is a Markov chain with values. 

<mark>Defnition</mark> 

A _Markov Reward Process_ is a tuple _hS, P, R, γi_ 

_S_ is a set of states 

_P_ is a state transition probability matrix, _Pss 0_ = P [ _St_ +1 = _s_<sup>_0_</sup> _| St_ = _s_ ] 

_R_ is a reward function, _Rs_ = E [ _Rt_ +1 _| St_ = _s_ ] _γ_ is a discount factor, _γ 2_ [0 _,_ 1] 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

MRP 

# Example: Student MRP 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Return 

# Return 

## <mark>Defnition</mark> 

The _return Gt_ is the total discounted reward from time-step _t_ . 

The _discount γ 2_ [0 _,_ 1] is the present value of future rewards The value of receiving reward _R_ after _k_ + 1 time-steps is _γ_<sup>_k_</sup> _R_ . This values immediate reward above delayed reward. 

_γ_ close to 0 leads to ”myopic” evaluation _γ_ close to 1 leads to ”far-sighted” evaluation 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Return 

# Why discount? 

Most Markov reward and decision processes are discounted. Why? Mathematically convenient to discount rewards Avoids infinite returns in cyclic Markov processes Uncertainty about the future may not be fully represented If the reward is financial, immediate rewards may earn more interest than delayed rewards Animal/human behaviour shows preference for immediate reward 

It is sometimes possible to use _undiscounted_ Markov reward processes (i.e. _γ_ = 1), e.g. if all sequences terminate. 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Value Function 

# Value Function 

The value function _v_ ( _s_ ) gives the long-term value of state _s_ 

<mark>Defnition</mark> 

The _state value function v_ ( _s_ ) of an MRP is the expected return starting from state _s_ 

_v_ ( _s_ ) = E [ _Gt | St_ = _s_ ] 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Value Function 

# Example: Student MRP Returns 

Sample returns for Student MRP: Starting from _S_ 1 = C1 with _γ_ =<sup><u>1</u></sup> 2 

Lecture 2: Markov Decision Processes 

Markov Reward Processes Value Function 

# Example: State-Value Function for Student MRP (1) 

Lecture 2: Markov Decision Processes 

Markov Reward Processes Value Function 

# Example: State-Value Function for Student MRP (2) 

Lecture 2: Markov Decision Processes 

Markov Reward Processes Value Function 

# Example: State-Value Function for Student MRP (3) 

Lecture 2: Markov Decision Processes 

Markov Reward Processes Bellman Equation 

# Bellman Equation for MRPs 

The value function can be decomposed into two parts: 

immediate reward _Rt_ +1 discounted value of successor state _γv_ ( _St_ +1) 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Bellman Equation 

# Bellman Equation for MRPs (2) 

Lecture 2: Markov Decision Processes 

Markov Reward Processes Bellman Equation 

# Example: Bellman Equation for Student MRP 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Bellman Equation 

# Bellman Equation in Matrix Form 

The Bellman equation can be expressed concisely using matrices, 

_v_ = _R_ + _γPv_ 

where _v_ is a column vector with one entry per state 

Lecture 2: Markov Decision Processes 

Markov Reward Processes 

Bellman Equation 

# Solving the Bellman Equation 

The Bellman equation is a linear equation It can be solved directly: 

Computational complexity is _O_ ( _n_<sup>3</sup> ) for _n_ states Direct solution only possible for small MRPs There are many iterative methods for large MRPs, e.g. 

Dynamic programming Monte-Carlo evaluation Temporal-Di↵erence learning 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

MDP 

# Markov Decision Process 

A Markov decision process (MDP) is a Markov reward process with decisions. It is an _environment_ in which all states are Markov. <mark>Defnition</mark> 

A _Markov Decision Process_ is a tuple _hS, A, P, R, γi_ 

_S_ is a set of states 

_A_ is a set of actions 

_P_ is a state transition probability matrix, _Pss_<sup>_a0_= P [</sup><sup>_St_+1=</sup><sup>_s0|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> _R_ is a reward function, _R_<sup>_a_</sup> _s_<sup>= E [</sup><sup>_Rt_+1</sup><sup>_|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> _γ_ is a discount factor _γ 2_ [0 _,_ 1]. 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

MDP 

# Example: Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Policies 

# Policies (1) 

## <mark>Defnition</mark> 

A _policy ⇡_ is a distribution over actions given states, 

_⇡_ ( _a|s_ ) = P [ _At_ = _a | St_ = _s_ ] 

A policy fully defines the behaviour of an agent MDP policies depend on the current state (not the history) i.e. Policies are _stationary_ (time-independent), _At ⇠ ⇡_ ( _·|St_ ) _, 8t >_ 0 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Policies 

# Policies (2) 

Given an MDP _M_ = _hS, A, P, R, γi_ and a policy _⇡_ The state sequence _S_ 1 _, S_ 2 _, ..._ is a Markov process _hS, P_<sup>_⇡_</sup> _i_ The state and reward sequence _S_ 1 _, R_ 2 _, S_ 2 _, ..._ is a Markov reward process _hS, P_<sup>_⇡_</sup> _, R_<sup>_⇡_</sup> _, γi_ where 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Value Functions 

# Value Function 

## <mark>Defnition</mark> 

The _state-value function v⇡_ ( _s_ ) of an MDP is the expected return starting from state _s_ , and then following policy _⇡_ 

_v⇡_ ( _s_ ) = E _⇡_ [ _Gt | St_ = _s_ ] 

## <mark>Defnition</mark> 

The _action-value function q⇡_ ( _s, a_ ) is the expected return starting from state _s_ , taking action _a_ , and then following policy _⇡_ 

_q⇡_ ( _s, a_ ) = E _⇡_ [ _Gt | St_ = _s, At_ = _a_ ] 

Lecture 2: Markov Decision Processes 

Markov Decision Processes Value Functions 

# Example: State-Value Function for Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation 

The state-value function can again be decomposed into immediate reward plus discounted value of successor state, 

The action-value function can similarly be decomposed, 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation for _V_<sup>_⇡_</sup> 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation for _Q_<sup>_⇡_</sup> 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation for _v⇡_ (2) 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation for _q⇡_ (2) 

_q⇡_ ( _s, a_ ) = _R_<sup>_a_</sup> _s_<sup>+</sup><sup>_γ_</sup> X _Pss_<sup>_a0_</sup> X _⇡_ ( _a_<sup>_0_</sup> _|s_<sup>_0_</sup> ) _q⇡_ ( _s_<sup>_0_</sup> _, a_<sup>_0_</sup> ) _s_<sup>_0_</sup> _2S a_<sup>_0_</sup> _2A_ 

Lecture 2: Markov Decision Processes 

Markov Decision Processes Bellman Expectation Equation 

# Example: Bellman Expectation Equation in Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Expectation Equation 

# Bellman Expectation Equation (Matrix Form) 

The Bellman expectation equation can be expressed concisely using the induced MRP, 

with direct solution 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Optimal Value Functions 

# Optimal Value Function 

## <mark>Defnition</mark> 

The _optimal state-value function v⇤_ ( _s_ ) is the maximum value function over all policies 

The _optimal action-value function q⇤_ ( _s, a_ ) is the maximum action-value function over all policies 

The optimal value function specifies the best possible performance in the MDP. 

An MDP is “solved” when we know the optimal value fn. 

Lecture 2: Markov Decision Processes 

Markov Decision Processes Optimal Value Functions 

# Example: Optimal Value Function for Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes Optimal Value Functions 

# Example: Optimal Action-Value Function for Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Optimal Value Functions 

# Optimal Policy 

Define a partial ordering over policies 

_⇡ ≥ ⇡_<sup>_0_</sup> if _v⇡_ ( _s_ ) _≥ v⇡0_ ( _s_ ) _, 8s_ 

<mark>Theorem</mark> 

_For any Markov Decision Process_ 

_There exists an optimal policy ⇡⇤ that is better than or equal to all other policies, ⇡⇤ ≥ ⇡, 8⇡ All optimal policies achieve the optimal value function, v⇡⇤_ ( _s_ ) = _v⇤_ ( _s_ ) _All optimal policies achieve the optimal action-value function, q⇡⇤_ ( _s, a_ ) = _q⇤_ ( _s, a_ ) 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Optimal Value Functions 

# Finding an Optimal Policy 

An optimal policy can be found by maximising over _q⇤_ ( _s, a_ ), 1 if _a_ = argmax _q⇤_ ( _s, a_ ) _⇡⇤_ ( _a|s_ ) = _a2A_ 0 _otherwise_ ( 

There is always a deterministic optimal policy for any MDP If we know _q⇤_ ( _s, a_ ), we immediately have the optimal policy 

Lecture 2: Markov Decision Processes 

Markov Decision Processes Optimal Value Functions 

# Example: Optimal Policy for Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Bellman Optimality Equation for _v⇤_ 

The optimal value functions are recursively related by the Bellman optimality equations: 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Bellman Optimality Equation for _Q_<sup>_⇤_</sup> 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Bellman Optimality Equation for _V_<sup>_⇤_</sup> (2) 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Bellman Optimality Equation for _Q_<sup>_⇤_</sup> (2) 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Example: Bellman Optimality Equation in Student MDP 

Lecture 2: Markov Decision Processes 

Markov Decision Processes 

Bellman Optimality Equation 

# Solving the Bellman Optimality Equation 

Bellman Optimality Equation is non-linear No closed form solution (in general) Many iterative solution methods 

Value Iteration Policy Iteration Q-learning Sarsa 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

# Extensions to MDPs 

(no exam) 

Infinite and continuous MDPs Partially observable MDPs Undiscounted, average reward MDPs 

Lecture 2: Markov Decision Processes 

Extensions to MDPs MDPs 

# MDPs 

(no exam) 

The following extensions are all possible: 

Countably infinite state and/or action spaces Straightforward 

Continuous state and/or action spaces Closed form for linear quadratic model (LQR) Continuous time 

Requires partial di↵erential equations Hamilton-Jacobi-Bellman (HJB) equation Limiting case of Bellman equation as time-step _!_ 0 

Lecture 2: Markov Decision Processes 

Partially Observable MDPs 

Extensions to MDPs 

# POMDPs 

(no exam) 

A Partially Observable Markov Decision Process is an MDP with hidden states. It is a hidden Markov model with actions. <mark>Defnition</mark> 

A _POMDP_ is a tuple _hS, A, O, P, R, Z, γi_ 

_S_ is a set of states 

_A_ is a set of actions 

_O_ is a set of observations 

_P_ is a state transition probability matrix, _Pss_<sup>_a0_= P [</sup><sup>_St_+1=</sup><sup>_s0|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> _R_ is a reward function, _R_<sup>_a_</sup> _s_<sup>= E [</sup><sup>_Rt_+1</sup><sup>_|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> _Z_ is an observation function, _Zs_<sup>_a0_</sup> _o_<sup>= P [</sup><sup>_Ot_+1=</sup><sup>_o|St_+1=</sup><sup>_s0, At_=</sup><sup>_a_]</sup> 

_γ_ is a discount factor _γ 2_ [0 _,_ 1]. 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Partially Observable MDPs 

# Belief States 

(no exam) 

## <mark>Defnition</mark> 

A _history Ht_ is a sequence of actions, observations and rewards, 

_Ht_ = _A_ 0 _, O_ 1 _, R_ 1 _, ..., At−_ 1 _, Ot, Rt_ 

## <mark>Defnition</mark> 

A _belief state b_ ( _h_ ) is a probability distribution over states, conditioned on the history _h_ 

_b_ ( _h_ ) = (P ⇥ _St_ = _s_<sup>1</sup> _| Ht_ = _h_ ⇤ _, ...,_ P [ _St_ = _s_<sup>_n_</sup> _| Ht_ = _h_ ]) 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Partially Observable MDPs 

# Reductions of POMDPs 

(no exam) 

The history _Ht_ satisfies the Markov property 

The belief state _b_ ( _Ht_ ) satisfies the Markov property 

A POMDP can be reduced to an (infinite) history tree A POMDP can be reduced to an (infinite) belief state tree 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Average Reward MDPs 

# Ergodic Markov Process 

(no exam) 

An ergodic Markov process is 

_Recurrent_ : each state is visited an infinite number of times _Aperiodic_ : each state is visited without any systematic period 

<mark>Theorem</mark> 

_An ergodic Markov process has a limiting stationary distribution d_<sup>_⇡_</sup> ( _s_ ) _with the property_ 

_d_<sup>_⇡_</sup> ( _s_ ) = X _d_<sup>_⇡_</sup> ( _s_<sup>_0_</sup> ) _Ps 0s s_<sup>_0_</sup> _2S_ 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Average Reward MDPs 

# Ergodic MDP 

(no exam) 

## <mark>Defnition</mark> 

An MDP is ergodic if the Markov chain induced by any policy is ergodic. 

For any policy _⇡_ , an ergodic MDP has an _average reward per time-step ⇢_<sup>_⇡_</sup> that is independent of start state. 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Average Reward MDPs 

# Average Reward Value Function 

(no exam) 

The value function of an undiscounted, ergodic MDP can be expressed in terms of average reward. 

_v_ ˜ _⇡_ ( _s_ ) is the extra reward due to starting from state _s_ , 

There is a corresponding average reward Bellman equation, 

Lecture 2: Markov Decision Processes 

Extensions to MDPs 

Average Reward MDPs 

# Questions? 

_The only stupid question is the one you were afraid to ask but never did._ 

- _-Rich Sutton_

---

## 源文件

- [02 MDP.pdf](attachments/documents/AI_Reinforcement-Learning-fb784b0b934a/02 MDP.pdf)
