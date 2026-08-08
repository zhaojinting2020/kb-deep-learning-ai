---
title: 01 intro_RL
source: converted:attachments/documents/AI_Reinforcement-Learning-5d710f1181cc/01
  intro_RL.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_Reinforcement-Learning-5d710f1181cc/01 intro_RL.pdf
  title: 01 intro_RL.pdf
---

# 01 intro_RL

Lecture 1: Introduction to Reinforcement Learning 

Lecture 1: Introduction to Reinforcement Learning 

David Silver 

Lecture 1: Introduction to Reinforcement Learning 

## Outline 

<mark>1</mark> Admin 

<mark>2</mark> About Reinforcement Learning 

<mark>3</mark> The Reinforcement Learning Problem 

- <mark>4</mark> Inside An RL Agent 

- <mark>5</mark> Problems within Reinforcement Learning 

Lecture 1: Introduction to Reinforcement Learning 

Admin 

## Class Information 

Thursdays 9:30 to 11:00am Website: http://www.cs.ucl.ac.uk/sta↵/D.Silver/web/Teaching.html Group: http://groups.google.com/group/csml-advanced-topics Contact me: d.silver@cs.ucl.ac.uk 

Lecture 1: Introduction to Reinforcement Learning 

Admin 

## Assessment 

Assessment will be 50% coursework, 50% exam Coursework 

Assignment A: RL problem Assignment B: Kernels problem Assessment = _max_ ( _assignment_ 1 _, assignment_ 2) 

#### Examination 

A: 3 RL questions B: 3 kernels questions Answer any 3 questions 

Lecture 1: Introduction to Reinforcement Learning 

Admin 

## Textbooks 

An Introduction to Reinforcement Learning, Sutton and Barto, 1998 

MIT Press, 1998 

_⇠_ 40 pounds 

Available free online! _⇠_ http://webdocs.cs.ualberta.ca/ sutton/book/the-book.html Algorithms for Reinforcement Learning, Szepesvari 

Morgan and Claypool, 2010 _⇠_ 20 pounds Available free online! 

_⇠_ http://www.ualberta.ca/ szepesva/papers/RLAlgsInMDPs.pdf 

Lecture 1: Introduction to Reinforcement Learning 

About RL 

## Branches of Machine Learning 

Lecture 1: Introduction to Reinforcement Learning 

About RL 

## Characteristics of Reinforcement Learning 

What makes reinforcement learning di↵erent from other machine learning paradigms? 

There is no supervisor, only a _reward_ signal Feedback is delayed, not instantaneous Time really matters (sequential, non i.i.d data) Agent’s actions a↵ect the subsequent data it receives 

Lecture 1: Introduction to Reinforcement Learning 

About RL 

## Examples of Reinforcement Learning 

Fly stunt manoeuvres in a helicopter Defeat the world champion at Backgammon Manage an investment portfolio Control a power station Make a humanoid robot walk Play many di↵erent Atari games better than humans 

Lecture 1: Introduction to Reinforcement Learning 

About RL 

## Helicopter Manoeuvres 

Lecture 1: Introduction to Reinforcement Learning 

## Outline 

<mark>1</mark> Admin 

<mark>2</mark> About Reinforcement Learning 

<mark>3</mark> The Reinforcement Learning Problem 

- <mark>4</mark> Inside An RL Agent 

- <mark>5</mark> Problems within Reinforcement Learning 

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem 

Reward 

## Rewards 

A reward _Rt_ is a scalar feedback signal Indicates how well agent is doing at step _t_ The agent’s job is to maximise cumulative reward 

Reinforcement learning is based on the reward hypothesis Definition (Reward Hypothesis) 

_All_ goals can be described by the maximisation of expected cumulative reward 

Do you agree with this statement? 

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem Reward 

## Examples of Rewards 

Fly stunt manoeuvres in a helicopter 

+ve reward for following desired trajectory 

_−_ ve reward for crashing 

Defeat the world champion at Backgammon 

+ _/−_ ve reward for winning/losing a game Manage an investment portfolio 

+ve reward for each $ in bank 

Control a power station 

+ve reward for producing power 

- ve reward for exceeding safety thresholds 

Make a humanoid robot walk 

+ve reward for forward motion 

- ve reward for falling over 

Play many di↵erent Atari games better than humans 

+ _/−_ ve reward for increasing/decreasing score 

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem Reward 

## Sequential Decision Making 

Goal: _select actions to maximise total future reward_ Actions may have long term consequences Reward may be delayed 

It may be better to sacrifice immediate reward to gain more long-term reward Examples: 

A financial investment (may take months to mature) Refuelling a helicopter (might prevent a crash in several hours) Blocking opponent moves (might help winning chances many moves from now) 

<mark>L</mark> 

<mark>L</mark> 

|i) >.<br><I<br>LY|* Ajewt infbaences the eywivonmentt ony |<br>5<br>.<br>2|
|---|---|
|Cnwtvove|have olefferont goals atsometime?<br>=havetoconpareandwegletthesediflerentGoods|

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem State 

## History and State 

The history is the sequence of observations, actions, rewards 

i.e. all observable variables up to time _t_ i.e. the sensorimotor stream of a robot or embodied agent What happens next depends on the history: 

The agent selects actions The environment selects observations/rewards 

State is the information used to determine what happens next Formally, state is a function of the history: 

<mark>L</mark> 

<mark>L</mark> 

<mark>L</mark> 

<mark>L</mark> 

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem 

State 

## Information State 

An information state (a.k.a. Markov state) contains all useful information from the history. 

<mark>Defnition</mark> 

A state _St_ is Markov if and only if 

P[ _St_ +1 _| St_ ] = P[ _St_ +1 _| S_ 1 _, ..., St_ ] 

“The future is independent of the past given the present” 

Once the state is known, the history may be thrown away i.e. The state is a sufficient statistic of the future The environment state _St_<sup>_e_isMarkov</sup> The history _Ht_ is Markov 

<mark>L</mark> 

<mark>L</mark> 

<mark>L</mark> 

Lecture 1: Introduction to Reinforcement Learning 

The RL Problem 

State 

## Partially Observable Environments 

Partial observability: agent indirectly observes environment: A robot with camera vision isn’t told its absolute location A trading agent only observes current prices A poker playing agent only observes public cards 

Now agent state = environment state 

Formally this is a partially observable Markov decision process (POMDP) 

Agent must construct its own state representation _St_<sup>_a_,e.g.</sup> Complete history: _St_<sup>_a_=</sup><sup>_Ht_</sup> Beliefs of environment state: _St_<sup>_a_= (P[</sup><sup>_S_</sup> _t_<sup>_e_=</sup><sup>_s_1]</sup><sup>_, ...,_P[</sup><sup>_S_</sup> _t_<sup>_e_=</sup><sup>_sn_])</sup> Recurrent neural network: _St_<sup>_a_=</sup><sup>_σ_(</sup><sup>_S_</sup> _t_<sup>_a_</sup> _−_ 1<sup>_Ws_+</sup><sup>_OtWo_)</sup> 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Major Components of an RL Agent 

An RL agent may include one or more of these components: Policy: agent’s behaviour function Value function: how good is each state and/or action Model: agent’s representation of the environment 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Policy 

A policy is the agent’s behaviour It is a map from state to action, e.g. Deterministic policy: _a_ = _⇡_ ( _s_ ) Stochastic policy: _⇡_ ( _a|s_ ) = P[ _At_ = _a|St_ = _s_ ] 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Value Function 

Value function is a prediction of future reward Used to evaluate the goodness/badness of states And therefore to select between actions, e.g. 

_v⇡_ ( _s_ ) = E _⇡_ ⇥ _Rt_ +1 + _γRt_ +2 + _γ_<sup>2</sup> _Rt_ +3 + _... | St_ = _s_ ⇤ 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Example: Value Function in Atari 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Model 

A model predicts what the environment will do next _P_ predicts the next state _R_ predicts the next (immediate) reward, e.g. 

_Pss_<sup>_a0_= P[</sup><sup>_St_+1=</sup><sup>_s0|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> _R_<sup>_a_</sup> _s_<sup>= E [</sup><sup>_Rt_+1</sup><sup>_|St_=</sup><sup>_s, At_=</sup><sup>_a_]</sup> 

# ~~<mark>a</mark>~~ 

<mark>a a a a</mark> 

<mark>a</mark> 

| 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Categorizing RL agents (1) 

#### Value Based 

No Policy (Implicit) Value Function 

#### Policy Based 

Policy No Value Function 

Actor Critic 

Policy Value Function 

Lecture 1: Introduction to Reinforcement Learning 

Inside An RL Agent 

## Categorizing RL agents (2) 

#### Model Free 

Policy and/or Value Function No Model 

#### Model Based 

Policy and/or Value Function Model 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Learning and Planning 

Two fundamental problems in sequential decision making Reinforcement Learning: 

The environment is initially unknown 

The agent interacts with the environment 

The agent improves its policy 

#### Planning: 

A model of the environment is known 

The agent performs computations with its model (without any external interaction) 

The agent improves its policy 

a.k.a. deliberation, reasoning, introspection, pondering, thought, search 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Exploration and Exploitation (1) 

Reinforcement learning is like trial-and-error learning The agent should discover a good policy From its experiences of the environment Without losing too much reward along the way 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Exploration and Exploitation (2) 

_Exploration_ finds more information about the environment _Exploitation_ exploits known information to maximise reward It is usually important to explore as well as exploit 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Examples 

Restaurant Selection 

Exploitation Go to your favourite restaurant Exploration Try a new restaurant 

Online Banner Advertisements 

Exploitation Show the most successful advert Exploration Show a di↵erent advert Oil Drilling 

Exploitation Drill at the best known location Exploration Drill at a new location Game Playing 

Exploitation Play the move you believe is best Exploration Play an experimental move 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Prediction and Control 

Prediction: evaluate the future Given a policy Control: optimise the future Find the best policy 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Gridworld Example: Prediction 

### (a) 

(b) 

What is the value function for the uniform random policy? 

Lecture 1: Introduction to Reinforcement Learning 

Problems within RL 

## Gridworld Example: Control 

What is the optimal value function over all possible policies? What is the optimal policy? 

Lecture 1: Introduction to Reinforcement Learning 

Course Outline 

## Course Outline 

Part I: Elementary Reinforcement Learning 

<mark>1</mark> Introduction to RL 

- <mark>2</mark> Markov Decision Processes 

<mark>3</mark> Planning by Dynamic Programming 

<mark>4</mark> Model-Free Prediction 

<mark>5</mark> Model-Free Control 

Part II: Reinforcement Learning in Practice 

<mark>1</mark> Value Function Approximation 

- <mark>2</mark> Policy Gradient Methods 

<mark>3</mark> Integrating Learning and Planning 

- <mark>4</mark> Exploration and Exploitation 

<mark>5</mark> Case study - RL in games

---

## 源文件

- [01 intro_RL.pdf](attachments/documents/AI_Reinforcement-Learning-5d710f1181cc/01 intro_RL.pdf)
