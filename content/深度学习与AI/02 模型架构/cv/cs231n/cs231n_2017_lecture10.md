---
title: cs231n_2017_lecture10
source: converted:attachments/documents/AI_CNN-076525596876/cs231n_2017_lecture10.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-076525596876/cs231n_2017_lecture10.pdf
  title: cs231n_2017_lecture10.pdf
---

### Lecture 10: Recurrent Neural Networks 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 1 **May 4, 2017** 

###### Administrative 

A1 grades will go out soon A2 is due today (11:59pm) 

Midterm is in-class on Tuesday! We will send out details on where to go soon 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 10 - 2 May 4, 2017 

a 

###### Today: Recurrent Neural Networks 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 10 - 10 May 4, 2017 

: : : 

<mark>\</mark> 

<mark>\</mark> 

#### <mark>Recurrent Neural Network</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 18 **May 4, 2017** 

#### <mark>Recurrent Neural Network</mark> 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 19 **May 4, 2017** 

<mark>fu] . = fattites 1</mark> 

<mark>hi = fw(he-1, £4)</mark> 

: 

. <mark>hi = fw (he-1, T+)</mark> = | <mark>Yu = Whyht</mark> : <mark>hi = tanh(W);,hi_1 + Wnt)</mark> 

###### RNN: Computational Graph 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 23 **May 4, 2017** 

###### RNN: Computational Graph 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 24 **May 4, 2017** 

###### RNN: Computational Graph 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 25 **May 4, 2017** 

###### RNN: Computational Graph 

Re-use the same weight matrix at every time-step 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 26 **May 4, 2017** 

###### RNN: Computational Graph: Many to Many 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

###### RNN: Computational Graph: Many to One 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 30 **May 4, 2017** 

###### RNN: Computational Graph: One to Many 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

###### Sequence to Sequence: Many-to-one + one-to-many 

**Many to one** : Encode input sequence in a single vector 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 32 **May 4, 2017** 

###### Sequence to Sequence: Many-to-one + one-to-many 

**One to many** : Produce output sequence from single input vector 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 33 **May 4, 2017** 

<mark>hy = tanh(Whpht_1 + Went)</mark> 

Backpropagation through time 

Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 41 **May 4, 2017** 

###### **Truncated** Backpropagation through time 

Loss 

Run forward and backward through chunks of the sequence instead of whole sequence 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 42 **May 4, 2017** 

###### **Truncated** Backpropagation through time 

Loss Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 43 **May 4, 2017** 

###### **Truncated** Backpropagation through time 

Loss 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 

44 **May 4, 2017** 

###### by William Shakespeare 

From fairest creatures we desire increase, That thereby beauty's rose might never die, But as the riper should by time decease, His tender heir might bear his memory: But thou, contracted to thine own bright eyes, Feed'st thy light's flame with self-substantial fuel, Making a famine where abundance lies, Thyself thy foe, to thy sweet self too cruel: Thou that art now the world's fresh ornament, And only herald to the gaudy spring, Within thine own bud buriest thy content, And tender churl mak'st waste in niggarding: Pity the world, or else this glutton be, To eat the world's due, by the grave and thee. 

When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field, Thy youth's proud livery so gazed on now, Will be a tatter'd weed of small worth held: Then being asked, where all thy beauty lies, Where all the treasure of thy lusty days; To say, within thine own deep sunken eyes, Were an all-eating shame, and thriftless praise. How much more praise deserv'd thy beauty's use, If thou couldst answer 'This fair child of mine Shall sum my count, and make my old excuse,' Proving his beauty by succession thine! This were to be new made when thou art old, And see thy blood warm when thou feel'st it cold. 

~~——_—_$>~~ 

tyntd-iafhatawiaoihrdemot lytdws e ,tfti, astai f ogoh eoase rrranbyne 'nhthnee e plia tklrgd t o idoe ns,smtt’~ h ne etie h,hregtrs nigtike,aoaenns lng 

"Tmont thithey" fomesscerliund Keushey. Thom here sheulke, anmerenith ol sivh I lalterthend Bleipile shuwy fil on aseterlome coaniogennc Phe lism thond hon at. MeiDimorotion in ther thize.” 

Aftair fall unsuch that the hall for Prince Velzonski's that me of her hearly, and behs to so arwage fiving were to it beloge, pavu say falling misfort how, and Gogition is so overelical and ofter. 

"Why do what that day," replied Natasha, and wishing to himself the fact the princess, Princess Mary was easier, fed in had oftened him. Pierre aking his soul came to the packs and drove up his father-in-law women. 

###### PANDARUS: 

###### VIOLA: 

Alas, I think he shall be come approached and the day When little srain would be attain'd into being never fed, And who is but a chain and subjects of his death, I should not sleep. 

###### Second Senator: 

They are away this miseries, produced upon my soul, Breaking and strongly should be buried, when I perish The earth and thoughts of many states. 

###### DUKE VINCENTIO: 

Why, Salisbury must find his flesh and thought That which I am not aps, not a man and in fire, To show the reining of the raven and the wars To grace my hand reproach within, and not a fair are hand, That Caesar and my goodly father's world; 

When I was heaven of presence and our fleets, We spare with hours, but cut thy council I am great, Murdered and by thy master's ready there My power to give thee but so much as hell: Some service in the noble bondman here, Would show him to her wine. 

Well, your wit is in the care of side and that. 

###### KING LEAR: 

###### Second Lord: 

They would be ruled after this chamber, and my fair nues begun out of the fact, to be conveyed, Whose noble souls I'll have the heart of the wars. 

O, if you were a feeble sight, the courtesy of your law, Your sight and several breath, will wear the gods With his heads, and my hands are wonder'd at the deeds, So drop upon your lordship's head, and your opinion Shall be against your honour. 

###### Clown: 

Come, sir, I will make did behold your worship. 

###### VIOLA: 

I'll drink it. 

Proof. Omitted. fi) This since F € F and x € G the diagram Lemma 0.1. Let C be a set of the construction.‘ Ss ———> Let C be a gerber covering. Let F be a quasi-coherent sheaves of O-modules. We | have to show that € ———>O>~x: zi Oox = Ox(L) gor, | "hg Proof. This is an algebraic space with the composition of sheaves F on Xgyaie we have =a’ ————> where Ox(F) = {morph xo, (G,F)} | G defines an isomorphism F — ¥ of O-modules. a) me, om x Lemma 0.2. This is an integer Z is injective. | Proof. See Spaces, Lemma ??. Oo Spec(Ky) Morsets d(Oxx,,+9) e . —— 7 Lemma 0.3. Let S be a scheme. Let X be a scheme and X is an affine open istypea limit.fs. ThisThenis Gof finiteis a finitetype typediagrams,and assumeand S is a flat and F and G is a finite covering. Let U Cc &X be a canonical and locally of finite type. Let X be a scheme. the composition of G is a regular sequence, Let X be a scheme which is equal to the formal comple. ¢ Ox: is a sheaf of rings. Oo The following to the construction of the lemma follows. . a —- i Proof. We have see that X = Spec(R) and F is a finite type representable by Let X be a scheme. Let X be a scheme covering. Let algebraic space. The property F is a finite morphism of algebraic stacks. Then the : cohomology of X is an open neighbourhood of U. Oo b:X—>Y } y ¥ x} x Proof. This is clear that G is a finite presentation, see Lemmas ??. be a morphism of algebraic spaces over S andy. A reduced above we conclude that U is an open covering of C. The functor F is a . “field Proof. Let X, be a nonzero scheme of X., Let X, be an algebraic: space. Let F be a Ox, —? Fe -\UOXeeate) —> OX,Oxr(Ox,)<1 vw quasi-coherent sheaf of Ox-modules. The following are equivalent is an isomorphism of covering of Ox,. If F is the unique element of F such that X . ‘ ~ (1) F is an algebraic space over S. isThe an propertyisomorphism,F is a disjoint union of Proposition ?? and we can filtered set of (2) If X is an affine open covering. presentations of a scheme Oy-algebra with F are opens of finite type over S. If F is a scheme theoretic image points. oO ConsiderV sider aa c¢common structurestruc > on X andé Xc the» functor+ 5 Ox(U)(U whichrhich isis locallyn¢ r of If F is: a finiteBs direct: sum Ox, is: a closed .immersion,. see Lemma 2027. Th:This is'a finite type. O sequence of ¥ is a similar morphism. 

es) 

z karpathy +. {G~~ = 

\ 

This repasilory 

Explore Gist Blog Help 

torvalds / linux 

@Watch3,711 wStar 23,054 YFork 9,141 

###### Linux kernel source tree 

> 520,037 commits 1 branch 420 releases 5,039 contributors Code **L** L T - ‘ ru | P branch: masterlinux / + iz Seppeull requesisser 74 Merge branch ‘drm-tixes' of git//people.treedesktop.org/~airlied/finux «++ " torvaids authored 9 hours ago latest sit 4b1706927¢d “= ce firmware firrnwareihex2tw.c: restore m g deta awitch statement 2 monins ag You can clone with HTTPS fs read file_hand f SSH, or Subversion. @ 

static void do_command(struct seq file *m, void *v) 

{ int column = 32 << (cmd[2] & 0x80); if (state) cmd = (int)(int_state “~ (in_8(&ch->ch_flags) & Cmd) ? 2 : 1); else seq = 1; for (i = 0; i< 16: i++) { if (k & (1 << 1)) pipe = (in_use & UMXTHREAD UNCCA) + ((count & Ox00000000ffFFELLS) & Ox000000f) << 8; if (count == 0) sub(pid, ppc_md.kexec handle, 0x20000000); pipeset_bytes(i, 0); 

} subsystem info = &of changes[PAGE_ SIZE]; rek_controls(offset, idx, &soffset); control _ check_polarity(&context, val, 0); for (i = 0; i < COUNTER; i++) seq puts(s, “policy "); } 

static void stat_PC_SEC read mostly offsetof(struct seq_argsqueue, \ pC>[1}); 

static void os_prefix(unsigned long sys) { 

PUT_PARAM_RAID(2, sel) = get_state_state(); set_pid_sum((unsigned long)state, current_state_str(), (unsigned long)-1i->lr_full; low; 

} 

#### <mark>Searching for  interpretable cells</mark> 

Karpathy, Johnson, and Fei-Fei: Visualizing and Understanding Recurrent Networks, ICLR Workshop 2016 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 

56 **May 4, 2017** 

Cell sensitive to position in line: The sole importance of the crossing of the Berezina lies in the fact that it plainly and indubitably proved the fallacy of all the plans for cutting off the enemy's retreat and the soundness of the only possible line of action--the one Kutuzov and the general mass of the army demanded--namely, simply to follow the enemy up. The French crowd fled at a continually increasing speed and all its energy was directed to reaching its goal. It fled like a wounded animal and it was impossibLe to block itsS path. This was shown not so much by the arrangements it made for crossing as by what took place at the bridges. When the bridges broke down, unarmed soldiers, people from Moscow and women with children who were with the French transport, all--carried on by vis inertiae-pressed forward into boats and into the ice-covered water and did not) surrender, 

Cell that turns on inside comments and quotes: ~~2~~ u a —— TT c | 1 |/ Lsm. = k sf->lsm_str, GF PQKIERWEUDT d <mark>—</mark> ret; 

#ifdef CONFIG_AUDITSYSCALL static inline int audit match_class_bits(int class, wu32 “mask) { Lata ws if (classes[{class]) { for (i = @; i < AUDIT_BITMASK_SIZE; i++) if commen & classes[class][{i]) } —_ a3)| 

###### test image 

<u>This image</u> is CC0 public domain 

|<sup>image</sup> 

conv-128 conv-128 __maxpool_ conv-256 .<sup>conv-256</sup> conv-512 conv-512 maxpool conv-512 conv-512 maxpool FC-4096 _ FC-4096 FC-1000 __ softmax _ 

image ss 

conv-64 _ 

conv-128 conv-128 __maxpool_ conv-256 _<sup>conv-256</sup> conv-512 conv-512 maxpool conv-512 conv-512 maxpool FC-4096 _ FC-4096 FC-1000 softmax 

image 

conv-128 conv-128 __maxpool_ conv-256 .<sup>conv-256</sup> conv-512 conv-512 maxpool conv-512 conv-512 maxpool FC-4096 _ FC-4096 

image 

— 

image 

— 

image 

— 

image 

— 

~~<mark>ae</mark>~~ 

## <mark>Ep:</mark> ~~<mark>-</mark> |~~ 

~~<u><mark>can</mark></u>~~ <u><mark>e</mark></u> z= ) piv 

# ~~<u><mark>ae</mark></u>~~ 

# ~~<u><mark>ae</mark></u>~~ 

##### Image Captioning with Attention 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 83 **May 4, 2017** 

##### Image Captioning with Attention 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 84 **May 4, 2017** 

A woman is throwing a frisbee in a park. A dog is standing on a hardwood floor. —— —_— 

nN ii A little girl sitting on a bed with A group of people sitting on a boat a teddy bear. in the water. 

A stop sign is on a road with a mountain in the background. 

Q: What endangered animal Q: Where willthe drivergo |§Q: When was the picture is featured on the truck? if turning right? taken? 

- A: A bald eagle. 

A: Onto 24% Rd. 

A: During a wedding. 

A: A sparrow. 

- A: Onto 25 % Rd. 

A: During a bar mitzvah. 

A: Ahumming bird. 

- A: Onto 23 % Rd. 

A: During a funeral. A: During a Sunday church carvira 

- A: Araven. 

- A: Onto Main Street. 

Q: Who is under the umbrella? 

A: Two women. 

A: A child. 

A: Anold man. 

A: A husband and a wife. 

vary in cach time Step { comes fom Symoid, qarantied p multiply a tmbev between o ond 4. 

> <sup>cabcndateGrandiontofSomehiddenstate.</sup> gvdwantaso} : only one tanh if we want<sup>$9</sup> 

MUTI: 

i= O(WerXt + Whrrhi_i + b,-) 

we ht = tanh(Wrnrrt + Wan (re © he-1) So bn) he = 2% OMr-1+1—-%) Oh_ 

+0 owid vanishing gradiant problem 

z = sigm(W,.27; + 6) r = sigm(W,.2; + Wir: + 5) Aegis = tanh(Wia(r © hy) + tanh(z,)+ h,) © z MUT2: z = sigm(Wy2; + Whale + 5.) = Seales We +6) heya = tanh(Whal(r © he) + Weare + bn) © 2 + hO(1—2z) MUT3: z = sigm(W,,2; + Wi, tanh(h;) + 6,) r = sigm(W yr; + Warhs + 6) Aes, = tanh(Wi(r © he) + Wonry + d,) © 2 + hb@ (1-2) 

#### Summary 

- RNNs allow a lot of flexibility in architecture design 

- - Vanilla RNNs are simple but don’t work very well 

- Common to use LSTM or GRU: their additive interactions improve gradient flow 

- Backward flow of gradients in RNN can explode or vanish. Exploding is controlled with gradient clipping. Vanishing is controlled with additive interactions (LSTM) 

- Better/simpler architectures are a hot topic of current research 

- - Better understanding (both theoretical and empirical) is needed. 

**Fei-Fei Li & Justin Johnson & Serena Yeung** 

**Lecture 10 -** 104 **May 4, 2017** 

###### Next time: Midterm! Then Detection and Segmentation 

Fei-Fei Li & Justin Johnson & Serena Yeung 

Lecture 10 - 105 May 4, 2017

---

## 源文件

- [cs231n_2017_lecture10.pdf](attachments/documents/AI_CNN-076525596876/cs231n_2017_lecture10.pdf)
