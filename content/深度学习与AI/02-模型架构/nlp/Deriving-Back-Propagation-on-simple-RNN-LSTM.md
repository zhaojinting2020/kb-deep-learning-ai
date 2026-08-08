---
title: Deriving Back Propagation on simple RNN LSTM
source: converted:attachments/documents/AI_CNN-200fec6cfec1/Deriving Back Propagation
  on simple RNN LSTM.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/AI_CNN-200fec6cfec1/Deriving Back Propagation on simple
    RNN LSTM.pdf
  title: Deriving Back Propagation on simple RNN LSTM.pdf
---

# Deriving Back Propagation on simple RNN LSTM

<!-- Start of picture text -->
><br><!-- End of picture text -->

<!-- Start of picture text -->
. =< wp =<br><!-- End of picture text -->

Blue Box — error rate for Cost Function when time stamp is 1 Green Box — error rate for Cost Function when time stamp is 2 

Hi Jae, Great article. 

Iama bit confused about your calculations at time t=1. I thought it should only include the blue portion. However at time t=2 then it should add both t=1 and t=2 derivations. 

Also the way you calculate the partial derivatives is different than 

Especially the way the author calculates the gradient of the state at time t. 

dou = Ay + Aout; dstate, = Sout, © o, © (1 — tanh?(state,)) + dstater.1 © fia da = dstatee OU © (1 — a?) df, = dstate, © state, © f, © (1 — fr) dot = dout: © tanh(state;) © o © (1 — o%) 62, = W'? - dgates, Aout;—1 = ul. dgates: 

His calculation includes future state gradients as well as forget gates (at time t =t+1). 

Who is correct? 

Thanks, 

Abe 

between my post and Aidan’s Post. 

Finally all of our labs at my school are getting renovated hence I do not have any whiteboard to write on so I’ll try my best to write neatly as possible on my note pad. 

_. . ._ 

Chain Rule: Example Ex. y= wl? a= T7x8 43x" dx du dx 2 45 fee 7. = 5 (7s +3x° | (56x! +6x) = (140x7 + 15x) (7x' +3x7)"" 

<!-- Start of picture text -->
z=in(2x'+4y")<br>&mga<br>ax g(x.y) ax (2x7 +4y"} (2x7 +4y"]<br>@ 1 dt gy 4<br>a gl(xy) & (2x7 +4y") (x7 +2")<br><!-- End of picture text -->

<!-- Start of picture text -->
(Inx)'=—-<br>x<br>ATE.<br><!-- End of picture text -->

<!-- Start of picture text -->
Ex:<br>Des<br>O83 | on<br>(0)(sr)“ (65)Os2<br>|<br>Ly ts 3<br><!-- End of picture text -->

<!-- Start of picture text -->
SSS Ee —— U<br>A Mi if: o<br>=a<br><!-- End of picture text -->

<!-- Start of picture text -->
oa. : ae a etal “<br>=—_ = —* - ee Beit soon es<br><!-- End of picture text -->

<!-- Start of picture text -->
Input activation:<br>aq = tanh(W, «zy + Uy + out, + ba)<br>Input gate:<br>it = (Wite + Uj: outy-1 + bi)<br>Forget gate:<br>fe = o(Wy + xy + Us + outy_1 + 65)<br>Output gate:<br>4 = o(W, - ry + Us - outy_1 + be)<br>Which leads to:<br>Internal state:<br>statey = a,Oi,+ fy © statey_y<br>Output:<br>out, = tanh(stater) © ot<br><!-- End of picture text -->

<!-- Start of picture text -->
7 ba a 7 = Ef oe on |<br>Age, ili eo , . oe<br>StokJ Fele<br>A _ ahi gal ,<br><!-- End of picture text -->

<!-- Start of picture text -->
RET= aaauv Wi Ai? Ut Usa@n See eesa<br>f3 TERETE: ten aa<br><!-- End of picture text -->

<!-- Start of picture text -->
Forward @)ie2<br>[orasrd<br>2 0.81775: @<br>0 fc.Om oai7s7<br>CH) Ovem-C-@— [0.5363]<br>sarees (+)Hi Label: 0.5<br>0)<br>day = tanh(W,y-19-+ Ug -out_y + by) = tan [0.45 0.25] ( + [0.15] [0] + [0.2]) = 0.81775<br>ig = o(1V) «29 + U;- outs +b) = o([0.95 0:8] (I + [0.8] [0] + [0.65]) ~ 0.96083<br>fo = 0(W, + 29+ Us out +by) = o(|0.7 0.45] [| + [0.1] [0] + [0.15}) = 0.85195<br>09 = o(W +9 + Uy + out_1 + bp) = 0( [0.6 0.4] [| + [0.25] [0] + [0.1]) = 0.81757<br>statey = ay © ig + fo ® state-) = 0.81775 x 0.96+ 08 . 51953 x 0 = 0.78572<br>outy = tanh(statey) © op = tanh(0.78572) x 0.81757 = 0.53631<br><!-- End of picture text -->

<!-- Start of picture text -->
Forward @ t=1<br>{ist 76)<br>Jesse 6)<br>os Go omen<br>0.53631  &YHHO—-C-4¢- 07197]<br>a Label:. 1.25<br>(0.78572)<br>4 = tanh (Wy «271 +Uq - outo+ by) = tanh ((0.45 0.25] (* + [0.15] [0.53631 ] + (0.2]) = 0.81980<br>i = o(W;j +11 +0; -onty +6) = 0([0.95 0.3] (*] + [0.8] [0.53631] + [0.65]) = 0.98118<br>fi =o(Wy <1 +Uy -outg +by) = 0((0.7 0.45] ('] + [0.1] (0.53631) + (0.15]) = 0.87030<br>01 =o (Wo: 21 + Up outo + bo) = o([0.6 0.4) [’; | + (0.25) [0.53631] + (0.1]) = 0.81993<br>state) = a, © iy + fi © stateo = 0.84980 x 0.98118 + 0.87080 x 0.78572 = 1.5176<br>out; = tanh(statey) ©, = tanh(1.5176) x 0.84993 = 0.77197<br><!-- End of picture text -->

Sout, = Ay + Aoute state, = Sout, © o, © (1 — tanh? (state,)) + Sstaters; © fisr day = dstater © ix © (1 — a?) bi, = Sstatey © a © iy © (1 — ip) of, = dstate, © state_, © fr (1 — fi) bo, = Sout, © tanh(stater) © op © (1 — o) 6x = wl. dgates, Aout); = UT - dgates: 

<!-- Start of picture text -->
“a = | 4<br><!-- End of picture text -->

Internal state: 

state, = a OU + f, © statey_y Output: out, = tanh( state; ) ©) OF 

<!-- Start of picture text -->
eee ae ae, of<br><!-- End of picture text -->

<!-- Start of picture text -->
aE<br>Sa<br><!-- End of picture text -->

doutk = At + Aout; dstate, = dout, ©) OF © (1 ~ tanh? (state;)) + dstate,.4 © fis da, = dstatey OO (1 — az) 

i, = dstatey © a,© © (1 — x) Of, = dstate, © statey_; © fr © (1 — fi) do, = Sout, © tanh( state.) © of © (1 — or) éz,=W! - dgates; Aoutt-1 = U" - gates: 

<!-- Start of picture text -->
Seay<br>hd<br>=m<br><!-- End of picture text -->

douts = A; + Aout; dstate, = Sout, © o, © (1 — tanh? (state,)) + Sstateys; © frsy da = dstate, © iy © (1 — a?) di; = dstate, Oa O% © —%&) Of; = dstate; © stater_; © fy @ (1 — fi) éz,=W? . dgates; Aout; = UT - égates; 

<!-- Start of picture text -->
SYS<br>iyJe bs sedine<br>=<br>et,<br>ae<br>ae s*<br>%<br>way<br><!-- End of picture text -->

<!-- Start of picture text -->
veROCAReeM a5aRE MRED LS DOASMTRE RNATR SNe Nae ele CED eeBd RRA Ma OTLE NESE AN<br>— —_— ook Propagationee Ressect.ne toeeWoo CTS) ee Es<br>| db tat /d We =[4& avo] +|dé,/dVe |<br>4 |<br><!-- End of picture text -->

<!-- Start of picture text -->
”<br>1)+ AJ : Ton States): Or’ (1-0,<br>anh( state:) - O,- (l- 0+)<br><!-- End of picture text -->

Input activation: 

a, = tanh(W, «2; + U, + out, + bg) Input gate: 

ip = o(Wy- ae + Uj outk-1 + Bj) Forget gate: 

fr = o(Wy + x + Us - outy_1 + by) Output gate: 

## Final words 

Hope this post can clear some confusion, however I know that my explanation in English isn’t the best. So if you have any questions please comment down below. 

If any errors are found, please email me at jae.duk.seo@gmail.com, if you wish to see the list of all of my writing please view my website here. 

Meanwhile follow me on my twitter here, and visit my website, or my Youtube channel for more content. 

## Reference 

1. Backpropogating an LSTM: A Numerical Example — Aidan Gomez — Medium. (2016). Medium. Retrieved 1 May 2018, from 

https://medium.com/@aidangomez/let-s-do-thisf9b699de31d9 

2. Calculus Review: Derivative Rules — Magoosh High School Blog. (2017). Magoosh High School Blog. Retrieved 1 May 2018, from 

https://magoosh.com/hs/ap-calculus/2017/calculus- 

## review-derivative-rules/ 

3. Rules of calculus — multivariate. (2018). Columbia.edu. Retrieved 1 May 2018, from http://www.columbia.edu/itc/sipa/math/calc_rules_ multivar.html 

4. The derivative of lnx and examples — MathBootCamps. (2016). MathBootCamps. Retrieved 1 May 2018, from 

https://www.mathbootcamps.com/derivative-naturallog-lnx/ 

5. Electricity price forecasting with Recurrent Neural Networks. (2018). Slideshare.net. Retrieved 1 May 2018, from 

https://www.slideshare.net/TaegyunJeon1/electricity -price-forecasting-with-recurrent-neural-networks 

6. DiLerentiation 3 Basic Rules of DiLerentiation The Product and Quotient Rules The Chain Rule Marginal Functions in Economics Higher-Order Derivatives. — ppt download. (2018). Slideplayer.com. Retrieved 1 May 2018, from 

http://slideplayer.com/slide/10776187/ 

7. Backpropogating an LSTM: A Numerical Example — Aidan Gomez — Medium. (2016). Medium. Retrieved 2 May 2018, from 

https://medium.com/@aidangomez/let-s-do-thisf9b699de31d9 

8. Only Numpy: Deriving Forward feed and Back Propagation in Long Short Term Memory (LSTM) part 1. (2018). Becoming Human: ArtiWcial Intelligence Magazine. Retrieved 2 May 2018, from 

https://becominghuman.ai/only-numpy-derivingforward-feed-and-back-propagation-in-long-shortterm-memory-lstm-part-1-4ee82c14a652 

Machine Learning Neural Networks Mathematics Derivatives ArtiCcial Intelligence 

# Discover Medium 

Welcome to a place where words matter. On Medium, smart voices and original ideas take center stage - with no ads in sight. Watch 

# Make Medium yours 

Follow all the topics you care about, and we’ll deliver the best stories for you to your homepage and inbox. Explore 

# Become a member 

Get unlimited access to the best stories on Medium — and support writers while you’re at it. Just $5/month. Upgrade 

About Help Legal

---

## 源文件

- [Deriving Back Propagation on simple RNN LSTM.pdf](attachments/documents/AI_CNN-200fec6cfec1/Deriving Back Propagation on simple RNN LSTM.pdf)
