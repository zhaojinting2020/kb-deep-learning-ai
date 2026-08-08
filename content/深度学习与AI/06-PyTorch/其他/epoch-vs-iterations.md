---
title: epoch vs iterations
source: converted:attachments/documents/Prog_Python-Neural-Network-299f193ddcfb/epoch
  vs iterations.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/Prog_Python-Neural-Network-299f193ddcfb/epoch vs iterations.pdf
  title: epoch vs iterations.pdf
---

## Page 1

![page 1](attachments/documents/Prog_Python-Neural-Network-299f193ddcfb/pages/page_001.png)

18:41
Ss stackoverflow

Home

PUBLIC

@ Stack Overflow
Tags
Users

Jobs

TEAMS       What's this?

_i Free 30 Day Trial
日

@ stackoverflow.com

Products       Customers       Use cases

12 Answers                                         Active | Oldest | Votes
In the neural network terminology:

563 one epoch = one forward pass and one backward pass of

all the training examples

¢ batch size = the number of training examples in one
forward/backward pass. The higher the batch size, the
more memory space you'll need.

number of iterations = number of passes, each pass
using [batch size] number of examples. To be clear, one
pass = one forward pass + one backward pass (we do not
count the forward pass and backward pass as two
different passes).

Example: if you have 1000 training examples, and your batch
size is 500, then it will take 2 iterations to complete 1 epoch.

FYI: Tradeoff batch size vs. number of iterations to train a
neural network

The term "batch" is ambiguous: some people use it to
designate the entire training set, and some people use it to
refer to the number of training examples in one
forward/backward pass (as | did in this answer). To avoid that
ambiguity and make clear that batch corresponds to the
number of training examples in one forward/backward pass,
‘one can use the term mini-batch.

share improve this answer             edited May 27 '17 at 19:06

follow

& Triage needs to be fixed urgently, and
users need to be notified upon...

区

Upcoming Feature: New Question Close
Experience

2 Dark Mode Beta - help us root out low-
contrast and un-converted bits

if)

OVHcloud

Preisgiinstige Hochleistungsserver

Jetzt ab €52.24 inkl mwst. / Monat

QAware GmbH

Softwareingenieur (miwid)
Munchen, Deutschland

java typescript

Softwareingenieur (miw/d)

---

## 源文件

- [epoch vs iterations.pdf](attachments/documents/Prog_Python-Neural-Network-299f193ddcfb/epoch vs iterations.pdf)
