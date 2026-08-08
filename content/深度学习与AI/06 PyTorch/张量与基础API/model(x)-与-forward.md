---
title: model(x) 与 forward
url: https://stackoverflow.com/questions/68645889/pytorch-nn-module-inference
fetch_source: stackexchange:api
fetched_at: '2026-06-28T08:17:02+00:00'
polished_at: '2026-06-27T19:21:52+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# model(x) 与 forward

## Question

I am planning on learning Pytorch. However at this stage I would like to ask a question so that I can understand some code I am reading
When you have a class whose base class is `nn.Module` say
class My_model(nn.Module)
```
how are inferences supposed to be run there?
In the code I am reading it says
tasks_output, other = my_model(data)
```
Wouldn't that just be creating an object? (like calling the class constructor)
How, in pytorch, are inference supposed to be made?

(for reference I am talking when `my_model` is set to `my_model.eval()`)
EDIT: My apologies. I made the mistake of declaring the class and object as one.. I corrected the code

## Answer 1 (score: 3)
You are confusion [ __init__](https://stackoverflow.com/q/625083/1714410) and
`__call__``my_model` is a ```
my_model_instance = my_model(arguments)
Invoke's `my_model.__init__` with `arguments`. The result of this call is a new *instance* of `my_model` in the variable `my_model_instance`.

Once you instantiated the *class* `my_model` as the variable `my_model_instance`, you can evaluate the model on the training data:

```
tasks_output, other = my_model_instance(data)
"Calling" (i.e., putting parenthesis after the variable name) the instance of the model causes python to invoke the method `__call__` of the class.
In the case of classes derived from `nn.Modules` this will invoke `__call__` of `nn.Module` that does some pytorch stuff and eventually calls your implementation of `forward` method of `my_class`.
Please see [this detailed thread](https://stackoverflow.com/q/9663562/1714410) on the difference between `__init__` and `__call__` in python in general.
It is often a convenient follow [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/#id41):
Class names should normally use the CapWords convention.
Function names should be lowercase, with words separated by underscores as necessary to improve readability. Variable names follow the same convention as function names.

## Answer 2 (score: 1)
You have for exemple :

```
class My_model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Call construtor of Class
my_model = My_model()
It's important to differentie the class and objet. The name's class start with capital letter in Python.

The constructor as you can see, it doesn't take a data/input parameter, alone the function forward have one.

After, for the training, you must to need :
Exemple :

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
For end, you must to need, with a loop, this elements :

```
# forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
Here, you have one iteration of back propagation.

If you want thinking the inference in backpropagation, you can read how create a layer with pytorch and how the pytorch use autograph.

The tensor use Autograph for backpropagation. Exemple with [Pytorch documentation](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
```
import torch
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward()
print(w.grad)
print(b.grad)
The resultat give the backpropagation, where the cross entropy criterion calcul the distance with model and label. The Tensor z is not unique matrice of value but a class with "memory the calcul" with w, b, x, y.
In the layer the gradiant use the forward function for this calcul or a function backward if necesserie.
Best regard

## Answer 3 (score: 1)
Models in PyTorch are defined with classes by inheriting from the base [ nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class:

```
class Model(nn.Module)
    pass
You can then implement a `forward` method that acts as the inference code. Whether it be for training or evaluation, it is supposed to return the output of your model.
```
class Model(nn.Module)
    forward(self, x)
        return x**2
Once you have that you can *initialize* a new model with:

```
model = Model()
To use your newly initialized model, you won't actually call `forward` directly. The underlying structure of `nn.Module` makes it such that you can call `__call__` instead. Which will handle the call to your `forward` implementation. To use it, you will just call your object like a function:

```
>>> model(2)

In the [documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval) you can see that `nn.Module.eval` will set the model to evaluation mode which affects particular layers such as batch normalization layers and dropouts. These types of layers are usually turned on for training and turned off for evaluation and testing. You can use it as
```
model.eval()
When doing model evaluation and testing, it is advised to use the [ torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html?highlight=grad#torch.no_grad) context manager. This avoids having to retain the activations which are used for gradient backpropagation.
```
with torch.no_grad():
    out = model(x)
Or as a decorator on top of your function/method declaration:

```
@torch.no_grad()
validate():
    pass
```

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)

[[PyTorch-内部机制|PyTorch 内部机制]]
