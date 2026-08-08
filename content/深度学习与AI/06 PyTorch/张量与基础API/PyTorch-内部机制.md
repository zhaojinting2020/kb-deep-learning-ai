---
title: PyTorch 内部机制
url: https://mp.weixin.qq.com/s/mGLFBEJWVG-ucxdNnN_aig
fetch_source: wechat_mobile_ua+cookies
fetched_at: '2026-06-27T16:29:15+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# PyTorch 内部机制

*作者：AI公园*点击上方“AI公园”，关注公众号，选择加“星标“或“置顶”作者：Daniel Godoy编译：ronghuaiyang网上有非常多的PyTorch的教程，但是这个教程很值得一读，结构化，增量化的学习PyTorch.
**PyTorch**是**增长最快的**深度学习框架，**Fast.ai**也在其MOOC课程中，如Deep Learning for Coders 中也用到了它。

PyTorch也非常pythonic，这意味着，如果你已经是Python开发人员，那么使用它会感觉更自然。此外，根据Andrej Karpathy的说法，使用PyTorch甚至可能改善您的健康:-)网上有非常的PyTorch教程，它的文档非常完整。那么，**为什么**你还要读我这个教程呢？

尽管人们可以找到关于PyTorch可以做的几乎所有事情的信息，但是没有一个从**基本原则**到它的**结构**, **逐渐**来学习的方法。在这篇文章中，我会让你了解PyTorch学起来很容易的主要原因和在Python中构建一个深度学习模型的更多**直觉**：**自动微分**,  **动态计算图** ,  **模型类**等等，我也会告诉你如何避免一些**常见的陷阱和错误**。首先，我们需要介绍一些基本概念，如果在进行全面建模之前没有很好地掌握它们，这些概念可能会使你学起来感到吃力。在深度学习中，我们随处可见**tensors**。嗯，谷歌的框架被称为TensorFlow是有原因的！那到底什么是张量？

在Numpy中，你可能有一个三维数组，对吧？这是一种技术上的说法。
**标量**（单个数字）维度为零，**向量**维度为1，**矩阵**维度为2，**张量**的维度为3或者更多，就是这样了！

但是，为了让事情变得简单，我们通常把向量和矩阵也叫作张量，所以，从现在开始，**所有的东西要么是标量，要么是张量！** : 张量就是高纬度的矩阵T你可能会问：“如何从Numpy的数组转换到PyTorch的张量 ？”这就是 ** from_numpy**的作用。不过，它返回一个“但是我想使用我的高级GPU……”，你说。不用担心，这就是**  to()**的好处。它将你的张量发送到你指定的任何`cuda` 或者 `cuda:0`).“如果没有GPU可用的话，我怎么让我的代码回退到CPU？“，你可以使用**  cuda.is_available()**来查看你是否有GPU，并相应地设置你的设备。你还可以使用 `float()`轻松地转换为精度较低的浮点数(32位浮点数).
`import torch` `import torch.optim as optim` `import torch.nn as nn` `from torchviz import make_dot` `device = 'cuda' if torch.cuda.is_available() else 'cpu'` `# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors` `# and then we send them to the chosen device` `x_train_tensor = torch.from_numpy(x_train).float().to(device)` `y_train_tensor = torch.from_numpy(y_train).float().to(device)` `# Here we can see the difference - notice that .type() is more useful` `# since it also tells us WHERE the tensor is (device)` `print(type(x_train), type(x_train_tensor), x_train_tensor.type())`如果你比较这两个变量的**类型**，你会得到：第一个是 `numpy.ndarray`，第二个是 `torch.Tensor`。但是你的张量放在哪里呢？在CPU还是GPU中？你不知道…你可以使用PyTorch的**  type()**，它将显示它的`torch.cuda.FloatTensor`—这是一个GPU中的张量。我们也可以反过来，使用**  numpy()**将张量转换回Numpy数组。它应该像`x_train_tensor.numpy()` `TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.`不幸的是，Numpy**不能**处理GPU张量……你需要首先使用**  cpu()**使它们成为CPU张量。如何区分不同的张量 —就像我们刚刚创建的**张量**—用作(可训练)**参数/权重** 的**张量**？

后一个张量需要**计算它的梯度**，所以我们可以**更新**它们的值(即参数的值). 这就是**  requires_grad=True**参数的作用。它告诉PyTorch我们想让它为我们计算梯度。你可能想为一个参数创建一个简单的张量，然后，把它发送到你选择的设备上，就像我们处理数据一样，对吧？没那么快……
`# FIRST` `# Initializes parameters "a" and "b" randomly, ALMOST as we did in Numpy` `# since we want to apply gradient descent on these parameters, we need` `# to set REQUIRES_GRAD = TRUE` `a = torch.randn(1, requires_grad=True, dtype=torch.float)` `b = torch.randn(1, requires_grad=True, dtype=torch.float)` `print(a, b)` `# SECOND` `# But what if we want to run it on a GPU? We could just send them to device, right?` `a = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float).to(device)` `print(a, b)` `# Sorry, but NO! The to(device) "shadows" the gradient...` `# THIRD` `# We can either create regular tensors and send them to the device (as we did with our data)` `a = torch.randn(1, dtype=torch.float).to(device)` `b = torch.randn(1, dtype=torch.float).to(device)` `# and THEN set them as requiring gradients...` `a.requires_grad_()` `b.requires_grad_()` `print(a, b)`第一个代码块为我们的参数创建了两个很好的张量，具有梯度等等。但是它们是**CPU**张量。
`# FIRST` `tensor([-0.5531], requires_grad=True)` `tensor([-0.7314], requires_grad=True)`在第二段代码中，我们尝试了将它们发送到GPU的**naive**方法。我们成功地将它们发送到另一个设备上，但不知怎的，我们“**丢失**”了**梯度**。
`# SECOND` `tensor([0.5158], device='cuda:0', grad_fn=<CopyBackwards>) tensor([0.0246], device='cuda:0', grad_fn=<CopyBackwards>)`在第三段代码中，我们**首先**把我们的张量发送到设置中，然后使用**  requires_grad_()**方法将张量中的`requires_grad`属性设置为 `True`。
`# THIRD` `tensor([-0.8915], device='cuda:0', requires_grad=True) tensor([0.3616], device='cuda:0', requires_grad=True)`在PyTorch中，每一种以下划线结束的函数都说是in-place的，意思是改变的变量本身。尽管最后一种方法工作得很好，但是最好是在创建张量的时候就**把张量分配给设备**。
`# We can specify the device at the moment of creation - RECOMMENDED!` `torch.manual_seed(42)` `a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `print(a, b)` `tensor([0.6226], device='cuda:0', requires_grad=True) tensor([1.4505], device='cuda:0', requires_grad=True)`现在我们知道了如何创建需要梯度的张量，让我们看看PyTorch如何处理它们。

Autograd是PyTorch的自动微分包。多亏了它，我们**不需要担心**偏导，链式法则或类似的东西。那么，我们如何告诉PyTorch执行它的操作并**计算所有的梯度**？这就是**  backward()**的好处。你还记得计算梯度的**起始点**吗？就是**损失**，我们需要对参数计算它的偏导数。因此，我们需要从对应的Python变量调用 `backward()`方法，比如 `loss.backward().`。那么**gradient *的**实际值*是多少呢？我们可以通过查看张量的 `grad` 属性来看到。如果看一下文档，可以清楚地看到**梯度是累积的**。所以，每次我们使用**gradients**来**更新**参数时，我们需要把前面的梯度归零。这就是**  zero_()**的好处。因此，让我们**抛弃手动计算梯度**，同时使用 `backward()`和 `zero_()`方法。
`lr = 1e-1` `n_epochs = 1000` `torch.manual_seed(42)` `a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `for epoch in range(n_epochs):` `yhat = a + b * x_train_tensor` `error = y_train_tensor - yhat` `loss = (error ** 2).mean()` `# No more manual computation of gradients!` `# a_grad = -2 * error.mean()` `# b_grad = -2 * (x_tensor * error).mean()` `# We just tell PyTorch to work its way BACKWARDS from the specified loss!` `loss.backward()` `# Let's check the computed gradients...` `print(a.grad)` `print(b.grad)` `# What about UPDATING the parameters? Not so fast...` `# FIRST ATTEMPT` `# AttributeError: 'NoneType' object has no attribute 'zero_'` `# a = a - lr * a.grad` `# b = b - lr * b.grad` `# print(a)` `# SECOND ATTEMPT` `# RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.` `# a -= lr * a.grad` `# b -= lr * b.grad` `# THIRD ATTEMPT` `# We need to use NO_GRAD to keep the update out of the gradient computation` `# Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...` `with torch.no_grad():` `a -= lr * a.grad` `b -= lr * b.grad` `# PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...` `a.grad.zero_()` `b.grad.zero_()` `print(a, b)`第一次尝试，如果我们使用和Numpy代码相同的，我们会得到下面的奇怪的错误……但我们可以得到一个提示，通过查看张量本身，发现我们在更新参数的时候，再次**把梯度丢了**。因此， `grad`属性变成了 `None`，它会引发错误…
`# FIRST ATTEMPT` `tensor([0.7518], device='cuda:0', grad_fn=<SubBackward0>)` `AttributeError: 'NoneType' object has no attribute 'zero_'`然后，在第二次尝试中，我们使用熟悉的**in-place Python赋值**对其进行了轻微的更改。而且，PyTorch再次抱怨它并提出了一个**错误**。
`# SECOND ATTEMPT` `RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.`为什么？！罪魁祸首是PyTorch能够从每一个Python操作中构建一个动态计算图，该操作涉及到任何梯度计算张量或及其依赖项。在下一节中，我们将更深入地研究动态计算图的内部工作原理。那么，我们如何告诉PyTorch **“back off”**，并让我们**更新我们的参数**，但不会打乱它的花哨的动态计算图呢？这就是 `torch.no_grad()`的好处。它允许我们**对张量**执行常规的Python操作，**独立于PyTorch的计算图**。最后，我们成功地运行了我们的模型并获得了**结果参数**。当然，它们和我们在Numpy only实现中得到的那些是匹配的。
`# THIRD ATTEMPT` `tensor([1.0235], device='cuda:0', requires_grad=True) tensor([1.9690], device='cuda:0', requires_grad=True)`“不幸的是，没有人知道动态计算图是什么。你得亲眼看看。” Morpheus PyTorchViz包及其`make_dot(variable)`方法允许我们轻松地可视化与给定Python变量关联的图。因此，让我们使用：两个(梯度计算)**张量**作为参数, 预测, 误差和损失。
`torch.manual_seed(42)` `a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `yhat = a + b * x_train_tensor` `error = y_train_tensor - yhat` `loss = (error ** 2).mean()`如果我调用 `make_dot(yhat)`我们会得到下面最左边的图：
让我们仔细看看它的几个部分：
**蓝色块**：这些是我们用作参数的张量，我们想让PyTorch来计算它们的梯度。
**灰色块**：包括了梯度计算张量及其依赖的Python操作。
**绿色块**：和灰色块一样，只不过它是gradients计算的起点。(假设 `backward()`方法是从用来进行可视化**图形的**变量调用的)—它们在图中是自底向上计算的。如果我们为 `error`(中间)和 `loss`(右边)**变量**绘图，那么它们与第一个变量之间的**惟一区别是**中间步骤的数量**(**灰色框). 现在，仔细看看**最左边**图的**绿色框**：有两个箭头指向它，因为它是将两个变量相加，即 `a` 和 `b*x` 。很明显吧？

然后，查看相同图的灰色框：它执行**乘法** ，即 `b*x`。但是只有一个箭头指向它！箭头来自绿色框，它对应于我们的参数 `b` 。为什么我们没有一个框来存放我们的**数据x**？答案是：我们**不为它计算梯度** ！因此，即使计算图执行的操作涉及多个张量，它只显示**梯度计算张量及其依赖项**。如果我们将参数a设置为 `requires_grad`为 `False`，计算图会发生什么？

毫无疑问，与**参数a**对应的**蓝色框**不再存在！很简单：**没有梯度，没有图形**。
**动态计算图**最好的一点是，你可以让它像你想要的那样复杂。您甚至可以使用控制流语句(例如if语句)来**控制梯度流**:-)
下面的图显示了一个例子。到目前为止，我们一直在使用计算的梯度手动更新参数。这对于两个参数来说可能没问题，但是如果我们有**多个参数**呢？我们使用PyTorch的**优化器**，比如[**SGD**或**Adam**。优化器获取我们想要更新的**参数**, 我们想要使用的**学习率**(可能还有许多其他超参数！)，并且**通过它的  step()**方法执行更新。此外，我们也不需要一个一个地把梯度置为零。我们只需要调用优化器的 `zero_grad()`方法，就这样！

在下面的代码中，我们创建了一个随机梯度下降 (SGD)优化器来更新我们的参数**a**和**b**。不要被优化器的名称所迷惑：如果我们同时使用所有的训练数据进行更新——正如我们在代码中所做的那样——优化器执行的是batchgradient descent，不要管它的名称是什么。
`torch.manual_seed(42)` `a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `print(a, b)` `lr = 1e-1` `n_epochs = 1000` `# Defines a SGD optimizer to update the parameters` `optimizer = optim.SGD([a, b], lr=lr)` `for epoch in range(n_epochs):` `yhat = a + b * x_train_tensor` `error = y_train_tensor - yhat` `loss = (error ** 2).mean()` `loss.backward()` `# No more manual update!` `# with torch.no_grad():` `# a -= lr * a.grad` `# b -= lr * b.grad` `optimizer.step()` `# No more telling PyTorch to let gradients go!` `# a.grad.zero_()` `# b.grad.zero_()` `optimizer.zero_grad()` `print(a, b)`让我们检查一下之前和之后的两个参数，以确保一切正常：
`# BEFORE: a, b` `tensor([0.6226], device='cuda:0', requires_grad=True) tensor([1.4505], device='cuda:0', requires_grad=True)` `# AFTER: a, b` `tensor([1.0235], device='cuda:0', requires_grad=True) tensor([1.9690], device='cuda:0', requires_grad=True)`太酷了！我们优化了**优化**流程:-)还剩下什么？

我们现在处理**损失的计算**。正如我们所料，PyTorch又一次帮我们搞定了。根据手头的任务，可以选择许多loss function. 由于我们的任务是一个回归，所以我们使用的是均方误差(MSE)损失。注意，`nn.MSELoss`实际上为我们创建了一个损失函数—它不是损失函数本身。此外，你还可以指定要应用的reduction method，即如何合并每个点的结果—你可以对它们进行平均(reduction = ' mean ')或简单地将它们求和(reduction = ' sum '). 然后，在后面的第20行，我们使用**创建的损失函数**计算给定的**预测**和**标签**的损失。我们的代码现在看起来是这样的：
`torch.manual_seed(42)` `a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)` `print(a, b)` `lr = 1e-1` `n_epochs = 1000` `# Defines a MSE loss function` `loss_fn = nn.MSELoss(reduction='mean')` `optimizer = optim.SGD([a, b], lr=lr)` `for epoch in range(n_epochs):` `yhat = a + b * x_train_tensor` `# No more manual loss!` `# error = y_tensor - yhat` `# loss = (error ** 2).mean()` `loss = loss_fn(y_train_tensor, yhat)` `loss.backward()` `optimizer.step()` `optimizer.zero_grad()` `print(a, b)`此时，只剩下一段代码需要更改：*predictions *。现在是时候介绍PyTorch的方法来实现…

在PyTorch中，**model**由一个常规的**Python类**表示，该类继承自**Module**类。它需要实现的最基本的方法是：
`__init__(self)`: **定义了模型的构建部分**—在我们的例子中，包括两个参数**a** 和**b**.

模型可以包含其他模型(或层)作为它的属性，所以可以很容易地嵌套它们。我们很快也会看到一个例子。
`forward(self, x)`: 进行了实际的计算，也就是说，对于输入**x**，输出一个预测。但是，你不应该调用`forward(x)`方法。你应该调用整个模型本身，就像`model(x)`中一样，以执行正向传输和输出预测。让我们为我们的回归任务构建一个合适的(但简单的)模型。它应该是这样的：
`class ManualLinearRegression(nn.Module):` `def __init__(self):` `super().__init__()` `# To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter` `self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))` `self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))` `def forward(self, x):` `# Computes the outputs / predictions` `return self.a + self.b * x`在 `__init__`方法中，我们定义了**两个参数**，**a**和**b**，使用 `Parameter()`类告诉PyTorch这些**张量应该被视为模型的参数**。我们为什么要关心这个？通过这样做，我们可以使用模型的 `Parameter()`方法迭代检索模型的所有参数，甚至是嵌套模型的那些参数，我们可以使用它们来提供给优化器(而不是自己构建参数列表！)此外，我们可以使用模型的 `state_dict()`方法获得所有参数的**当前值**。重要：我们需要将模型发送到数据所在的设备。如果我们的数据是由GPU张量构成的，我们的模型也必须“活”在GPU内部。我们可以使用所有这些方便的方法来改变我们的代码，应该是这样的：
`torch.manual_seed(42)` `# Now we can create a model and send it at once to the device` `model = ManualLinearRegression().to(device)` `# We can also inspect its parameters using its state_dict` `print(model.state_dict())` `lr = 1e-1` `n_epochs = 1000` `loss_fn = nn.MSELoss(reduction='mean')` `optimizer = optim.SGD(model.parameters(), lr=lr)` `for epoch in range(n_epochs):` `# What is this?!?` `model.train()` `# No more manual prediction!` `# yhat = a + b * x_tensor` `yhat = model(x_train_tensor)` `loss = loss_fn(y_train_tensor, yhat)` `loss.backward()` `optimizer.step()` `optimizer.zero_grad()` `print(model.state_dict())`现在，打印出来的语句将是这样的——参数**a**和**b**的最终值仍然相同，所以一切正常:-) `OrderedDict([('a', tensor([0.3367], device='cuda:0')), ('b', tensor([0.1288], device='cuda:0'))])` `OrderedDict([('a', tensor([1.0235], device='cuda:0')), ('b', tensor([1.9690], device='cuda:0'))])`我希望你注意到了代码中的一个特殊语句，我给它写了一个注释，**“What is this?!?” —  model.train()**。在PyTorch中，模型有一个`train()`方法，但是，这个并不是执行训练步骤。其唯一目的是将模型设置为训练模式。为什么这很重要？例如，有些模型可能使用Dropout这样的机制，这些机制在训练和评估阶段具有不同的行为。在我们的模型中，我们手动创建了两个参数来执行线性回归。让我们使用PyTorch的**Linear**模型作为我们自己的属性，从而创建一个嵌套模型。尽管这显然是一个人为设计的例子，因为我们几乎是在包装底层模型而没有添加任何有用的东西(或者，根本没有!)，但它很好地说明了这个概念。在 `__init__`方法中，我们创建了一个**属性**，其中包含我们的**嵌套** `Linear` **模型** 。在 `forward()`方法中，我们**调用嵌套模型本身**来执行向前传递(注意，我们不是调用 `self.linear.forward(x)`!).
`class LayerLinearRegression(nn.Module):` `def __init__(self):` `super().__init__()` `# Instead of our custom parameters, we use a Linear layer with single input and single output` `self.linear = nn.Linear(1, 1)` `def forward(self, x):` `# Now it only takes a call to the layer to make predictions` `return self.linear(x)`现在，如果我们调用这个模型的 `parameters()`方法，**PyTorch将以递归方式**显示其属性的参数。你可以使用类似于 `LayerLinearRegression().parameters()`这样的语句来获得所有参数的列表。你还可以添加新的 `Linear`属性，即使在前向传递中根本不使用它们，它们仍然会在 `parameters()`下列出。我们的模型非常简单……你可能会想:“为什么还要为它创建一个类呢？”嗯，你说的有道理……

对于使用 **run-of-the-mill layers**的**straightforward models**，其中一层的输出按顺序作为下一层的输入，我们可以使用**Sequential**模型:-)在我们的例子中，我们将用一个参数构建一个序列模型，即我们用来训练线性回归的 `Linear`层。模型应该是这样的:
`# Alternatively, you can use a Sequential model` `model = nn.Sequential(nn.Linear(1, 1)).to(device)`非常简单，是不是？

到目前为止，我们已经定义了一个**优化器**, 一个**loss函数**和一个**model**。向上滚动一点，快速查看循环内的代码。如果我们使用**不同的优化器**，或者**loss**，甚至**model**，它会**改变**吗？如果没有，我们如何使它**更通用**？

好吧，我想我们可以说所有这些代码行**执行一个训练步骤**，给定这些**三个元素**(优化器, 损失和模型), **特征**和**标签**。那么，如果**编写一个函数，该函数接受这三个元素**，然后**返回执行训练步骤**的另一个函数，将一组特征和标签作为参数并返回相应的损失，情况会如何呢？

然后，我们可以使用这个通用函数来构建一个 `train_step()`函数，该函数将在我们的训练循环中调用。现在我们的代码应该是这样的……看看现在的训练循环有多**小** ?
`def make_train_step(model, loss_fn, optimizer):` `# Builds function that performs a step in the train loop` `def train_step(x, y):` `# Sets model to TRAIN mode` `model.train()` `# Makes predictions` `yhat = model(x)` `# Computes loss` `loss = loss_fn(y, yhat)` `# Computes gradients` `loss.backward()` `# Updates parameters and zeroes gradients` `optimizer.step()` `optimizer.zero_grad()` `# Returns the loss` `return loss.item()` `# Returns the function that will be called inside the train loop` `return train_step` `# Creates the train_step function for our model, loss function and optimizer` `train_step = make_train_step(model, loss_fn, optimizer)` `losses = []` `# For each epoch...` `for epoch in range(n_epochs):` `# Performs one train step and returns the corresponding loss` `loss = train_step(x_train_tensor, y_train_tensor)` `losses.append(loss)` `# Checks model's parameters` `print(model.state_dict())`让我们休息一下，暂时关注一下我们的**data**，到目前为止，我们只是使用了Numpy数组 转成**PyTorch张量**。但我们可以做得更好，我们可以建立一个……

在PyTorch中，**dataset**由一个常规的**Python类**表示，该类继承自**dataset**类。你可以将它看作一种Python **元组列表**，每个元组对应于**一个数据点(特性, 标签)**。它需要实现的最基本的方法是:
`__init__(self)` : 它接受**任何需要的参数**建立一个**元组列表**—可能是一个CSV文件的名称，用来加载和处理，可能是两个张量，一个代表特征，另一个代表标签，或者其他什么，取决于手头的任务。不需要在构造函数方法 ( `__init__`)中加载整个数据集。如果你的dataset很大(例如，成千上万的图像文件)，立即加载不会提高内存效率。建议按需加载(每次调用`__get_item__`的时候).
`__get_item__(self, index)`: 运行dataset进行索引，所以可以让dataset像列表一样进行操作 ( `dataset[i]`) —返回一个**元组（特征，标签）**，对应所取到的数据点。我们可以返回我们预加载数据集的对应切片或则按需进行加载。
`__len__(self)`: 返回整个数据集的大小，不管有没有被采样到，得到的是实际索引的上限的大小。我们来构建一个简单的自定义数据集，它接受两个张量作为参数:一个用于特征，一个用于标签。对于任何给定的索引，我们的dataset类将返回每个张量的对应切片。它应该是这样的：
`from torch.utils.data import Dataset, TensorDataset` `class CustomDataset(Dataset):` `def __init__(self, x_tensor, y_tensor):` `self.x = x_tensor` `self.y = y_tensor` `def __getitem__(self, index):` `return (self.x[index], self.y[index])` `def __len__(self):` `return len(self.x)` `# Wait, is this a CPU tensor now? Why? Where is .to(device)?` `x_train_tensor = torch.from_numpy(x_train).float()` `y_train_tensor = torch.from_numpy(y_train).float()` `train_data = CustomDataset(x_train_tensor, y_train_tensor)` `print(train_data[0])` `train_data = TensorDataset(x_train_tensor, y_train_tensor)` `print(train_data[0])`再一次，你可能会想“为什么要在一个类中这么麻烦来包装几个张量？”。如果一个数据集除了几个张量什么也没有的话，我们可以用PyTorch的**TensorDataset**类，这就是上面我们在自定义数据集上做的事情。你是否注意到我们用Numpy数组构建了我们的训练张量，但是我们没有将它们发送到设备？所以，它们现在是CPU张量！为什么呢？我们不希望我们的全部训练数据被加载到GPU中，就像我们到目前为止的例子中所做的那样，因为这样占用了我们宝贵的显卡RAM中的空间。好吧，不过话说回来，我们为什么要构建数据集呢？我们这么做是因为我们想用…

到目前为止，我们在每个训练步骤都使用了**完整的训练数据**。一直以来都是**批梯度下降**。当然，对于我们的小得可笑的数据集来说，这是可以的，但是如果我们想认真对待这一切，我们**必须**使用**小批量**梯度下降。因此，我们需要小批量。因此，我们需要相应地对数据集进行**切片**。你想“手工”做吗？我不想！

因此，我们使用PyTorch的**DataLoader**类来完成这项任务。我们告诉它使用哪个**dataset**(我们在前一节中刚刚构建的数据集), 所需的**minibatch size**，以及是否需要**shuffle**它。就是这样！

我们的**加载器**将像**迭代器**一样工作，所以我们可以**循环它**并*每次获取不同的minibatch *。
`from torch.utils.data import DataLoader` `train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)`要检索一个mini-batch的样本，可以简单地运行下面的命令—它将返回一个包含两个张量的列表，一个用于特征，另一个用于标签。
`next(iter(train_loader))`这如何改变我们的训练循环？我们来看看！
`losses = []` `train_step = make_train_step(model, loss_fn, optimizer)` `for epoch in range(n_epochs):` `for x_batch, y_batch in train_loader:` `# the dataset "lives" in the CPU, so do our mini-batches` `# therefore, we need to send those mini-batches to the` `# device where the model "lives"` `x_batch = x_batch.to(device)` `y_batch = y_batch.to(device)` `loss = train_step(x_batch, y_batch)` `losses.append(loss)` `print(model.state_dict())`现在有两件事不同了：不仅我们有一个内部循环来从我们的 `DataLoader`加载每个mini-batch，更重要的是，我们现在**只向设备**发送一个mini-batch. 对于更大的数据集，可以使用Dataset的`__get_item__`，通过采样加载数据样本（到CPU的tensor中），然后把一个minibatch中所有的样本发送到GPU中，这样来最大程度利用我们的显卡显存。此外，如果你有*多个gpu *来训练模型，那么最好保持你的数据集“agnostic”，并在训练期间将batch分配给不同的gpu. 到目前为止，我们只关注**训练**数据。我们为它构建了一个dataset和一个data loader. 我们可以对**validation**数据执行同样的操作，使用我们在本文开头执行的**split** ，或者我们可以使用random_split.

PyTorch的 `random_split()`方法是执行**训练验证split**的一种简单而熟悉的方法。请记住，在我们的例子中，我们需要将它应用于**整个数据集**(不是我们在前两节中构建的训练数据集). 然后，对于每个数据子集，我们构建一个对应的 `DataLoader`，因此我们的代码如下:
`from torch.utils.data.dataset import random_split` `x_tensor = torch.from_numpy(x).float()` `y_tensor = torch.from_numpy(y).float()` `dataset = TensorDataset(x_tensor, y_tensor)` `train_dataset, val_dataset = random_split(dataset, [80, 20])` `train_loader = DataLoader(dataset=train_dataset, batch_size=16)` `val_loader = DataLoader(dataset=val_dataset, batch_size=20)`现在，我们的**验证集**有了一个**数据加载器**，因此，将它用于…

这是我们的**最后**部分—我们需要更改训练循环，以包含对模型的评估，即计算**验证损失**。第一步是包含另一个内部循环来处理来自validation loader的mini-batch，将它们发送到与我们的模型相同的设备。接下来，我们使用模型(第23行)进行**预测**，并计算相应的**损失**(第24行). 这就差不多了，但是还有两件事需要考虑：
`torch.no_grad()`: 虽然在我们的简单模型中不会有什么不同，但是用这个**上下文管理器**包装验证**内部循环是一个**好的实践**，以禁用你可能无意中触发的任何梯度计算**——**梯度属于训练**，验证步骤不需要。
`eval()`: 它所做的唯一一件事就是**将模型设置为评估模式**(就像它的 `train()`对应项所做的那样)，这样模型就可以调整它的行为，比如**Dropout**。现在，我们的训练循环应该是这样的:
`losses = []` `val_losses = []` `train_step = make_train_step(model, loss_fn, optimizer)` `for epoch in range(n_epochs):` `for x_batch, y_batch in train_loader:` `x_batch = x_batch.to(device)` `y_batch = y_batch.to(device)` `loss = train_step(x_batch, y_batch)` `losses.append(loss)` `with torch.no_grad():` `for x_val, y_val in val_loader:` `x_val = x_val.to(device)` `y_val = y_val.to(device)` `model.eval()` `yhat = model(x_val)` `val_loss = loss_fn(y_val, yhat)` `val_losses.append(val_loss.item())` `print(model.state_dict())`还有什么我们可以改进或改变的吗？当然，总是有**其他东西**可以添加到你的模型中——例如，使用**学习率策略**。但是这篇文章已经太长了，所以我就到此为止。
“带有所有花哨功能的完整工作代码在哪里？”你可以在这里找到它：https://gist.github.com/dvgodoy/1d818d86a6a0dc6e7c07610835b46fe4。

虽然这篇文章是比我预期的长了，但是我不会让它有所不同——我相信这是人们进行学习的时候最必要的步骤，**结构化，增量化**的方法来学习如何使用PyTorch开发深度学习模型。希望在完成本文中的所有代码之后，你能够更好地理解PyTorch的官方教程，并更轻松地学习它。英文原文：https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e请长按或扫描二维码关注本公众号
**喜欢的话，请给我个好看吧****！**

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-内部机制-qq|PyTorch 内部机制]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/model(x)-与-forward|model(x) 与 forward]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Lightning-+-TensorBoard|Lightning + TensorBoard]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-einsum-爱因斯坦求和|PyTorch einsum 爱因斯坦求和]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/PyTorch-命名张量|PyTorch 命名张量]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-显示为图片|Tensor 显示为图片]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/Tensor-标准化|Tensor 标准化]]
[[content/深度学习与AI/06 PyTorch/张量与基础API/TensorFlow-2-与-Keras-入门|TensorFlow 2 与 Keras 入门]]
