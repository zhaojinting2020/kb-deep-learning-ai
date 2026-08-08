---
title: Dataset 子集 Subset
url: https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
fetch_source: stackexchange:api
fetched_at: '2026-06-28T08:16:59+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
math_repaired_at: '2026-06-27T20:24:23+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# Dataset 子集 Subset

## Question

I have a network which I want to train on some dataset (as an example, say `CIFAR10`). I can create data loader object via
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
```
My question is as follows: Suppose I want to make several different training iterations. Let's say I want at first to train the network on all images in odd positions, then on all images in even positions and so on. In order to do that, I need to be able to access to those images. Unfortunately, it seems that `trainset` does not allow such access. That is, trying to do `trainset[:1000]` or more generally `trainset[mask]` will throw an error.
I could do instead
trainset.train_data=trainset.train_data[mask]
trainset.train_labels=trainset.train_labels[mask]
```
and then
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
```
However, that will force me to create a new copy of the full dataset in each iteration (as I already changed `trainset.train_data` so I will need to redefine `trainset`). Is there some way to avoid it?
Ideally, I would like to have something "equivalent" to
trainloader = torch.utils.data.DataLoader(trainset[mask], batch_size=4,
                                              shuffle=True, num_workers=2)
```
## Answer 1 (score: 128)
[ torch.utils.data.Subset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) is easier, supports
`shuffle`, and doesn't require writing your own sampler:

import torchvision
import torch
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)
evens = list(range(0, len(trainset), 2))
odds = list(range(1, len(trainset), 2))
trainset_1 = torch.utils.data.Subset(trainset, evens)
trainset_2 = torch.utils.data.Subset(trainset, odds)
trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=4,
                                            shuffle=True, num_workers=2)
trainloader_2 = torch.utils.data.DataLoader(trainset_2, batch_size=4,
                                            shuffle=True, num_workers=2)
```
## Answer 2 (score: 28)
You can define a custom sampler for the dataset loader avoiding recreating the dataset (just creating a new loader for each different sampling).
class YourSampler(Sampler):
    def __init__(self, mask):
        self.mask = mask
    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))
    def __len__(self):
        return len(self.mask)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
sampler1 = YourSampler(your_mask)
sampler2 = YourSampler(your_other_mask)
trainloader_sampler1 = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          sampler = sampler1, shuffle=False, num_workers=2)
trainloader_sampler2 = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          sampler = sampler2, shuffle=False, num_workers=2)
```
PS: You can find more info here: [http://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html#Sampler](http://pytorch.org/docs/master/_modules/torch/utils/data/sampler.html#Sampler)
