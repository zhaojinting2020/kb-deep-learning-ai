---
title: Pytorch Lightning 笔记
source: converted:attachments/documents/root-a2f0cf872f07/Pytorch Lightning 笔记.pdf
source_type: PDF
quality: draft
attachments:
- file: attachments/documents/root-a2f0cf872f07/Pytorch Lightning 笔记.pdf
  title: Pytorch Lightning 笔记.pdf
---

# Pytorch Lightning 笔记

import torch . . . . . . . . . . . . . . . . . . . . . . from torch import nn from torch import optim 0 O o 0 0 0 0 ° 0 0 0 0 O o 0 0 0 0 ° 0 0 from torchvision import datasets, transforms from torch.utils.data import random_split, DataLoader : . . . : : . . . . : : . . . : : . . . . train_data = datasets.MNIST(‘data', train=True, download=False, transform=transforms.ToTensor()) : : . . : . : : : : . : : train, val = random_split(train_data, [55000, 5000}) train_loader = DataLoader(train, batch_size=62) 0 0 ° 0 0 9 9 ° ° 0 9 0 0 val_loader = DataLoader(val, batch_size=62) # Define a more flexible model [82] # Define a more flexible model class ResNet(nn.Module): class ResNet(nn.Module): . . . . 5 . . . . 5 . . . def __init__ (self): def __init (self): super().__init__() super().__init__() . . . . . . . . . . . . . self.11 = nn.Linear(28 * 28, 64) self.11 = nn.Linear(28 * 28, 64) self.12 = nn.Linear(64, 64) self.12 = nn.Linear(64, 64) . . . . . . . . . . . . . self.13 = nn.Linear(|64, 10) self.13 = nn.Linear(64, 10) self.do = nn.Dropout(0.1) self.do = nn.Dropout(0.1) . . . . . . . . . . . . . def forward(self, x): def forward(self, x): . . . . . . . . . , , . , hl = nn. functional .relu(self.11(x) .relu(self.11(x) ff hl = nn.functional.relu(self.11(x)) h2 = nn.functional.relu(self.12(h1j) h2 = nn. functional.relu(self.12(h1)) . . . . . . . . . . . . . do = self.do(h2 + hl) do = self.do(h2 hi)@ logits = self.13(do) logitsA = self.13(do) + build a HighN between ha out Res return===== ROS E3U)logits model return= ResNetlogitsW + Cobiasthetto clisublermodel canLo decide wether shadaue [a or jiegive Lsea vr3) Neseliga - k Define my optimiser ° to 4on werne ie Saranctelacdsltcaraaeceret) Se tue fastey optimiser = optim.SGD(params, lr=le-2) # Define my loss loss = nn.CrossEntropyLoss() - . . . . . . C 0 o o 5 2 2 o ° 0 C 0 o o 0 0 0 0 0 0 0 # My training and validation loops . . 5 A 5 . . . . . . , . 5 A 5 . . . . . . . . . nb_epochs = 5 for epoch in range(nb_epochs): 5 p p p ; ; . . . . . . . . . . . . . . . . . . . osses = list() del .train|()| .train|()| .train|()| BK +o enable dropout . . . . . . . . . . . . . . . . . . . . . . . . . or batchx,x,x, y =tbatchtrain_loader:tbatchtrain_loader:batchtrain_loader:train_loader:tbatchtrain_loader:batchtrain_loader:train_loader:batchtrain_loader:train_loader:train_loader: . . . . . . . . . . . . . . . . . . . . . . . . . b = x.size(0) x = x.view(b, -1).cudat) send x t cde , , , 5 , , , , , , , , , , 5 . . . . . . . . . . 1 = model(x) # 1: logit : : # 2 compute the objective function . . . . : . . . . : : . . . . : . . . . : 0 d J = loss(l, y.cuda())——~send 49——~send 49send 49 49 cadla c c 5 ; 0 c c 0 ; ; c c c 5 ; 0 c c 0 ; ; # 3 cleaning the gradients ° d model.zero_grad() O o ° 0 0 0 C ° 0 0 ; O o ° 0 0 0 C ° 0 0 # optimiser.zero_ grad() grad() } ° # params.grad._zero() 0 o o 5 2 2 o ° 0 C 0 o o 5 2 2 o ° 0 . . # 4 accumulate the partial derivatives of J wrt params . . . . . . . . . . . . . . . . . . . . . J.backward() . d # 5 step in the opposite direction of the gradient . . . . . . . . . . . . . . . . . . . . . optimiser.step() ° losses.append(J.item()) : . . : . . . . ° 0 5 0 0 ° 0 0 accuracies .append(y.eq(1.detach() .argmax(dim=1 .argmax(dim=1 yy ca )-mean()) )-mean()) print(f'Epoch {epoch + 1}', end=', ') . print(f'training loss: {torch.tensor(losses).mean():.2£}', end=', ') t . . . . . . . . . . . . . . . . print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}') ° leeeiencignlosseslosses = list()©© Migeap 9 oO O C ° 0 0 oO O ° 9 9 oO O C ° : 0 oO O ° 9 9 ° ‘foemodel.eval()needamodel.eval()needaneeda ah coal Dendtree 0 ; O o ° 0 0 0 C ° 0 0 ; O o ° 0 0 0 C ° 0 0 x, Y = batch #x: bx 1x 28 x 28 t b = x.size(0) x = x.view(b, -1) # 1 forward with torch.no_grad(): 5 i = model(x) #1: logits a, # 2 compute the objective function . . . . . . J = loss(l, y) losses.append(J.item() ) . print(f'Epoch {epoch + 1}', end=', ') : : : : . : : : : . . . . . . . . . print(f'validation loss: {torch.tensor(losses).mean():.2£}', end=', ') : print(f'validation accuracy: {torch.tensor(accuracies) .mean():.2£}') : ° . . . ‘ ° . . G : ° . . . ‘ 

# Reformer into Liptorch Lightning 

0 import pytorch_lightning as pl 5 0 0 0 9 0 0 0 0 0 0 0 0 0 0 9 9 0 0 0 0 0 0 5 LFromclassRytorchResNet(pl.LightningModule)—UgntningclassRytorchResNet(pl.LightningModule)—UgntningRytorchResNet(pl.LightningModule)—UgntningResNet(pl.LightningModule)—Ugntning—Ugntning . metrics . fanctional import:: accarac Ys 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 5 5 5 . 5 ef init (self): . super ( ) hehe __( hehe __( __( ) . . . . . . . . . . . . . . . . . . . . . . . self.11 = nn.Linear(28 * 28, 64) . self.12 = nn.Linear(64, 64) . . . . . : . : . . : . . . . . . . : . . : . , self.13self.do == nn.Linear(64,nn.Dropout(0.1)10) , ; , ; : ; , , ; , , , ; , ; : ; , , ; , , , ; sett. loss = pn. Cross Entropy Entropy Lass c) . . . . . . . . . . . . . . . . . . . . . . . ef forward(self, x): : hi = nn.functional.relu(self.11(x))) : : : : : : : : : : : : : : : : : : : : : : : h2 = nn.functional.relu(self.12(h1)) 0 do = self.do(h2 + h1) 0 0 0 0 0 0 0 0 O 0 0 0 0 0 0 0 0 0 0 O 0 0 0 logits = self.13(do) . ef configure_optimizers (xf): } : . : : : : . : : : : : . : : : : . : : : optimiserreturn optimizer= optim.SGD(self.parameters(), lr=le-2) : ; . . : : : : : : . : ; : . : : : : : . ef training step(self, batch, batch_idx): 5 . 5 5 5 5 5 5 5 5 5 5 . 5 5 0 0 0 0 0 0 0 x, y = batch ni SH Jey Sg Ab ple PAS bie 783 . b = x.size(0) . . . . . . . . . . . . . . . . . . . . . . x = x.view(b, -1) # 1 forward . logits = self(x) # logits: logits . . . . . . . . . . . . . . . . . . . . . . . # 2 compute the objective function . . . . . . . . . . . . . . . . . . . . . . J = self.loss(logits, y) acc = accuracy(logits, y) 0 0 | pbar = {'train_acc': acc} 0 0 0 0 0 0 0 O 0 0 0 0 0 0 0 0 0 0 O 0 0 0 return fl'loss': J, ‘progress bar’: pbar}| def validation_step(self, batch, batch_idx): . . results = self.training step(batch, batch_idx) . . . . . . . . . . . . . . results['progress_bar'][‘val_acc'] = results[{'progress_bar'][{'train_acc'] . . del results['progress_bar']['‘train_acc'] . . . . . . . . . . . . . . return results def validation_epoch_end(self, val_step_outputs): . . avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean() . . . . . . . . . . . . . . avg_val_acc = torch.tensor([x[‘progress bar'][{'val_acc'] for x in outputs]).mean() . . pbarreturn= {'avg_val_acc':{'val_loss': avg_val_loss,avg _val_acc}‘progress _bar': pbar} . . . . . . . . . . : . . . . . def prepfre_data(self): . . . . . . . . . . . . . datasets.MNIST(‘data', train=True, download=True, transform=transforms.ToTensor()) ne only copy one fy oh . . . , def setup(self): 0 ° dataset = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor( ~ machines, ° 9 C 9 3 ° ° 0 ° 0 self.train, self.val = random_split(train_data, [55000, 5000}) def train_dataloader(self): . . train_loader = DataLoader(self.train, batch_size=32) . . . . . . . . . . . . . return train_loader def val_dataloader(self): . . val_loader = DataLoader(self.val, batch_size=32) . . . . . : . . . . . . . return val_loader model= Rac Met) trainer = pl.Trainer|(progress_bar_refresh_rate=20, max_epochs=5, gpus=8, num_nodef=32)] . trainer.fit(model) . . . . . . . . . . . . . 

Enables auto adding of DistributedSampler .In PyTorch, you must use it in distributed settings such as TPUs or multi-node. The sampler makes sure each GPU sees the appropriate part of your data. By default it will add shuffle=True for trainsampler and shuffle=False for val/test sampler. If you already use a custom sampler, Lightning will wrap it in a way that it samples from your sampler in a distributed manner. If you want to customize it, you can set replace lsamplerlddp=False and add your own distributed sampler. If replace sampler jddp=True and a distributed sampler was already added, Lightning will not replace the existing one. 

# default used by the Trainer trainer = Trainer(replace_sampler_ddp=True) 

trainer = Trainer(replace_sampler_ddp=False) sampler * torch.utils.data.distributed.DistributedSampler(dataset, shuffle#Truec) dataloader = DataLoader(dataset, batch_size=32, sampler=sampler | 

Enable synchronization between batchnorm layers across all GPUs. 

###### trainer = Trainer(sync_batchnorm=True) 

Ifsetto True will call prepare_data() on LOCAL_RANK=0for every node. If set to False will only call from NODE_RANK=0, LOCAL_RANK=0. 

class LitModel(LightningModule): def _init__(self): super().__init__() self.prepare_data_per_node = True 

# 2. Init Trainer HHAHHHHHHAAAAHHHHABE trainer = pl.Trainer(auto_lr find=True) 

# tune to find the learning rate trainer.tune(model) 

model.learning rate 

HERRERAHATH ET 

# 2. Init Trainer BREAAER EEE trainer = pl.Trainer(auto_ir find=True) 

lr_finder = trainer.tuner.lr find(model) # Run learning rate finder 

fig = lr_finder.plot(suggest=True) # Plot fig. show( ) 

model.hparams.learning rate = lr_finder.suggestion( ) 

he ita aE EEE EEE EEE ER EE trainer.fit(model, train_loader, val_loader) 

trainer = pl.Trainer(callbacks=(EarlyStopping( 'val_loss|')}) 

## Stopping an Epoch Early 

You can stop and skip the rest of the current epoch early by overriding on_train_batch_start() toreturn -1 when some condition is met. 

If you do this repeatedly, for every epoch you had originally requested, then this will stop your entire training. 

### EarlyStopping Callback 

The EarlyStopping callback can be used to monitor a metric and stop the training when no improvement is observed. 

To enable it: 

- e Import EarlyStopping callback. 

- e Log the metric you want to monitor using 1og() method. 

- e Init the callback, and set monitor to the logged metric of your choice. 

- e Set the mode based on the metric needs to be monitored. 

- e Passthe EarlyStopping callbacktothe Trainer callbacks flag. 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping 

class LitModel(LightningModule): 

def validation_step(self, batch, batch_idx): 10ss = ... self.log("val_loss", loss) 

model = LitModel() 

trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")]) trainer.fit (model) 

You can customize the callbacks behaviour by changing its parameters. 

early_stop_callback = EarlyStopping(monitor="val_ accuracy", min_delta=0.00, patience=3, verbose=False, mode="max") trainer = Trainer(callbacks=[early_stop_callback]) 

In case you need early stopping in a different part of training, subclass EarlyStopping and change where it is called: 

###### class MyEarlyStopping(EarlyStopping): 

def on_validation_end(self, trainer, pl_module): 

# override this to disable early stopping at the end of val loop pass 

###### def on_train_end(self, trainer, pl_module): 

# instead, do it at the end of training loop self._run_early_stopping_check(trainer) 

###### patience ( int ) - 

number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch. However, the frequency of validation can be modified by setting various parameters on the Trainer , for example check_val_every_n_epoch and val_check_interval . 

It must be noted that the patience parameter counts the number of validation checks with no improvement, and not the number of training epochs. Therefore, with parameters check_val_every_n_epoch=10 and patience=3,the trainer will perform at least 40 training epochs before being stopped. 

trainer = pl.Trainer(weights_summary='top') 

- | Name | Type | Params 

- 0 | encoder | Sequential | 50 K 

- 1 | decoder | Sequential | 51 K 

trainer = pl.Trainer(weights_summary='full') 

|||Name|| Type|| Params|
|---|---|---|---|
|0<br>||encoder|| Sequential||<br>50 K|
|1<br>| <br><br>|encoder.0<br>|| Linear<br>||<br>50 K<br>|
|2<br>|<br><br>|encoder.1<br>||ReLU<br>||0<br><br>|
|<br>3<br>| <br><br>|<br> encoder.2<br>|<br>| Linear<br>|<br>|<br>195<br>|
|4<br>| <br>|decoder<br>|| Sequential<br>|| 51 K<br>|
|5||decoder.||Linear|=|256|

##### Profiler Report 

|Action<br>||Mean duration (s)||<br>Total time (s)|
|---|---|---|
|on_fit_start<br>||2.8406e-05||<br>2.8406e-05|
|on_validation_start<br>||0.015724||<br>0.031448|
|on_validation_epoch_start||<br>2.3732e-05||<br>4.7464e-05|
|on_validation_batch_start||<br>1.6738e-05||<br>0.0026614|
|validation_step_end<br>||1.476e-05||<br>0.0023469|
|on_validation_batch_end<br>||9.0561le-05||<br>0.014399|
|on_validation_epoch_end<br>||3.5103e-05||<br>7.0205e-05|
|on_validation_end<br>|<br><br>|0.0054017<br>||<br>0.010803<br><br>|
|on_train_start<br>|<br><br>|0.043631<br>||<br>0.043631<br><br>|
|on_epoch_start<br>||0.0035997||<br>0.0035997|
|on_train_epoch_start<br>||2.1428e-05||<br>2.1428e-05|
|get_train batch<br>||0.0027387||<br>4.7078|
|on_batch_start<br>||1.7416e-05||<br>0.029937|
|on_train_batch_start<br>||9.9967e-06||<br>0.017184|
|training _step_end<br>||1.5519e-05||<br>0.026677|
|model _forward<br>||0.00071537||<br>1.2297|
|model backward<br>|<br><br>|0.00057614<br>||<br>0.99038<br><br>|
|on_after_backward<br>||1.2127e-05||<br>0.020846|
|optimizerstep<br>||0.0011936||<br>2.0518|
|<br><br>on_batch_end<br>|<br><br>|1.3813e-05<br>||<br>0.023745<br><br>|
|<br><br>jontrein_batchend.<br>||0.00014947||<br>0.25693|

Runs nifsetto n (int) else 1 ifsetto True batch(es) to ensure your code will execute without errors. This applies to fitting, validating, testing, and predicting. This flag is only recommended for debugging purposes and should not be used to limit the number of batches to run. 

# default used by the Trainer trainer = Trainer(fast_dev_run=False) 

# runs only 1 training and 1 validation batch and the program ends trainer = Trainer(fast_dev_run=True) trainer.fit(...) 

# runs 7 predict batches and program ends trainer = Trainer(fast_dev_run=7) trainer.predict(...) 

#### trainer = Trainer(overfit_batches=0.01) 

Uses this much data of the training & validation set. If the training & validation dataloaders have shuffle=True , Lightning will automatically disable it. 

Useful for quickly debugging or trying to overfit on purpose. 

# default used by the Trainer trainer = Trainer(overfit_batches=0.©) # use only 1% of the train & val set trainer = Trainer(overfit_batches=0.01) 

# overfit on 10 of the same batches trainer = Trainer(overfit_batches=10) 

How often to add logging rows (does not write to disk) 

# default used by the Trainer trainer = Trainer(log_every_n_steps=50)

---

## 源文件

- [[attachments/documents/root-a2f0cf872f07/Pytorch Lightning 笔记.pdf|Pytorch Lightning 笔记。pdf]]
