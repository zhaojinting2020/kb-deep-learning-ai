---
title: 不平衡类别 WeightedRandomSampler
url: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T17:42:21+00:00'
polished_at: '2026-06-27T19:21:51+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# 不平衡类别 WeightedRandomSampler

## post by kkorovesis on Dec 17, 2017
I am trying to balance my data for a multi-classes classification task to get better scores, using weights and torch.utils.data.sampler.WeightedRandomSampler() I get an error that I don’t understand. Is there any other way to handle imbalanced classes easily ? Here is a snippet of my code and the error in hand:
.
train_set = SentimentDataset(file=TRAIN_DATA, word2idx=word2idx, tword2idx=tword2idx,                              max_length=0, max_topic_length=0, topic_bs=True) val_set = SentimentDataset(file=VAL_DATA, word2idx=word2idx, tword2idx=tword2idx,                            max_length=0, max_topic_length=0, topic_bs=True) _weights =  torch.FloatTensor(train_set.weights) # train_set.weights : [296, 3381, 12882, 12857, 1016] _weights = _weights.view(1, 5) _weights = _weights.double() sampler = torch.utils.data.sampler.WeightedRandomSampler(_weights, BATCH_SIZE) loader_train = DataLoader(train_set, batch_size=BATCH_SIZE,                           shuffle=False, sampler=sampler, num_workers=4) loader_val = DataLoader(val_set, batch_size=BATCH_SIZE,                         shuffle=False, sampler=sampler, num_workers=4) model = RNN(embeddings, num_classes=num_classes, **_hparams) model.cuda() criterion = torch.nn.CrossEntropyLoss() parameters = filter(lambda p: p.requires_grad, model.parameters()) optimizer = torch.optim.Adam(parameters)     # TRAIN ...
    class SentimentDataset(Dataset):
        def __init__(self, file, max_length, max_topic_length, word2idx, tword2idx, topic_bs):
        	...
		self.data = [SocialTokenizer(lowercase=True).tokenize(x)for x in self.data] 	        self.topics = [SocialTokenizer(lowercase=True).tokenize(x) for x in self.topics] 	        self.label_encoder = preprocessing.LabelEncoder() 	        self.label_encoder = self.label_encoder.fit(self.labels) 	        self.label_count = Counter(self.labels) 	        self.weights = [self.label_count['-2'], self.label_count['-1'], 	                        self.label_count['0'], self.label_count['1'], 	                        self.label_count['2']] 	        ...
    	def __getitem__(self, index):
   		        sample, label, topic = self.data[index], self.labels[index], self.topics[index]   File "/home/kostas/anaconda3/envs/pytorch_env/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 40, in <listcomp>     samples = collate_fn([dataset[i] for i in batch_indices])   File "/home/kostas/Gitlab/SemTest/models/datasets.py", line 108, in __getitem__     sample, label, topic = self.data[index], self.labels[index], self.topics[index] TypeError: list indices must be integers or slices, not torch.LongTensor read  16 min

## post by ptrblck on Dec 17, 2017
What kind of error do you get?

Here is a sample code, which should work fine:
numDataPoints = 1000 data_dim = 5 bs = 100

# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim) target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),                     np.ones(int(numDataPoints * 0.1), dtype=np.int32))) print 'target train 0/1: {}/{}'.format(     len(np.where(target == 0)[0]), len(np.where(target == 1)[0])) class_sample_count = np.array(     [len(np.where(target == t)[0]) for t in np.unique(target)]) weight = 1. / class_sample_count samples_weight = np.array([weight[t] for t in target]) samples_weight = torch.from_numpy(samples_weight) samples_weigth = samples_weight.double() sampler = WeightedRandomSampler(samples_weight, len(samples_weight)) target = torch.from_numpy(target).long() train_dataset = torch.utils.data.TensorDataset(data, target) train_loader = DataLoader(     train_dataset, batch_size=bs, num_workers=1, sampler=sampler) for i, (data, target) in enumerate(train_loader):
    print "batch index {}, 0/1: {}/{}".format(         i,         len(np.where(target.numpy() == 0)[0]),         len(np.where(target.numpy() == 1)[0]))

## post by Chahrazad on Dec 18, 2017
what is the interpretation of weight here?
> weight = 1. / class_sample_count

shouldn’t the weight be the class frequency ?
> weight = numDataPoints / class_sample_count

## post by kkorovesis on Dec 18, 2017
I am guessing that the problem is that my train_set consists of 6 data and 1 target, instead of 1 data and 1 target. In your examples you have only (data, target). So even if I had fixed weights they wouldn’t be multiplied with the correct data. In comments you can see what my train_set contains. I need all 6 inputs in my model therefore I can’t change that. Is there a way around it ?
train_set = SentimentDataset(file=TRAIN_DATA, word2idx=word2idx, tword2idx=tword2idx,                              max_length=0, max_topic_length=0, topic_bs=True)  # train_set: message, topic, len(self.data[index]), len(self.topics[index]), self.weights, index, label _weights = [0.8,0.8,0.3,0.4,0.8] sampler = torch.utils.data.sampler.WeightedRandomSampler(_weights, BATCH_SIZE) 3 months later

## post by jusjusjus on Mar 4, 2018
## post by ptrblck on Mar 5, 2018
9 months later

## post by surojit_sengupta on Nov 24, 2018
## post by ptrblck on Nov 24, 2018
## post by surojit_sengupta on Nov 27, 2018
## post by ptrblck on Nov 27, 2018
## Load more posts below

## 相关笔记

[深度学习与AI（主题索引）](../../../../index/MOC-dl-ai.md)
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/DataLoader-原理深解|DataLoader 原理深解]]
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/DataLoader-完全指南|DataLoader 完全指南]]
[[content/深度学习与AI/06 PyTorch/DataLoader与采样/Dataset-子集-Subset|Dataset 子集 Subset]]
