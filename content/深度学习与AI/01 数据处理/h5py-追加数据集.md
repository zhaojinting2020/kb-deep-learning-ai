---
title: h5py 追加数据集
url: https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py?rq=1
fetch_source: agent_reach:jina
fetched_at: '2026-06-27T20:10:19+00:00'
polished_at: '2026-06-27T18:51:39+00:00'
wikilinks_unbulleted_at: '2026-06-28T05:48:28+00:00'
---

# h5py 追加数据集

I am looking for a possibility to append data to an existing dataset inside a `.h5` file using Python (`h5py`).

A short intro to my project: I try to train a CNN using medical image data. Because of the huge amount of data and heavy memory usage during the transformation of the data to NumPy arrays, I needed to split the "transformation" into a few data chunks: load and preprocess the first 100 medical images and save the NumPy arrays to hdf5 file, then load the next 100 datasets and append the existing `.h5` file, and so on.

Now, I tried to store the first 100 transformed NumPy arrays as follows:
import h5py
from LoadIPV import LoadIPV
X_train_data, Y_train_data, X_test_data, Y_test_data = LoadIPV()
with h5py.File('.\PreprocessedData.h5', 'w') as hf:
    hf.create_dataset("X_train", data=X_train_data, maxshape=(None, 512, 512, 9))
    hf.create_dataset("X_test", data=X_test_data, maxshape=(None, 512, 512, 9))
    hf.create_dataset("Y_train", data=Y_train_data, maxshape=(None, 512, 512, 1))
    hf.create_dataset("Y_test", data=Y_test_data, maxshape=(None, 512, 512, 1))
```
As can be seen, the transformed NumPy arrays are splitted into four different "groups" that are stored into the four `hdf5` datasets`[X_train, X_test, Y_train, Y_test]`. The `LoadIPV()` function performs the preprocessing of the medical image data.
My problem is that I would like to store the next 100 NumPy arrays into the same `.h5` file into the existing datasets: that means that I would like to append to, for example, the existing `X_train` dataset of shape `[100, 512, 512, 9]` with the next 100 NumPy arrays, such that `X_train` becomes of shape `[200, 512, 512, 9]`. The same should work for the other three datasets `X_test`, `Y_train` and `Y_test`.

## 2 Answers 2
I have found a solution that seems to work!
Have a look at this: [incremental writes to hdf5 with h5py](https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1)!
In order to append data to a specific dataset it is necessary to first resize the specific dataset in the corresponding axis and subsequently append the new data at the end of the "old" nparray.
Thus, the solution looks like this:
with h5py.File('.\PreprocessedData.h5', 'a') as hf:
    hf["X_train"].resize((hf["X_train"].shape[0] + X_train_data.shape[0]), axis = 0)
    hf["X_train"][-X_train_data.shape[0]:] = X_train_data
    hf["X_test"].resize((hf["X_test"].shape[0] + X_test_data.shape[0]), axis = 0)
    hf["X_test"][-X_test_data.shape[0]:] = X_test_data
    hf["Y_train"].resize((hf["Y_train"].shape[0] + Y_train_data.shape[0]), axis = 0)
    hf["Y_train"][-Y_train_data.shape[0]:] = Y_train_data
    hf["Y_test"].resize((hf["Y_test"].shape[0] + Y_test_data.shape[0]), axis = 0)
    hf["Y_test"][-Y_test_data.shape[0]:] = Y_test_data
```
However, note that you should create the dataset with `maxshape=(None,)`, for example
h5f.create_dataset('X_train', data=orig_data, compression="gzip", chunks=True, maxshape=(None,))
```
otherwise the dataset cannot be extended.
