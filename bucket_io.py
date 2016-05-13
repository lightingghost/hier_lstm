import sys
import numpy as np
import mxnet as mx
from collections import defaultdict

def default_gen_buckets(sentences, lens, batch_size):
    len_dict = defaultdict(int)
    max_len = lens.max()
    for length in lens:
        len_dict[length] += 1

    tl = 0
    buckets = []
    for l, n in len_dict.items(): # TODO: There are better heuristic ways to do this    
        if n + tl >= batch_size:
            buckets.append(l)
            tl = 0
        else:
            tl += n
    if tl > 0:
        buckets.append(max_len)
    return buckets


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

class BucketLabelIter(mx.io.DataIter):
    def __init__(self, data, label, buckets, batch_size,
                 init_states, data_name='data', label_name='label'):
        super(BucketLabelIter, self).__init__()

        self.data_dim = data.shape[1]
        assert(data.shape[0] == label.shape[0])
        self.nsamples = label.shape[0]
        
        label_lens = np.argwhere(label == 1)[:, 1] + 1
        if len(buckets) == 0:
            buckets = default_gen_buckets(label, label_lens, batch_size)

        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.bucket_lens = buckets
        self.idx_buckets = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        for idx in range(self.nsamples):
            for i, bkt in enumerate(buckets):
                if bkt >= label_lens[idx]:
                    self.idx_buckets[i].append(idx)
                    break
            # we just ignore the sentence it is longer than the maximum
            # bucket size here

        # convert data into ndarrays for better speed during training
        nlabel = []
        ndata = []
        for i, x in enumerate(self.idx_buckets):
            ndata.append(data[x])
            nlabel.append(label[x, :buckets[i]])
        self.data = ndata
        self.label = nlabel
        
        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.idx_buckets]

        print("Summary of dataset ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, (batch_size, self.data_dim))] + init_states
        self.provide_label = [(label_name, (self.batch_size, self.default_bucket_key))]
        
        

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i in range(len(self.idx_buckets)):
            bucket_n_batches.append(int(len(self.idx_buckets[i]) / self.batch_size))
            self.data[i] = self.data[i][:int(bucket_n_batches[i]*self.batch_size)]
            self.label[i] = self.label[i][:int(bucket_n_batches[i]*self.batch_size)]

        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        np.random.shuffle(bucket_plan)

        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]
        


    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:

            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size
            data = self.data[i_bucket][idx]
            label = self.label[i_bucket][idx]
            

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = [self.data_name] + init_state_names
            label_names = [self.label_name]

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.bucket_lens[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
