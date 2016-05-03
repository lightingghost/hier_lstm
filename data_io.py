import sys
import numpy as np
import mxnet as mx

def default_read_content(path, sent_len, title_len):
    import json
    data = []
    label = []
    with open(path) as ins:
        for line in ins:
            sample = json.loads(line)
            title = sample['title']
            content = sample['content']
            add = True
            if len(title) > title_len:
                add = False
            for sent in content:
                if len(sent) > sent_len:
                    add = False
            if add:
                data.append(content)
                label.append(title)
    return data, label

def default_gen_bucket(data_batch, label_batch):
    data_max_len = [0, 0, 0]
    for data in data_batch:
        for i in range(3):
            max_len[i] = max(max_len[i], len(data[i]))
    
    label_max_len = 0
    for label in label_batch:
        max_label_len = max(label_max_len, len(label))
        
    return data_max_len, label_max_len

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
        
class BucketSentenceIter(mx.io.DataIter):
    def __init__(self, path, batch_size, init_states, 
                 sent_len=100, n_sent=3, title_len=30, data_name='data', label_name='label'):
        super(BucketSentenceIter, self).__init__()

        data, label = default_read_content(path, sent_len, title_len)
        self.data = data
        self.label = label
        self.data_name = data_name
        self.label_name = label_name
        self.sent_len = sent_len
        self.n_sent = n_sent
        self.title_len = title_len
        self.batch_size = batch_size
        
        assert(len(data) == len(label))
        self.nsamples = len(data)
        
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (self.batch_size, self.sent_len))] + init_states
        self.provide_label = [('label', (self.batch_size, self.title_len))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        data = np.zeros((self.nsamples, self.sent_len * self.n_sent), dtype=np.int32)
        data.fill(-1)
        label = np.zeros((self.nsamples, self.title_len), dtype=np.int32)
        label.fill(-1)
        for i in range(self.nsamples):
            for j in range(self.n_sent):
                sent_len = len(self.data[i][j])
                start_idx = j * self.sent_len
                data[i, start_idx : start_idx + sent_len] = self.data[i][j]
            title_len = len(label[i])
            label[i, :title_len] = self.label[i]
        self.data = data
        self.label = label
        n_batches = int(self.nsamples / self.batch_size)
        n_sum = n_batches * self.batch_size
        self.batches = np.array_split(np.random.permutation(n_sum), n_batches)
        self.curr_batch_idx = 0
        

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for batch_idxs in self.batches:
            yield self.data[batch_idxs]

    def reset(self):
        self.curr_batch_idx = 0