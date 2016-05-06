import sys
import numpy as np
import mxnet as mx


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

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
        
class array_iter_with_init_states(mx.io.DataIter):
    def __init__(self, data, label, batch_size, init_states, 
                 data_name='data', label_name='label', random=False):
        super(array_iter_with_init_states, self).__init__()

        self.data = data
        self.label = label
        self.data_name = data_name
        self.label_name = label_name
        self.data_shape = data.shape
        self.label_shape = label.shape
        self.batch_size = batch_size
        self.random = random
        
        assert(data.shape[0] == label.shape[0])
        self.nsamples = data.shape[0]
        
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [(data_name, (self.batch_size, self.data_shape[1]))] + init_states
        self.provide_label = [(label_name, (self.batch_size, self.label_shape[1]))]

    def make_data_iter_plan(self):
        "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        n_batches = int(self.nsamples / self.batch_size)
        n_sum = n_batches * self.batch_size
        if self.random:
            gen_seq = np.random.permutation
        else:
            gen_seq = np.arange
        self.batches = np.array_split(gen_seq(n_sum, dtype=np.int32), n_batches)
        self.curr_batch_idx = 0
        

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for batch_idxs in self.batches:
            data = self.data[batch_idxs]
            label = self.label[batch_idxs]
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = [self.data_name] + init_state_names
            label_names = [self.label_name]
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        self.curr_batch_idx = 0