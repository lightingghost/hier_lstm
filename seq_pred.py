import mxnet as mx
import os
import sys
import numpy as np
from normal_lstm import HyperPara, lstm_model, get_input_shapes
from copy import copy
import json



class Model:
    def __init__(self, symbol, init_paras, pretrained, idx2word, ctx=mx.cpu()):
        self.idx2word = idx2word
        state_dict = copy(init_paras)

        init_paras['data'] = (1, 300)
        init_paras['label'] = (1, 30)

        self.model_exec = symbol.simple_bind(ctx=mx.cpu(), **init_paras)

        for key in self.model_exec.arg_dict.keys():
            if key in pretrained.arg_params:
                pretrained.arg_params[key].copyto(self.model_exec.arg_dict[key])
        for name, shape in state_dict.items():
            mx.nd.zeros(shape).copyto(self.model_exec.arg_dict[name])

    def predict(self, data, label):
        mx.nd.array(data).copyto(self.model_exec.arg_dict['data'])
        mx.nd.array(label).copyto(self.model_exec.arg_dict['label'])
        self.model_exec.forward()
        
        prob = self.model_exec.outputs[0].asnumpy()
        idxs = np.argmax(prob, axis=1)
        
        # import pdb; pdb.set_trace()
        pred = [self.idx2word[str(i)] for i in idxs if str(i) in self.idx2word]
        
        return pred

def predict(epoch):         
    #model para
    _dict_len       = 55496
    _test           = False
    _num_lstm_layer = 3
    _input_size     = _dict_len + 3
    _num_hidden     = 512
    _num_embed      = 300
    _num_label      = _dict_len + 3
    _dropout        = 0.5
    #opt para
    _learning_rate  = 0.001
    #training para
    _devs           = [mx.cpu()]
    _batch_size     = 1
    _num_epoch      = 4


    #data
    data_name = 'data'
    label_name = 'label'
    name = 'val'
    data_path = os.path.join('data', 'normal_lstm', name + '_data.npy')
    label_path = os.path.join('data', 'normal_lstm', name + '_label.npy')
    # data_path = os.path.join('data', 'nndata1000.npy')
    # label_path = os.path.join('data', 'label1000.npy')
    data = np.load(data_path)
    label = np.load(label_path)
    _nsamples = label.shape[0]


    # #model

    enc_para = HyperPara(num_lstm_layer = _num_lstm_layer,
                         seq_len        = 300,
                         input_size     = _input_size,
                         num_hidden     = _num_hidden,
                         num_embed      = _num_embed,
                         num_label      = None,
                         dropout        = _dropout)
    dec_para = HyperPara(num_lstm_layer = _num_lstm_layer,
                         seq_len        = 30,
                         input_size     = None,
                         num_hidden     = _num_hidden,
                         num_embed      = None,
                         num_label      = _num_label,
                         dropout        = _dropout)

    init_dict = get_input_shapes(enc_para, dec_para, _batch_size)
    sym = lstm_model(data_name, label_name, enc_para, dec_para)

    checkpoint_path = os.path.join('checkpoint0', 'auto_sum')
    pretrained_model = mx.model.FeedForward.load(checkpoint_path, epoch)
    
    file = open('data/idx2word', 'r')
    idx2word = json.load(file)
    file.close()
    
    pre_model = Model(sym, init_dict, pretrained_model, idx2word)
    
    print('Previous model load complete.')
    
    data_lens = np.argwhere(data == -1)[:, 1]
    for i in range(self.nsamples):
        data[i, data_lens[i]] = 0
    

    for i in range(20):
        t_data = data[i, :].reshape((1, 300))
        t_label = label[i, :].reshape((1, 30))
        print(i)
        print(pre_model.predict(t_data, t_label))
        print([idx2word[str(i)] for i in t_label[0] if str(i) in idx2word])
            
if __name__ == '__main__':
    predict(18)
