import mxnet as mx
import os
import sys
import numpy as np
from hier_lstm import HyperPara, hier_lstm_model, get_input_shapes
from data_io import array_iter_with_init_states as array_iter
#setup logging
from imp import reload
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.DEBUG, datefmt='%I:%M:%S')

         
#model para
_dict_len       = 55496
_test           = False
_num_lstm_layer = 1
_input_size     = _dict_len + 2
_num_hidden     = 512
_num_embed      = 300
_num_label      = _dict_len + 2
_dropout        = 0.75
#opt para
_learning_rate  = 0.001
#training para
_devs           = [mx.gpu()]
_batch_size     = 1
_num_epoch      = 4

#data

name = 'test'
data_path = os.path.join('data', name + '_data.npy')
label_path = os.path.join('data', name + '_label.npy')
data = np.load(data_path)
label = np.load(label_path)
_nsamples = label.shape[0]



# #model

sent_enc_para = HyperPara(num_lstm_layer = _num_lstm_layer,
                          seq_len        = 100,
                          input_size     = _input_size,
                          num_hidden     = _num_hidden,
                          num_embed      = _num_embed,
                          num_label      = _num_label,
                          dropout        = _dropout)
doc_enc_para  = HyperPara(num_lstm_layer = _num_lstm_layer,
                          seq_len        = 3,
                          input_size     = _input_size,
                          num_hidden     = _num_hidden,
                          num_embed      = _num_embed,
                          num_label      = _num_label,
                          dropout        = _dropout)
dec_para      = HyperPara(num_lstm_layer = _num_lstm_layer,
                          seq_len        = 30,
                          input_size     = _input_size,
                          num_hidden     = _num_hidden,
                          num_embed      = _num_embed,
                          num_label      = _num_label,
                          dropout        = _dropout)
sym = hier_lstm_model(data_name, label_name, sent_enc_para, doc_enc_para, dec_para)
init_dict = get_input_shapes(sent_enc_para, doc_enc_para, dec_para, _batch_size)

data_idx = 1
data = data[1]
label = label[1]

print('Data loading complete.')


epoch = int(sys.argv[1])
_devs = [mx.gpu()]
if sys.argv[2] == 'cpu':
    _devs = [mx.cpu(i) for i in range(8)]
else:
    _devs = [mx.gpu()]    
checkpoint_path = os.path.join('checkpoint', 'auto_sum')
pretrained_model = mx.model.FeedForward.load(checkpoint_path, epoch)

print('Previous model load complete.')

data_name = 'data'
label_name = 'label'
data_dict = copy(init_dict)

init_dict[data_name] = (_batch_size, sent_enc_para.seq_len * 3)
init_dict[label_name] = (_batch_size, dec_para.seq_len)

model_exec = sym.simple_bind(ctx=_devs, **init_dict)
for key in model_exec.arg_dict.keys():
    if key in pretrained_model.arg_params:
        pretrained_model.arg_params[key].copyto(model_exec.arg_dict[key])
        

