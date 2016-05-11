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
_input_size     = _dict_len + 3
_num_hidden     = 512
_num_embed      = 300
_num_label      = _dict_len + 3
_dropout        = 0.75
#opt para
_learning_rate  = 0.001
#training para
_devs           = [mx.gpu()]
_batch_size     = 32
_num_epoch      = 4

#data

name = 'val'
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

init_dict = get_input_shapes(sent_enc_para, doc_enc_para, dec_para, _batch_size)

data_iter = array_iter(data, label, _batch_size, list(init_dict.items()),
                       data_name='data', label_name='label', random=False)

print('Data loading complete.')


epoch = int(sys.argv[1])
_devs = [mx.gpu()]
if sys.argv[2] == 'cpu':
    _devs = [mx.cpu(i) for i in range(8)]
if sys.argv[2] == 'gpu':
    _devs = [mx.gpu()]    
checkpoint_path = os.path.join('checkpoint', 'auto_sum')
pretrained_model = mx.model.FeedForward.load(checkpoint_path, epoch)


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)
    

    
opt = mx.optimizer.Adam(learning_rate=_learning_rate)



model = mx.model.FeedForward(ctx         = _devs,
                             symbol      = pretrained_model.symbol,
                             arg_params  = pretrained_model.arg_params,
                             aux_params  = pretrained_model.aux_params,
                             num_epoch   = _num_epoch,
                             begin_epoch = epoch,
                             optimizer   = opt)
print('Previous model load complete.')

y = pretrained_model.predict(X           = data_iter, 
                             num_batch   = 1)
import pdb; pdb.set_trace()                            
                             
# y = pretrained_model.score(X                  = data_iter, 
#                            eval_metric        = mx.metric.np(Perplexity),
#                            batch_end_callback = mx.callback.Speedometer(_batch_size, 50),
#                            num_batch          = None)

print(y)