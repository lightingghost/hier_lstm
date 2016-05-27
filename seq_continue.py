import mxnet as mx
import os
import numpy as np
from normal_lstm import HyperPara, lstm_model, get_input_shapes
from bucket_io import BucketDataIter
from data_io import array_iter_with_init_states as array_iter
#setup logging
from imp import reload
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.DEBUG, datefmt='%I:%M:%S')
                    
begin_epoch     = 6
#model para
_test           = False
_auto_bucketing = True
_use_pretrained = True
_dict_len       = 55496
_num_lstm_layer = 3
_input_size     = _dict_len + 3
_num_hidden     = 512
_num_embed      = 300
_num_label      = _dict_len + 3
_dropout        = 0.
#opt para
_learning_rate  = 0.001
#training para
_devs           = [mx.gpu()]
_batch_size     = 20
_num_epoch      = 100

#data

if _test:
    data_path = os.path.join('data', 'ndata1000.npy')
    label_path = os.path.join('data', 'label1000.npy')
else:
    name = 'val'
    data_path = os.path.join('data', 'normal_lstm', name + '_data.npy')
    label_path = os.path.join('data', 'normal_lstm', name + '_label.npy')

data = np.load(data_path)
label = np.load(label_path)
_nsamples = label.shape[0]

embed_path = os.path.join('data', 'embed.npy')
embed_weight = np.load(embed_path)
embed_weight = mx.nd.array(embed_weight)

print('Data loading complete.')
#model
                           
def sym_gen(seq_len):
    enc_para = HyperPara(num_lstm_layer = _num_lstm_layer,
                         seq_len        = seq_len,
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

    data_name = 'data'
    label_name = 'label'
    sym = lstm_model(data_name, label_name, enc_para, dec_para)
    return sym

#data iter  

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

if _auto_bucketing:
    data_iter = BucketDataIter(data, label, _batch_size, list(init_dict.items()))
    symbol = sym_gen
else:
    data_iter = array_iter(data, label, _batch_size, list(init_dict.items()),
                           data_name='data', label_name='label', random=False)
    symbol = sym_gen(300)



checkpoint_path = os.path.join('checkpoint0', 'auto_sum')
pretrained_model = mx.model.FeedForward.load(checkpoint_path, begin_epoch)


def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)
    

    
opt = mx.optimizer.Adam(learning_rate=_learning_rate)



model = mx.model.FeedForward(ctx         = _devs,
                             symbol      = symbol,
                             arg_params  = pretrained_model.arg_params,
                             aux_params  = pretrained_model.aux_params,
                             num_epoch   = _num_epoch,
                             begin_epoch = begin_epoch,
                             optimizer   = opt)
print('Previous model load complete.')

model.sym = None
model.sym_gen = symbol


model.fit(X                  = data_iter,
          eval_metric        = mx.metric.np(Perplexity),
          batch_end_callback = mx.callback.Speedometer(_batch_size, 20),
          epoch_end_callback = mx.callback.do_checkpoint(checkpoint_path))
