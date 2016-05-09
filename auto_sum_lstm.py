import mxnet as mx
import os
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
_dropout        = 0.
#opt para
_learning_rate  = 0.0005
#training para
_devs           = [mx.gpu()]
_batch_size     = 32
_num_epoch      = 1

#data

if _test:
    name = 'test'
else:
    name = 'training'
data_path = os.path.join('data', name + '_data.npy')
label_path = os.path.join('data', name + '_label.npy')
data = np.load(data_path)
label = np.load(label_path)
_nsamples = label.shape[0]

embed_path = os.path.join('data', 'embed.npy')
embed_weight = np.load(embed_path)
embed_weight = mx.nd.array(embed_weight)

print('Data loading complete.')
#model

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


data_name = 'data'
label_name = 'label'
sym = hier_lstm_model(data_name, label_name, sent_enc_para, doc_enc_para, dec_para)

print('Model set up complete.')


#data iter
input_dict = {'data': data}
init_dict = get_input_shapes(sent_enc_para, doc_enc_para, dec_para, _batch_size)


# for state, shape in init_dict.items():
#     input_dict[state] = mx.nd.zeros((_nsamples, _num_hidden))    
# data_iter = mx.io.NDArrayIter(data              = input_dict, 
#                               label             = label, 
#                               batch_size        = _batch_size, 
#                               last_batch_handle = 'discard')


data_iter = array_iter(data, label, _batch_size, list(init_dict.items()),
                       data_name='data', label_name='label', random=False)
                          
#train
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)
    
# def Perplexity(label, pred):
#     return np.exp(np.mean(pred))
    
opt = mx.optimizer.Adam(learning_rate=_learning_rate)

pre_trained = {'embed_weight': embed_weight}
init = mx.initializer.Load(pre_trained,
                           default_init=mx.initializer.Xavier(magnitude=0.5))
group2ctx = {'embed'      : mx.cpu(0),
             'preproc'    : mx.cpu(1),
             'sent_layers': mx.gpu(),
             'doc_layers' : mx.gpu(),
             'dec_layers' : mx.gpu(),
             'loss'       : mx.gpu()}
model = mx.model.FeedForward(ctx         = _devs,
                             symbol      = sym,
                             num_epoch   = _num_epoch,
                             optimizer   = opt,
                             initializer = init)
checkpoint_path = os.path.join('checkpoint', 'auto_sum')

model.fit(X                  = data_iter,
          eval_metric        = mx.metric.np(Perplexity),
          batch_end_callback = mx.callback.Speedometer(_batch_size, 2000),
          epoch_end_callback = mx.callback.do_checkpoint(checkpoint_path))
