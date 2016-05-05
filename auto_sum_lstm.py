import mxnet as mx
import os
import numpy as np
from hier_lstm import HyperPara, hier_lstm_model

#model para
_dict_len       = 55496
_test           = True
_num_lstm_layer = 1
_input_size     = _dict_len + 2
_num_hidden     = 256
_num_embed      = 300
_num_label      = _dict_len + 2
_dropout        = 0.75
#opt para
_learning_rate  = 0.002
#training para
_devs           = [mx.gpu()]
_batch_size     = 128
_num_epoch      = 20

#data

if _test:
    name = 'test'
else:
    name = 'training'
data_path = os.path.join('data', name + '_data.npy')
label_path = os.path.join('data', name + '_label.npy')
data = np.load(data_path)
label = np.load(label_path)
data_iter = mx.io.NDArrayIter(data              = data, 
                              label             = label, 
                              batch_size        = _batch_size, 
                              last_batch_handle = 'discard')
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

print('Model set up.')

#train
def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)
    
opt = mx.optimizer.Adam(learning_rate=_learning_rate)

pre_trained = {'embed_weight': embed_weight}
init = mx.initializer.Load(pre_trained,
                           default_init=mx.initializer.Xavier())
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
          batch_end_callback = mx.callback.Speedometer(_batch_size),
          epoch_end_callback = mx.callback.do_checkpoint(checkpoint_path))
