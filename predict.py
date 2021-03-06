import mxnet as mx
import os
import sys
import numpy as np
from hier_lstm_pred import HyperPara, hier_lstm_model, get_input_shapes
from copy import copy
import json

file = open('data/idx2word', 'r')
idx2word = json.load(file)

def predict(epoch, data_idx):         
    #model para
    _dict_len       = 55496
    _test           = False
    _num_lstm_layer = 1
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
    name = 'training'
    # data_path = os.path.join('data', name + '_data.npy')
    # label_path = os.path.join('data', name + '_label.npy')
    data_path = os.path.join('data', 'data1000.npy')
    label_path = os.path.join('data', 'label1000.npy')
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

    
    data = data[data_idx, :].reshape((1, 300))
    label = label[data_idx, :].reshape((1, 30))



    print('Data loading complete.')


    
    checkpoint_path = os.path.join('checkpoint0', 'auto_sum')
    pretrained_model = mx.model.FeedForward.load(checkpoint_path, epoch)

    print('Previous model load complete.')


    state_dict = copy(init_dict)

    init_dict[data_name] = (_batch_size, sent_enc_para.seq_len * 3)
    init_dict[label_name] = (_batch_size, dec_para.seq_len)

    model_exec = sym.simple_bind(ctx=mx.gpu(), **init_dict)

    for key in model_exec.arg_dict.keys():
        if key in pretrained_model.arg_params:
            pretrained_model.arg_params[key].copyto(model_exec.arg_dict[key])
    for name, shape in state_dict.items():
        mx.nd.zeros(shape).copyto(model_exec.arg_dict[name])

    mx.nd.array(data).copyto(model_exec.arg_dict[data_name])
    mx.nd.array(label).copyto(model_exec.arg_dict[label_name])
    model_exec.forward()
    out = model_exec.outputs
    prob = model_exec.outputs[0].asnumpy()
    


    # # import pdb; pdb.set_trace()
    idxs = np.argmax(prob, axis=1)
    
    pre = [idx2word[str(i)] for i in idxs if str(i) in idx2word]
    orig = [idx2word[str(i)] for i in label[0] if str(i) in idx2word]
    # import pdb; pdb.set_trace()
    print(pre)
    print(orig)
    return out[1].asnumpy()
            
if __name__ == '__main__':
    # epoch = int(sys.argv[1])
    # data_idx = int(sys.argv[2])  
    # result = predict(epoch, data_idx)
    result = np.zeros((20, 512))
    for i in range(20):
        result[i] = predict(40, i)
    import pdb; pdb.set_trace()
