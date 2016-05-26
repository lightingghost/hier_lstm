import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
from copy import copy

LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
HyperPara = namedtuple('HyperPara', ['num_lstm_layer', 'seq_len', 'input_size',
                                     'num_hidden', 'num_embed', 'num_label', 
                                     'dropout'])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)
    

def lstm_encoder(data_wv, seq_len, num_lstm_layer, input_size, num_hidden, num_embed, dropout=0.):
    #multilayer lstm
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        with mx.AttrScope(ctx_group='enc_layers'):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("enc_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("enc_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("enc_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("enc_l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("enc_l%d_init_c" % i),
                              h=mx.sym.Variable("enc_l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    

        
    for seqidx in range(seq_len):
        hidden = data_wv[seqidx]
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            with mx.AttrScope(ctx_group='enc_layers'):
                next_state = lstm(num_hidden, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                hidden = next_state.h
                last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

    return last_states

def lstm_decoder(label_wv, in_lstm_state, num_lstm_layer, seq_len, num_hidden, num_label, dropout=0.):
    # pass the state             

    with mx.AttrScope(ctx_group='decode'):
        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")

    param_cells = []
    last_states = copy(in_lstm_state)
    for i in range(num_lstm_layer):
        with mx.AttrScope(ctx_group='dec_layers'):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("dec_l%d_i2h_weight" % i),
                                         i2h_bias=mx.sym.Variable("dec_l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable("dec_l%d_h2h_weight" % i),
                                         h2h_bias=mx.sym.Variable("dec_l%d_h2h_bias" % i)))
    hidden_all = []
    hidden = mx.sym.Variable('dec_start')
    for seqidx in range(seq_len):
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            with mx.AttrScope(ctx_group='dec_layers'):
                next_state = lstm(num_hidden, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                hidden = label_wv[seqidx]
                last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)
    with mx.AttrScope(ctx_group='decode'):                  
        hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
        pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                    weight=cls_weight, bias=cls_bias, name='pred')

    return pred
    
def seq_softmax(label, pred):
    with mx.AttrScope(ctx_group='loss'):
        label = mx.sym.transpose(data=label)
        label = mx.sym.Reshape(data=label, target_shape=(0,))
        sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return sm
     
def lstm_model(data_name, label_name, enc_para, dec_para):
    data = mx.sym.Variable(data_name)
    label = mx.sym.Variable(label_name)
    with mx.AttrScope(ctx_group='embed'):
        embed_weight = mx.sym.Variable("embed_weight")
    data_embed = mx.sym.Embedding(data=data, input_dim=enc_para.input_size, weight=embed_weight, 
                                 output_dim=enc_para.num_embed)
    data_wv = mx.sym.SliceChannel(data=data_embed, num_outputs=enc_para.seq_len, squeeze_axis=1)
        
    label_embed = mx.sym.Embedding(data=label, input_dim=enc_para.input_size, weight=embed_weight, 
                                 output_dim=enc_para.num_embed)
    label_wv = mx.sym.SliceChannel(data=label_embed, num_outputs=dec_para.seq_len, squeeze_axis=1)        
    enc_state = lstm_encoder(data_wv, enc_para.seq_len, enc_para.num_lstm_layer, enc_para.input_size,
                             enc_para.num_hidden, enc_para.num_embed, enc_para.dropout)
    pred = lstm_decoder(label_wv, enc_state, dec_para.num_lstm_layer, dec_para.seq_len,
                        dec_para.num_hidden, dec_para.num_label, dec_para.dropout)
    loss = seq_softmax(label, pred)
    return loss
    
def get_input_shapes(enc_para, dec_para, batch_size):
    init_state_shapes = {}
    for i in range(enc_para.num_lstm_layer):
        init_state_shapes['enc_l{}_init_h'.format(i)] = (batch_size, enc_para.num_hidden)
        init_state_shapes['enc_l{}_init_c'.format(i)] = (batch_size, enc_para.num_hidden) 
    init_state_shapes['dec_start'] = (batch_size, dec_para.num_hidden)   
    return init_state_shapes