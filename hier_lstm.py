import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])
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
    

def sentence_lstm(indata, num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):
    #indata: list of indexs of the words.
    embed_weight = mx.sym.Variable("embed_weight")

    #multilayer lstm
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("sent_l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("sent_l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("sent_l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("sent_l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("sent_l%d_init_c" % i),
                          h=mx.sym.Variable("sent_l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    embed = mx.sym.Embedding(data=indata, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)

    return last_states[-1]
    
def document_lstm(indata, num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):
                
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("doc_l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("doc_l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("doc_l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("doc_l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("doc_l%d_init_c" % i),
                          h=mx.sym.Variable("doc_l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    
    for seqidx in range(seq_len):
        hidden = indata[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                                prev_state=last_states[i],
                                param=param_cells[i],
                                seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
            
    return last_states[-1]
    
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    return last_states[-1]

def hier_lstm(indata, level1_para, level2_para, num_sent=3):
    para = mx.sym.SliceChannel(data=indata, num_outputs=num_sent)
    sentence_vecs = [] 
       for sentence in para:
        vec = sentence_lstm(sentence, level1_para.num_lstm_layer, level1_para.seq_len, 
                          level1_para.input_size, level1_para.num_hidden, level1_para.num_embed, 
                          level1_para.num_label, level1_para.dropout)
        sentence_vecs.append(vec.h)
    
    final_state = document_lstm(sentence, level2_para.num_lstm_layer, level2_para.seq_len, 
                          level2_para.input_size, level2_para.num_hidden, level2_para.num_embed, 
                          level2_para.num_label, level2_para.dropout)
    return final_state
    

def lstm_decoder(in_lstm_state, num_lstm_layer, input_size,
                 num_hidden, num_embed, num_label, dropout=0.):
                 
    embed_weight=mx.sym.Variable("embed_weight")

    param_cells = []
    last_states = [in_lstm_state] * num_lstm_layer
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("dec_l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("dec_l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("dec_l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("dec_l%d_h2h_bias" % i)))

    hidden_all = []
    hidden = in_lstm_state.h

    for seqidx in range(seq_len):
        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

                               
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    return pred
    
def seq_cross_entropy(label, pred):

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, target_shape=(0,))
    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')
    return sm   





        

    

