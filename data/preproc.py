import json
import sys
import nltk
import pickle
from collections import defaultdict
import numpy as np

def convert(fin, fout, num=sys.maxsize):
    i = 1
    nparas = []
    nsents = []

    for line in fin:
        article = json.loads(line)
        title = article['title']
        content = article['content']
        paras = [para for para in content.split('\n') if len(para) >= 30]
        
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        nparas.append(len(paras))
        for para in paras:
            sents = sent_detector.tokenize(para)
            nsents.append(len(sents))
        #enc = json.dumps({'title': title, 'content': paras})
        #fout.write(enc + '\n')
        print('Converted: ' + str(i), end='\r')
        i += 1
        if i > num:
            break
            
def short_version(fin, fout, num=sys.maxsize):
    i = 1
    
    for line in fin:
        article = json.loads(line)
        title = article['title'].lower()
        content = article['content']
        paras = ''.join([para for para in content.split('\n') if len(para) >= 40])
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = [sent.lower() for sent in sent_detector.tokenize(paras)[:3]]
        
        enc = json.dumps({'title': title, 'content': sents})
        fout.write(enc + '\n')
        print('Converted: ' + str(i), end='\r')
        i += 1
        if i > num:
            break

def analyse(fin, num=sys.maxsize):
    i = 1
    nparas = []
    nsents = []

    for line in fin:
        article = json.loads(line)
        content = article['content']
        paras = [para for para in content.split('\n') if len(para) >= 30]
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        nparas.append(len(paras))
        for para in paras:
            sents = sent_detector.tokenize(para)
            nsents.append(len(sents))

        print('Analysed: ' + str(i), end='\r')
        i += 1
        if i > num:
            break
    return nparas, nsents
    
def build_vocab(fin, num=sys.maxsize, drop_off=50):
    import itertools
    i = 1
    word_count = defaultdict(int)
    title_count = defaultdict(int)
    for line in fin:
        article = json.loads(line)
        sents = article['content']
        title = article['title']
        for sent in sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                word_count[word] += 1
        for word in nltk.word_tokenize(title):
            title_count[word] += 1 

        print('Processed: ' + str(i), end='\r')
        i += 1
        if i > num:
            break
            
    idx2word = dict()
    word2idx = dict()
    idx = 1
    for k, v in title_count.items():
        if v > 10:
            word2idx[k] = idx
            idx2word[idx] = k
            idx += 1
    for k, v in word_count.items():
        if v > drop_off and not k in word2idx:
            word2idx[k] = idx
            idx2word[idx] = k
            idx += 1

    fout1 = open('word2idx', 'wb')
    fout2 = open('idx2word', 'wb')
    pickle.dump(word2idx, fout1)
    pickle.dump(idx2word, fout2)
    fout1.close()
    fout2.close()
    print('Number of word in the vocabulary: '+ str(len(word2idx)))        
        
    
def get_hist(data, figure, title, num_bins=10):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figure)
    n, bins, patches = plt.hist(np.array(data), num_bins, normed=1)
    plt.title(title)
    plt.show()
    
def trans_word2idx(fin, fout, drop_off=0.2):
    vocab_file = open('word2idx','rb')
    vocab = pickle.load(vocab_file)
    vocab = defaultdict(int, vocab)
    for line in fin:
        article = json.loads(line)
        title = article['title']
        content = article['content']
        title_trans = [vocab[word] for word in nltk.word_tokenize(title)]
        if float(title_trans.count(0) / len(title_trans)) > drop_off:
            continue
        title_trans.append(-1)
        content_trans = []
        add = True
        for sent in content:
            sent_trans = [vocab[word] for word in nltk.word_tokenize(sent)]
            if float(sent_trans.count(0) / len(sent_trans)) > drop_off:
                add = False
            sent_trans.append(-1)
            content_trans.append(sent_trans)
        if not add:
            continue
        enc = json.dumps({'title': title_trans, 'content': content_trans})
        fout.write(enc + '\n')
        
def trans_idx2word(fin, fout):
    vocab_file = open('idx2word','rb')
    vocab = pickle.load(vocab_file)
    vocab = defaultdict(str, vocab)
    for line in fin:
        article = json.loads(line)
        title = article['title']
        content = article['content']
        title_trans = [vocab[idx] for idx in title]
        content_trans = []
        for sent in content:
            content_trans.append([vocab[idx] for idx in sent])
        enc = json.dumps({'title': title_trans, 'content': content_trans})
        fout.write(enc + '\n')

def add_sent_term(fin, fout):
    for line in fin:
        article = json.loads(line)
        title = article['title']
        content = article['content']
        title.append(-1)
        for sent in content:
            sent.append(-1)
        enc = json.dumps({'title': title, 'content': content})
        fout.write(enc + '\n')
       
def trans2npy(fin, max_sent_len=100, max_title_len=30):
    data = []
    label = []
    for line in fin:
        sample = json.loads(line)
        title = sample['title']
        content = sample['content']
        add = True
        if len(title) > max_title_len:
            add = False
        for sent in content:
            if len(sent) > max_sent_len:
                add = False
        if add:
            data.append(content)
            label.append(title)
            
    print('Converting to npy file')
    
    assert(len(data) == len(label))
    nsamples = len(data)
    n_sent = 3
    data1 = np.zeros((nsamples, max_sent_len * n_sent), dtype=np.int32)
    data1.fill(-1)
    label1 = np.zeros((nsamples, max_title_len), dtype=np.int32)
    label1.fill(-1)
    for i in range(nsamples):
        j = 0
        for sent in data[i]:
            sent_len = len(sent)
            start_idx = j * max_sent_len
            data1[i, start_idx : start_idx + sent_len] = sent
            j += 1
        title_len = len(label[i])
        label1[i, :title_len] = label[i]
        
    print('Saving')
    
    np.save('data.npy', data1)
    np.save('label.npy', label1)
                

if __name__ == '__main__':
    fin = open('sample_short', 'r')
    fout = open('sample_short_trans', 'w')
    # short_version(fin, fout)
    # build_vocab(fin)
    # get_hist(nparas, 1, 'Number of Paragraphs', 20)
    # get_hist(nsents, 2, 'Number of Sentences', 20)
    trans_word2idx(fin, fout, 0.15)
    # trans_idx2word(fin, fout)
    # add_sent_term(fin, fout)
    # trans2npy(fin)
    fin.close()
    fout.close()