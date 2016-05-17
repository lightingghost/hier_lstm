import numpy as np
import json
import os
from itertools import chain
from collections import defaultdict
from statistics import mean

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer
from sumy.utils import get_stop_words

from pyrouge import Rouge155

_language = 'english'
_sent_count = 1


def basic_sum(file, test_ratio=0.10, israndom=True):
    # extract test files
    file_lines = file.read().splitlines()
    nsamples = len(file_lines)
    ntests = int(nsamples * test_ratio)
    if israndom:
        seq = np.random.permutation(nsamples)
    else:
        seq = np.arange(nsamples)
    
    # summerizer
    stemmer = Stemmer(_language)
    summarizer = Summarizer (stemmer)
    summarizer.stop_words = get_stop_words(_language)
    
    # rouge
    rouge = Rouge155()
    
    scores = defaultdict(list)
    for i in range(ntests):
        line = file_lines[seq[i]]
        sample = json.loads(line)
        content = sample['content']
        title = sample['title']
        ref_text = {'A': title}
        doc = ' '.join(content)
        parser = PlaintextParser.from_string(doc, Tokenizer(_language))
        sum_sents = summarizer(parser.document, _sent_count)
        if len(sum_sents) != _sent_count:
            continue
        summary = str(sum_sents[0])
        score = rouge.score_summary(summary, ref_text)
        for k, v in score.items():
            scores[k].append(v)
        print('{} / {} processed.'.format(i, ntests), end='\r')
    result = {}
    for k, v in scores.items():
        result[k] = mean(v)
    return result
        
if __name__ == '__main__':
    path = os.path.join('data', 'sample_short')
    file = open(path, 'r')
    result = basic_sum(file)
    file.close()
    result_path = os.path.join('result', 'baseline')
    outfile = open(result_path, 'w')
    json.dump(result, outfile)
    