A Hierarchical Model for Text Autosummarization
===============================================

The mxnet implementation of the project of [A Hierarchical Model for Text Autosummarization](https://cs224d.stanford.edu/reports/zhenpeng.pdf)

# Abstract

Summarization is an important challange in natural language processing. Deep
learning methods, however, have not been widely used in text summarization,
although neural networks have been proved to be powerful in natural language
processing. In this paper, an encoder-decoder neural network model is applied to
text summarization, as an important step toward this task. Besides, a hierarchical
model, which builds the sentence representations and then paragraph representations,
enables the summarization for long documents.


# Usage

run
```python
python auto_sum_lstm.py
```
to train the model

run
```python
python validation.py
```
to evaluate the model on validation set
