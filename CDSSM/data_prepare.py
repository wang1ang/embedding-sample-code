import tensorflow as tf
import numpy as np
import random
import sys

import codecs
import collections
import json
import math
import os
import sys
import time
import string

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2

def trilettergram():
    voc_dict = {}
    alphas = 'abcdefghijklmnopqrstuvwxyz '
    digits = '0123456789'
    alphadigit = alphas + digits

    voc_dict['<GO>'] = 0
    voc_dict['<END>'] = 1
    voc_dict['<UNK>'] = 2
    idx = 3
    # 36 ^ 3
    for c1 in alphadigit:
        for c2 in alphadigit:
            for c3 in alphadigit:
                voc_dict[''.join([c1, c2, c3])] = idx
                idx += 1
    # 36 ^ 2 * 2
    for c1 in alphadigit:
        for c2 in alphadigit:
            voc_dict[''.join(['#', c1, c2])] = idx
            idx += 1
            voc_dict[''.join([c1, c2, '#'])] = idx
            idx += 1
    # 36
    for c in alphadigit:
        voc_dict[''.join(['#', c, '#'])] = idx
        idx += 1
    print('vocabulary size=%s'%idx)
    return voc_dict

def sparse_embedding(text, voc_dict, max_words=128, max_letters=64):
    letterngram = 3
    norm = text.translate(str.maketrans('','', string.punctuation))
    words = list(filter(None, norm.split(' ')))[:max_words]
    indices = []
    values = []
    for i, word in enumerate(words):
        word = ('#' + word + '#')[:max_letters]
        for j in range(len(word) - letterngram + 1):
            token = word[j: j + letterngram]
            if token in voc_dict:
                indices.append([i, j]) # i-th word & j-th char
                values.append(voc_dict[token])
    return [indices, values, len(words)]

class DataFeeder:
    def __init__(self,         
            batch_size, input_filename,
            query_col, doc_col, 
            rawquery_col, rawurl_col,
            vocab,
            input_max_length, doc_max_length,
            skip_count = 0, 
            training = True):
        self.batch_size = batch_size
        self.input_filename = input_filename
        self.query_col = query_col
        self.doc_col = doc_col
        self.rawquery_col = rawquery_col
        self.rawurl_col = rawurl_col
        self.vocab = vocab
        self.input_max_length = input_max_length
        self.doc_max_length = doc_max_length
        self.skip_count = skip_count
        self.training = training
        self.buffer = dict()
        self.line_iter = self.get_line()
    def get_line(self):
        while True:
            print (f'reading {self.input_filename}')
            with open(self.input_filename, "r", encoding="utf-8") as finput:
                for _ in range(self.skip_count):
                        next(finput)
                for in_line in finput:
                    tokens = in_line.rstrip().split('\t')
                    query = tokens[self.query_col].lower()
                    doc = tokens[self.doc_col].lower()
                    url = tokens[self.rawurl_col]
                    data = {
                        'input': sparse_embedding(query, self.vocab, self.input_max_length),
                        'doc': sparse_embedding(doc, self.vocab, self.doc_max_length),
                        'url': url,
                        'query': query
                    }
                    yield data
    def combine_sparse(self, samples, max_length):
        indices_batch = []
        values_batch = []
        length_batch = []
        for i, [indices, values, length] in enumerate(samples):
            length_batch.append(length)
            for j, [row, col] in enumerate(indices):
                indices_batch.append([i * max_length + row, col])
                values_batch.append(values[j])
        batch_size = len(samples)
        # extend to 1 extra size to pad the SparseTensor in future step
        indices_batch.append([batch_size * max_length, 0])
        values_batch.append(END_TOKEN)

        return indices_batch, values_batch, length_batch

    def get_batch(self):
        while True:
            inputs, docs, input_length, doc_length = [], [], [], []
            queries, urls = [], []

            # fill buffer
            while len(self.buffer) < self.batch_size:
                rec = next(self.line_iter) # will raise StopIteration
                key = rec['url']
                if key in self.buffer:
                    self.buffer[key].append(rec)
                else:
                    self.buffer[key] = [rec]
            # get different urls from buffer
            for key in list(self.buffer):
                rec = self.buffer.pop(key)
                if len(rec) > 1:
                    self.buffer[key] = rec[1:] # put back
                rec = rec[0]
                inputs.append(rec['input'])
                docs.append(rec['doc'])
                queries.append(rec['query'])
                urls.append(rec['url'])
            assert len(inputs) == self.batch_size
            input_indices_batch, input_values_batch, input_length = self.combine_sparse(inputs, self.input_max_length)
            doc_indices_batch, doc_values_batch, doc_length = self.combine_sparse(docs, self.doc_max_length)
            yield {
                'input_indices': input_indices_batch,
                'input_values': input_values_batch,
                'doc_indices': doc_indices_batch,
                'doc_values': doc_values_batch,
                'input_length': input_length,
                'doc_length': doc_length,
                'query': queries,
                'url': urls
            }
    def data_tensor(self):
        dataset = tf.data.Dataset.from_generator(self.get_batch, {
            'input_indices': tf.int64,
            'input_values': tf.int32,
            'doc_indices': tf.int64,
            'doc_values': tf.int32,
            'input_length': tf.int32,
            'doc_length': tf.int32,
            'query': tf.string,
            'url': tf.string
        }, {
            'input_indices': tf.TensorShape([None, 2]),
            'input_values': tf.TensorShape([None]),
            'doc_indices': tf.TensorShape([None, 2]),
            'doc_values': tf.TensorShape([None]),
            'input_length': tf.TensorShape([None]),
            'doc_length': tf.TensorShape([None]),
            'query': tf.TensorShape([None]),
            'url': tf.TensorShape([None])
        })
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

if __name__ == '__main__':
    vocab = trilettergram()

    data_feeder = DataFeeder(
        128,
        'AllData_part1_2018_07_27_top1M_top4k.tsv',
        1,
        2,
        1,
        0,
        vocab, 16, 64,
        skip_count=0,
        training = False)

    one_query = data_feeder.data_tensor()
    with tf.Session() as sess:
        while True:
            try:
                data = sess.run(one_query)
                print (data['input_length'])
            except tf.errors.OutOfRangeError:
                print ('end!')
