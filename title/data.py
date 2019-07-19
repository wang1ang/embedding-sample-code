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

def lgram_embedding(text, voc_dict, max_length=128):
    letterngram = 3
    token = ''.join(['#', text.strip(), '#'])
    lgram = [token[i:i+letterngram] for i in range(len(token) - letterngram + 1)]
    embedding = [voc_dict[lg] for lg in lgram if lg in voc_dict][:max_length]
    length = len(embedding)
    embedding += [END_TOKEN] * (max_length - length)
    return [embedding, length]

def whash_embedding(text, voc_dict, max_words=128, max_letters=64):
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
            batch_size, input_filenames,
            vocab,
            doc_max_length,
            skip_count = 0,
            training = True
            ):
        self.batch_size = batch_size
        self.input_filenames = input_filenames
        self.vocab = vocab
        self.doc_max_length = doc_max_length
        self.skip_count = skip_count
        self.training = training
        self.buffer = dict()
        self.line_iter = self.get_line(whash_embedding)
        self.fn_batch = self.get_batch if training else self.get_batch_test
    def get_line(self, func_embedding):
        epoch = 0
        while True:
            print ('epoch {0}'.format(epoch))
            epoch += 1
            for input_filename in self.input_filenames:
                print (f'reading {input_filename}')
                with open(input_filename, "r", encoding="utf-8") as finput:
                    for _ in range(self.skip_count):
                            next(finput)
                    for in_line in finput:
                        tokens = in_line.rstrip().split('\t')
                        url = tokens[0]
                        doc = tokens[1].lower()
                        content = [float(x) for x in tokens[2].split(',')]
                        data = {
                            'url': url,
                            'doc': func_embedding(doc, self.vocab, self.doc_max_length),
                            'content': content
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
    def make_sparse_batch(self, docs):
        doc_indices_batch, doc_values_batch, doc_length = self.combine_sparse(docs, self.doc_max_length)
        return {
            'doc_indices': doc_indices_batch,
            'doc_values': doc_values_batch,
            'doc_length': doc_length,
        }
    def get_batch(self):
        while True:
            urls, docs, contents = [], [], []

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
                urls.append(rec['url'])
                docs.append(rec['doc'])
                contents.append(rec['content'])
            assert len(docs) == self.batch_size
            batch = self.make_sparse_batch(docs)
            batch['content'] = contents
            batch['url'] = urls
            yield batch
    def get_batch_test(self):
        while True:
            urls, docs, contents = [], [], []
            for i in range(self.batch_size):
                rec = next(self.line_iter)
                urls.append(rec['url'])
                docs.append(rec['doc'])
                contents.append(rec['content'])
            batch = self.make_sparse_batch(docs)
            batch['content'] = contents
            batch['url'] = urls
            yield batch

    def sparse_data_tensor(self):
        with tf.name_scope('sparse_data_tensor'):
            dataset = tf.data.Dataset.from_generator(self.fn_batch, {
                'url': tf.string,
                'doc_indices': tf.int64,
                'doc_values': tf.int32,
                'doc_length': tf.int32,
                'content': tf.float32
            }, {
                'url': tf.TensorShape([None]),
                'doc_indices': tf.TensorShape([None, 2]),
                'doc_values': tf.TensorShape([None]),
                'doc_length': tf.TensorShape([None]),
                'content': tf.TensorShape([None, 2048])
            })
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
