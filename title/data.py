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
from tokenizer.simplifier import Simplifier
from tokenizer.tokenizer import Tokenizer
from tokenizer.lowercase_and_remove_accent import run_strip_accents

import sentencepiece as spm
class TextEmbedding(object):
    pass
    #def __init__(self):
class XlmEmbedding(TextEmbedding):
    def __init__(self):
        pass
    def get_embedding_fn(self, max_length=12):
        self.max_length = max_length
        self.s = Simplifier('tokenizer/zh_mapping.txt')
        self.t = Tokenizer('tokenizer/spiece_all_bpe/spiece.all.bpe.130000.lower.model',
            'tokenizer/lg.all.voc',
            max_length
        )
        self.max_length = max_length
        return self.embedding
    def embedding(self, text):
        simple = self.s.simplify(text)
        tokens = self.t.tokenize(simple)
        accents = run_strip_accents(tokens)
        ids = self.t.token_to_id(accents)
        return ids
    def size(self):
        return self.t.dico.counts
    @classmethod
    def get_feeder(cls):
        return DenseDataFeeder
class SentencePieceEmbedding(TextEmbedding):
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        #self.sp.load('mtier1_20w.model')
        #self.sp.load('mtier1_10w.model')
        self.sp.load(r'E:\embedding\title\top24_10w.model')
    def get_embedding_fn(self, max_length=128):
        self.max_length = max_length
        return self.embedding
    def embedding(self, text):
        values = self.sp.encode_as_ids(text)[:self.max_length]
        length = len(values)
        indices = [[i, 0] for i in range(length)]
        return [indices, values, length]
    def size(self):
        return self.sp.get_piece_size()
    @classmethod
    def get_feeder(cls):
        return SparseDataFeeder

class TriLetterEmbedding(TextEmbedding):
    GO_TOKEN = 0
    END_TOKEN = 1
    UNK_TOKEN = 2
    def __init__(self):
        self.voc_dict = self.trilettergram()
    def size(self):
        return len(self.voc_dict)
    #@staticmethod
    @classmethod
    def trilettergram(cls):
        voc_dict = {}
        alphas = 'abcdefghijklmnopqrstuvwxyz '
        digits = '0123456789'
        alphadigit = alphas + digits

        voc_dict['<GO>'] = cls.GO_TOKEN
        voc_dict['<END>'] = cls.END_TOKEN
        voc_dict['<UNK>'] = cls.UNK_TOKEN
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
    def lgram_embedding(self, text, max_length=128):
        letterngram = 3
        token = ''.join(['#', text.strip(), '#'])
        lgram = [token[i:i+letterngram] for i in range(len(token) - letterngram + 1)]
        embedding = [self.voc_dict[lg] for lg in lgram if lg in self.voc_dict][:max_length]
        length = len(embedding)
        embedding += [self.END_TOKEN] * (max_length - length)
        return [embedding, length]
    def get_embedding_fn(self, max_length):
        self.max_length = max_length
        return self.whash_embedding
    def whash_embedding(self, text, max_letters=64):
        letterngram = 3
        norm = text.translate(str.maketrans('','', string.punctuation))
        words = list(filter(None, norm.split(' ')))[:self.max_length]
        indices = []
        values = []
        for i, word in enumerate(words):
            word = ('#' + word + '#')[:max_letters]
            for j in range(len(word) - letterngram + 1):
                token = word[j: j + letterngram]
                if token in self.voc_dict:
                    indices.append([i, j]) # i-th word & j-th char
                    values.append(self.voc_dict[token])
        return [indices, values, len(words)]
    @classmethod
    def get_feeder(cls):
        return SparseDataFeeder

class DataFeeder(object):
    def __init__(self,         
            batch_size, input_filenames,
            doc_max_length,
            skip_count = 0,
            training = True
            ):
        self.batch_size = batch_size
        self.input_filenames = input_filenames
        self.doc_max_length = doc_max_length
        self.skip_count = skip_count
        self.training = training
        self.buffer = dict()
        self.get_batch = self.get_batch_train if training else self.get_batch_test

    def get_line(self, func_embedding, filenames):
        epoch = -1
        while True:
            epoch += 1
            print ('epoch {0}'.format(epoch))
            for filename in filenames:
                print (f'(epoch {epoch}) reading {filename}')
                with open(filename, "r", encoding="utf-8") as finput:
                    if filename == filenames[0]:
                        for _ in range(self.skip_count):
                                next(finput)
                    for in_line in finput:
                        tokens = in_line.rstrip().split('\t')
                        url = tokens[0]
                        doc = tokens[1].lower()
                        content = [float(x) for x in tokens[2].split(',')]
                        data = {
                            'url': url,
                            'doc': func_embedding(doc),
                            'content': content,
                            'rating': .0
                        }
                        if len(tokens) > 3:
                            data['rating'] = float(tokens[3])
                        yield data
    def make_batch(self):
        pass
    def poll_batch(self):
        batch_fetcher = [self.get_batch(it) for it in self.line_iter]
        if self.training:
            while True:
                for fetcher in batch_fetcher:
                    yield next(fetcher)
        else:
            while True:
                multiple_test = {'url': [], 'doc_ids': [], 'doc_mask': [], 'doc_type': [], 'content': [], 'rating': []}
                for fetcher in batch_fetcher:
                    test = next(fetcher)
                    for key in multiple_test:
                        multiple_test[key] += test[key]
                yield multiple_test
    def get_batch_train(self, dataset):
        while True:
            urls, docs, contents, ratings = [], [], [], []

            # fill buffer
            while len(self.buffer) < self.batch_size:
                rec = next(dataset) # will raise StopIteration
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
                ratings.append(rec['rating'])
            assert len(docs) == self.batch_size
            batch = self.make_batch(docs)
            batch['content'] = contents
            batch['url'] = urls
            batch['rating'] = ratings
            yield batch
    def get_batch_test(self, dataset):
        while True:
            urls, docs, contents, ratings = [], [], [], []
            for i in range(self.batch_size):
                rec = next(dataset)
                urls.append(rec['url'])
                docs.append(rec['doc'])
                contents.append(rec['content'])
                ratings.append(rec['rating'])
            batch = self.make_batch(docs)
            batch['content'] = contents
            batch['url'] = urls
            batch['rating'] = ratings
            yield batch

class DenseDataFeeder(DataFeeder):
    def __init__(self,         
            batch_size, input_filenames,
            doc_max_length,
            skip_count,
            embed_func,
            training = True
            ):
        DataFeeder.__init__(self, batch_size, input_filenames, doc_max_length, skip_count, training)
        self.embedding = embed_func()
        self.line_iter = [self.get_line(self.embedding.get_embedding_fn(doc_max_length), fns) for fns in self.input_filenames]
    def get_vocab_size(self):
        return self.embedding.size()
    def make_batch(self, docs):
        doc_ids = []
        doc_mask = []
        doc_type = []
        for ids in docs:
            l = len(ids)
            doc_ids.append(ids + [0] * (self.doc_max_length - l)) 
            doc_mask.append([1] * l + [0] * (self.doc_max_length - l))
            doc_type.append([0] * self.doc_max_length)
        return {
            'doc_ids': doc_ids,
            'doc_mask': doc_mask,
            'doc_type': doc_type,
        }
    def data_tensor(self):
        with tf.name_scope('sparse_data_tensor'):
            dataset = tf.data.Dataset.from_generator(self.poll_batch, {
                'url': tf.string,
                'doc_ids': tf.int32,
                'doc_mask': tf.int32,
                'doc_type': tf.int32,
                'content': tf.float32,
                'rating': tf.float32
            }, {
                'url': tf.TensorShape([None]),
                'doc_ids': tf.TensorShape([None, self.doc_max_length]),
                'doc_mask': tf.TensorShape([None, self.doc_max_length]),
                'doc_type': tf.TensorShape([None, self.doc_max_length]),
                'content': tf.TensorShape([None, 2048]),
                'rating': tf.TensorShape([None])
            })
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()

class SparseDataFeeder(DataFeeder):
    def __init__(self,         
            batch_size, input_filenames,
            doc_max_length,
            skip_count,
            embed_func,
            training = True
            ):
        DataFeeder.__init__(self, batch_size, input_filenames, doc_max_length, skip_count, training)
        self.embedding = embed_func()
        self.line_iter = [self.get_line(self.embedding.get_embedding_fn(doc_max_length), fns) for fns in self.input_filenames]
       
    def get_vocab_size(self):
        return self.embedding.size()
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
        values_batch.append(TriLetterEmbedding.END_TOKEN)

        return indices_batch, values_batch, length_batch
    def make_batch(self, docs):
        doc_indices_batch, doc_values_batch, doc_length = self.combine_sparse(docs, self.doc_max_length)
        return {
            'doc_indices': doc_indices_batch,
            'doc_values': doc_values_batch,
            'doc_length': doc_length,
        }
    def data_tensor(self):
        with tf.name_scope('sparse_data_tensor'):
            dataset = tf.data.Dataset.from_generator(self.poll_batch, {
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

from datafiles import get_files
if __name__ == '__main__':
    """
    files = get_files()
    files = files[:1]
    train_feeder = SparseDataFeeder(
        128,
        files,
        64,
        skip_count=0,
        embed_func=SentencePieceEmbedding,
        training = True)
    #batch_iter = train_feeder.poll_batch()
    #batch = next(batch_iter)
    #print (batch)
    print (train_feeder.get_vocab_size())
    """
    files = get_files(1)
    train_feeder = DenseDataFeeder(
        16,
        files,
        12,
        skip_count = 0,
        embed_func=XlmEmbedding,
        training = False
    )
    batch_iter = train_feeder.poll_batch()
    batch = next(batch_iter)
    print (batch)