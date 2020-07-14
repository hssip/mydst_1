#coding=utf8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import codecs
from config import *
from utils import *


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, batch_size, epoch=1, mode='train'):
        self.batch_size = batch_size
        self.epoch = epoch
        self.mode = mode

        self._examples_num = 0
        self._labels_num = 0
        self._word2idx = {'utterances': {}, 'domain': {}, 'gate':{}, 'slots':{}}
        self._idx2word = {'utterances': {}, 'domain': {}, 'gate':{}, 'slots':{}}

        for w in builtin_words:
            self._word2idx['utterances'][w] = len(self._word2idx['utterances'])
            self._idx2word['utterances'][len(self._idx2word['utterances'])] = w

        self._cur_epoch_idx = 0

    def get_examples_num(self):
        return self._examples_num

    def get_cur_epoch_idx(self):
        return self._cur_epoch_idx

    def _add_vocab(self, field_name, word_list):
        assert isinstance(word_list, list), 'training data field is not list'

        if field_name not in self._word2idx:
            self._word2idx[field_name] = {}
            self._idx2word[field_name] = {}

        w2i = self._word2idx[field_name]
        i2w = self._idx2word[field_name]
        for w in word_list:
            if w not in w2i:
                w2i[w] = len(w2i)
                i2w[len(i2w)] = w

    def get_vocab_size(self, field_name):
        assert field_name in self._word2idx, 'error field name in vacab: %s' % field_name
        return len(self._word2idx[field_name])

    def _fill_batch_data(self, insts, bias=False):
        pad_idx = self._word2idx['utterances']['PAD']
        #print (insts)
        max_len = max(len(inst[0]) for inst in insts)
        sens_word = np.array([ [i for i in inst[0]]  + [pad_idx] * (max_len - len(inst[0])) for inst in insts])
        #sens_tag = np.array([ [i+1 for i in inst[1]] + [pad_idx] * (max_len - len(inst[1])) for inst in insts])
        sens_intent = np.array([inst[1] for inst in insts]).reshape(-1, 1)

        sens_mask = np.array([ [1] * len(inst[0]) + [0] * (max_len - len(inst[0])) for inst in insts])
        #print (sens_word, sens_tag, sens_intent)

        sens_gate = np.array([[i for i in inst[2]]  for inst in insts])
        sens_slot = np.array([[i for i in inst[3]] for inst in insts])

        sens_mask1 = np.array([ [0.0] * len(inst[0]) + [-np.inf] * (max_len - len(inst[0])) for inst in insts]).astype('float32')
        
        # max_res_len = 10
        sens_generate = np.array([[i for i in inst[4]] for inst in insts]).reshape(args['batch_size'], -1)

        all_word = np.array([i for i in range(self.get_vocab_size('utterances'))])
        '''
        for s in sens_word:
            for w in s:
                print (w)
        '''
        return all_word, sens_word, sens_mask, sens_intent, sens_gate, sens_slot, sens_mask1, sens_generate

    def get_examples(self):
        examples = []
        with codecs.open(args['train_file'], 'r', encoding='utf8') as fin:
            dial_list_json = json.load(fin)
            # a full session
            for dial_json in dial_list_json:
                dialogue = dial_json['dialogue']
                uttr_list = []
                # one turn with 1 system utterrance and 1 user utterrance
                for turn in dialogue:
                    sys_uttr = turn['system_transcript'].strip().split()
                    user_uttr = turn['transcript'].strip().split()
                    domain = turn['domain']
                    uttr_list += sys_uttr + ['SEP-SYS'] + user_uttr + ['SEP-UTTR']
                    gates, generates = belief2gate_generate(turn['belief_state'])
                    slots = raw_slots
                    examples.append({'utterances': uttr_list + ['EOS'], 'domain': [domain], 'gate':gates, 'slot':slots, 'generate':generates })
                    self._add_vocab('utterances', sys_uttr + user_uttr)
                    self._add_vocab('domain', [domain])
                    self._add_vocab('gate', gates)
                    self._add_vocab('slot', slots)
                    self._add_vocab('utterances', generates)
                #print [sample_json['word_nums'], sample_json['istags'], [sample_json['intent_num']]]
                #np_example = np.array([sample_json['word_nums'], sample_json['istags'], [sample_json['intent_num']]])
                #print np_example.shape
        self._examples_num = len(examples)
        print ('load examples finish, examples_num = %d' % self._examples_num)

        examples_idx = []
        for e in examples:
            uttr = e['utterances']
            domain = e['domain']
            gates = e['gate']
            slots = e['slot']
            generates = e['generate']
            uttr_idx = self.word2idx('utterances', uttr)
            domain_idx = self.word2idx('domain', domain)
            gate_idx = self.word2idx('gate', gates)
            slot_idx = self.word2idx('slot', slots)
            generate_idx = self.word2idx('utterances', generates)
            # context_len = [len(uttr_idx)]
            fout = open('tokens.txt', mode = 'a+', encoding='utf8')
            stra = str(e) + '\n'
            fout.write(stra)
            fout.close()
            examples_idx.append({ 'utterances': uttr_idx, 'domain': domain_idx, 'gate':gate_idx, 'slot':slot_idx, 'generate': generate_idx})


        self._print_examples(examples, examples_idx)

        #for key, values in self._vocab.items():

        #    values_str = [str(v) for v in values]
        #    print ('%s: size=%d, value_list: %s' % (key, len(values), ';'.join(values_str)))
        return examples, examples_idx

    def word2idx(self, field_name, word_list):
        assert field_name in self._word2idx, 'word2idx field_name error: %s' % field_name
        assert type(word_list) == list, 'list type error in word2idx' 

        idx_list = []

        for w in word_list:
            if field_name == 'domain':
                idx_list.append(self._word2idx['domain'][w])
            elif field_name == 'utterances' or field_name == 'generate':
                idx_list.append(self._word2idx['utterances'][w] if w in self._word2idx['utterances'] else self._word2idx['utterances']['UNK'])
            elif field_name == 'gate':
                idx_list.append(self._word2idx['gate'][w])
            elif field_name == 'slot':
                idx_list.append(self._word2idx['slot'][w])
        return idx_list

    def _print_examples(self, examples, examples_idx):
        print_file = '%s/%s.txt' % (args['data_path'], self.mode)
        with codecs.open(print_file, 'w', encoding='utf8') as fout:
            for e in examples:
                fout.write(json.dumps(e) + '\n')

        print_file_idx = '%s/%s_idx.txt' % (args['data_path'], self.mode)
        with codecs.open(print_file_idx, 'w', encoding='utf8') as fout:
            for e in examples_idx:
                fout.write(json.dumps(e) + '\n')

        vocab_file = '%s/%s_vocab.txt' % (args['data_path'], self.mode)
        with codecs.open(vocab_file, 'w', encoding='utf8') as fout:
            json.dump(self._word2idx, fout)


    def data_generator(self, mode='train'):
        examples, examples_idx = self.get_examples() 
        self._examples_num = len(examples)

        def instance_reader():
            for epoch_index in range(self.epoch):
                self._cur_epoch_idx = epoch_index
                np.random.shuffle(examples_idx)
                for (i, example) in enumerate(examples_idx):
                    #yield example
                    #print (example)
                    yield [example['utterances'], example['domain'], example['gate'], example['slot'], example['generate']]

        def batch_reader():
            batch = []
            for instance in instance_reader():
                if len(batch) < self.batch_size:
                    batch.append(instance)
                else:
                    batch = self._fill_batch_data(batch)
                    #print (batch)
                    yield batch
                    batch = []

            if len(batch) > 0:
                #print (batch)
                batch = self._fill_batch_data(batch)
                yield batch

        return batch_reader

if __name__ == '__main__':
    pass
