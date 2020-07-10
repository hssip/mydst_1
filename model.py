#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import paddle
import paddle.fluid as fluid
import numpy as np
from config import *


def create_model(data_generator):
    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        cell = fluid.layers.GRUCell(hidden_size=args['enc_hidden_dim'])
        gru_outs, gru_final = fluid.layers.rnn(cell=cell, inputs=input_feature)

        cell_r = fluid.layers.GRUCell(hidden_size=args['enc_hidden_dim'])
        gru_outs_r, gru_final_r = fluid.layers.rnn(cell=cell, inputs=input_feature, is_reverse=True)
        #gru_r_outs, gru_r_final = fluid.layers.rnn(cell=cell, inputs=pre_gru, is_reverse=True)
        if args['debug']:
            gru_outs = fluid.layers.Print(gru_outs, message='gru out: ', summarize=-1)
            #gru_r_outs_print = fluid.layers.Print(gru_r_outs, message='gru_r out: ', summarize=-1)

        # bi_merge = fluid.layers.concat(input=[gru_outs, gru_outs_r], axis=-1)
        # bi_last_h = fluid.layers.concat(input=[gru_final, gru_final_r], axis=-1)
        gru_outs_r = fluid.layers.reverse(x = gru_outs_r, axis=1)
        bi_merge = gru_outs + gru_outs_r 
        bi_last_h = gru_final + gru_final_r 
        #bi_merge = gru_outs
        if args['debug']:
            bi_merge = fluid.layers.Print(bi_merge, message='bi_merge: ', summarize=-1)
        return bi_merge, bi_last_h

    def attend(seq, cond, sent_mask):
        a = fluid.layers.expand(x = fluid.layers.unsqueeze(cond, axes=[1]), 
                                expand_times=[1, fluid.layers.shape(seq)[1], 1])

        scores_ = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(a, seq),2)
        sent_mask = fluid.layers.cast(fluid.layers.expand(sent_mask, expand_times=[args['all_slot_num'], 1]),dtype='float32')
        scores = fluid.layers.elementwise_add(scores_, sent_mask)
        scores = fluid.layers.softmax(scores_, axis=1)
        
        # scores = fluid.layers.elementwise_add(sent_mask, scores)

        a = fluid.layers.expand(x = fluid.layers.unsqueeze(scores, axes=[2]),
                                expand_times=[1,1,args['slot_emb_dim']])
        a = fluid.layers.elementwise_mul(a, seq)
        context = fluid.layers.reduce_sum(a,dim=1)
        print('context size: %s'%(str(context.shape)))
        return context

    def vocab_atten():
        return

    def _slot_gate(encoder_outs, encoder_last_h, slots_embedding, sent_mask):

        slots_embedding1 = fluid.layers.transpose(x= slots_embedding, perm=[1,0,2])
        slots_embedding1 = fluid.layers.reshape(x=slots_embedding1, shape=[args['all_slot_num'] * args['batch_size'], -1, args['slot_emb_dim']])
        dec_input = fluid.layers.dropout(slots_embedding1, dropout_prob=args['dropout'])
        hidden = fluid.layers.expand(encoder_last_h, expand_times=[args['all_slot_num'], 1])
        
        cell = fluid.layers.GRUCell(hidden_size=args['slot_emb_dim'])
        dec_outs, hidden = fluid.layers.rnn(cell = cell,
                                            inputs = dec_input,
                                            initial_states= hidden)

        enc_out = fluid.layers.expand(encoder_outs, expand_times=[args['all_slot_num'], 1, 1])
        context = attend(enc_out, hidden, sent_mask)
        gate_probs = fluid.layers.fc(context, size = args['gate_kind'], act='softmax')

        return gate_probs 

    def _net_conf_at(word, sent_mask, intent_label, gates_label, slots, sent_mask1):
        """
        Configure the network
        """
        word_emb = fluid.embedding(
            input=word,
            size=[data_generator.get_vocab_size('utterances'), args['word_emb_dim']],
            param_attr=fluid.ParamAttr(
                name='word_emb',
                initializer=fluid.initializer.Normal(0., args['word_emb_dim']**-0.5)))
        word_emb = fluid.layers.scale(x=word_emb, scale=args['word_emb_dim']**0.5)

        if args['dropout'] > 0.00001:
            word_emb = fluid.layers.dropout(word_emb, dropout_prob=args['dropout'], seed=None, is_test=False)

        input_feature = word_emb 
        #input_feature = fluid.layers.concat(input=[word_emb, istag_emb], axis=1, name="emb_concat")
        bigru_output, bigru_last_h = _bigru_layer(input_feature)
        if args['debug']:
            bigru_out = fluid.layers.Print(input=bigru_output, message='bigru_output: ')

        #mask padding tokens

        sent_mask_r = fluid.layers.reverse(sent_mask, -1)
        sent_mask_cat = fluid.layers.concat(sent_mask, sent_mask_r)
        sent_mask_cat = fluid.layers.cast(sent_mask_cat, 'float32')
        #bigru_output = fluid.layers.elementwise_mul(bigru_output, sent_mask, axis=0)
        bigru_output = fluid.layers.elementwise_mul(bigru_output, sent_mask_cat, axis=0)
        sent_rep = fluid.layers.reduce_max(input=bigru_output, dim=-2, keep_dim=False)
        #sent_rep = fluid.layers.reduce_mean(input=bigru_output, dim=-2, keep_dim=False)
        if args['debug']:
            sent_rep = fluid.layers.Print(input=sent_rep, message='sent_rep: ')

        sent_fc = fluid.layers.fc(
            input=sent_rep,
            size=data_generator.get_vocab_size('domain'),
            param_attr=fluid.ParamAttr(
                learning_rate=1.0,
                trainable=True,
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr( name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
        if args['debug']:
            sent_fc = fluid.layers.Print(input=sent_fc, message='sent_fc: ')

        ce_loss, intent_probs = fluid.layers.softmax_with_cross_entropy(
            logits=sent_fc, label=intent_label, return_softmax=True)

        ################ slot #########################
        slot_emb = fluid.embedding(
            input=slots,
            size=[data_generator.get_vocab_size('slot'), args['slot_emb_dim']],
            param_attr=fluid.ParamAttr(
                name='slot_emb',
                initializer=fluid.initializer.Normal(0., args['slot_emb_dim']**-0.5)))
        slot_emb = fluid.layers.scale(x=slot_emb, scale=args['slot_emb_dim']**0.5)
        gate_prob = _slot_gate(encoder_outs=bigru_output,
                                encoder_last_h=bigru_last_h,
                                slots_embedding=slot_emb,
                                sent_mask=sent_mask1)
        gates_label1 = fluid.layers.transpose(gates_label, perm=[1, 0])     
        gates_label1 = fluid.layers.reshape(gates_label1, shape=[args['batch_size'] * args['all_slot_num'], -1])
        # gates_label1 = fluid.layers.unsqueeze(gates_label, axes=[1])
        gate_loss = fluid.layers.mean(fluid.layers.cross_entropy(input= gate_prob, label=gates_label1))
        # gate_acc = fluid.layers.mean(fluid.layers.cast(fluid.layers.equal(fluid.layers.argmax(gate_prob,axis=-1), 1), dtype='float32'))
        gate_acc = fluid.layers.accuracy(input=gate_prob, label=gates_label1)
        
        ############## slot end #########################
        
        loss = fluid.layers.mean(x=ce_loss)
        accuracy = fluid.layers.accuracy(input=intent_probs, label=intent_label)
        if args['debug']:
            print ('loss: %s,  intent_probs: %s' % (str(loss.shape), str(intent_probs.shape)))
            intent_probs = fluid.layers.Print(intent_probs, message='intent_probs: ', summarize=-1)
        
        ########## chose loss and acc###############
        loss = gate_loss
        accuracy = gate_acc
        ########## chose loss and acc end ##########

        return loss, accuracy, intent_probs, gate_prob

    word = fluid.data(name='word', shape=[None, None], dtype='int64', lod_level=0)
    sent_mask = fluid.data(name='sent_mask', shape=[None, None], dtype='int64', lod_level=0)
    intent_label = fluid.data(name="intent_label", shape=[None, 1], dtype='int64', lod_level=0)
    gates_label = fluid.data(name="gates_label", shape=[args['batch_size'], args['all_slot_num']], dtype='int64', lod_level=0)
    slots = fluid.data(name="slots", shape=[args['batch_size'], args['all_slot_num']], dtype='int64', lod_level=0)
    # context_len = fluid.layers.data(name='context_len', shape=[None, 1], dtype='int64', lod_level=0)
    sent_mask1 = fluid.data(name='sent_mask1', shape=[None, None], dtype='float32', lod_level=0)
    loader = fluid.io.DataLoader.from_generator(feed_list=[word, sent_mask, intent_label, gates_label, slots, sent_mask1], capacity=16, iterable=False)
    avg_cost, accuracy, intent_probs, gate_probs = _net_conf_at(word, sent_mask, intent_label, gates_label, slots, sent_mask1)
    return loader, avg_cost, accuracy, intent_probs, intent_label, gate_probs


