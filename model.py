#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import paddle
import paddle.fluid as fluid
import numpy as np
from config import *
from utils import *


def create_model(data_processor):
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
        
        return context, scores

    def attend_vocab(seq, cond):
        scores_ = fluid.layers.matmul(cond, fluid.layers.transpose(x=seq, perm=[1,0]))
        scores = fluid.layers.softmax(scores_, axis=1)
        return scores
    
    def while_cond(i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list):
        return i < n

    def while_body(i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list):
        # cell = fluid.layers.GRUCell(hidden_size=args['slot_emb_dim'])
        # dec_outs, hidden = fluid.layers.rnn(cell = cell,
        #                                     inputs = dec_input,
        #                                     initial_states= hidden)
        init_c = fluid.layers.fill_constant(shape=[1, args['batch_size'] * args['all_slot_num'], args['slot_emb_dim']], dtype='float32', value=0.0 )
        dec_outs, hidden, _ = fluid.layers.lstm(input = dec_input,
                                            init_h= hidden,
                                            init_c=init_c,
                                            max_len=400,
                                            hidden_size=args['slot_emb_dim'],
                                            num_layers=1)

        enc_out = fluid.layers.expand(encoder_outs, expand_times=[args['all_slot_num'], 1, 1])
        context, prob = attend(enc_out, hidden, sent_mask)
        gate_probs = fluid.layers.fc(context, size = args['gate_kind'], act='softmax')
        # gate_probs = fluid.layers.softmax(fluid.layers.matmul(context, hidden2gate))
        
        
        p_vocab = attend_vocab(words_emb, hidden)
        p_gen_vec=  fluid.layers.concat([fluid.layers.squeeze(dec_outs, axes=[1]), context, fluid.layers.squeeze(dec_input, axes=[1])], axis=-1)
        vocab_pointer_switches = fluid.layers.fc(p_gen_vec, size=1,act='sigmoid')
        # vocab_pointer_switches = fluid.layers.sigmoid(fluid.layers.matmul(p_gen_vec, hidden2pgen))
        p_context_ptr = fluid.layers.fill_constant(shape=[args['batch_size'] * args['all_slot_num'], data_processor.get_vocab_size('utterances')],dtype='float32',value=0.0)
        
        # print('p_context_ptr size: %s'%(str(p_context_ptr.shape)))
        # print('story size: %s'%(str(story.shape)))
        # print('prob size: %s'%(str(prob.shape)))

        story_index = fluid.layers.unsqueeze(fluid.layers.expand(story, expand_times=[args['all_slot_num'], 1]),axes=[2])
        updates = fluid.layers.expand(fluid.layers.unsqueeze(prob,axes=[2]), expand_times=[1,1,data_processor.get_vocab_size('utterances')])
        p_context_ptr = fluid.layers.scatter_nd_add(ref = p_context_ptr, index = story_index, updates=updates)

        print('vocab_pointer_switches size: %s'%(str(vocab_pointer_switches.shape)))
        print('p_context_ptr size: %s'%(str(p_context_ptr.shape)))
        print('p_vocab size: %s'%(str(p_vocab.shape)))


        final_p_vocab = fluid.layers.expand((1 - vocab_pointer_switches), expand_times=[1, data_processor.get_vocab_size('utterances')]) * p_context_ptr + \
                        fluid.layers.expand(vocab_pointer_switches, expand_times=[1, data_processor.get_vocab_size('utterances')]) * p_vocab

        # p_final_word = fluid.layers.argmax(final_p_vocab, axis=1)

        npfw = fluid.layers.reshape(final_p_vocab,shape=[args['all_slot_num'], args['batch_size'], -1 ,data_processor.get_vocab_size('utterances')])
        # if i == 0:
        #     all_point_outputs_list.append(gate_probs)
        # else:
        #     all_point_outputs_list.append(npfw)
        i = i + 1
        return i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list

    def get_slot_acc(gate_prob, gates_label, generate_prob, generates_label):
        #gate_prob (slot*batch)*gate
        #gate_prob (slot*batch)*1
        #generate_prob slot*batch*max*wocab
        #generate_label (slot*batch)*max
        gate_loss = fluid.layers.mean(fluid.layers.cross_entropy(input= gate_prob, label=gates_label))
        gate_acc = fluid.layers.accuracy(input=gate_prob, label=gates_label)

        # generates_label1 = fluid.layers.reshape(generates_label, shape=[args['batch_size'] * args['all_slot_num'], -1])
        squs_gates_label = fluid.layers.squeeze(gates_label, axes=[1])
        slot_mask = fluid.layers.cast(fluid.layers.equal(fluid.layers.argmax(gate_prob, axis=-1), squs_gates_label),dtype='float32')

        arg_generate = fluid.layers.argmax(generate_prob, axis=-1)
        reshape_generate = fluid.layers.reshape(arg_generate, shape=[args['batch_size'] * args['all_slot_num'], -1])
        ok_generate = fluid.layers.reduce_mean(fluid.layers.cast(fluid.layers.equal(reshape_generate, generates_label), dtype='float32'), dim=-1)
        symbol = fluid.layers.fill_constant(shape=[args['batch_size'] * args['all_slot_num']],dtype='float32', value=1.0)
        ok_generate_num = fluid.layers.cast(fluid.layers.equal(ok_generate, symbol), dtype='float32')
        generate_acc = fluid.layers.reduce_mean(fluid.layers.elementwise_mul(slot_mask, ok_generate_num))

        generate_loss = fluid.layers.mean(fluid.layers.cross_entropy(generate_prob, fluid.layers.unsqueeze(generates_label,axes=[2])))

        return gate_acc, gate_loss, generate_acc, generate_loss
    

    def _slot_gate(encoder_outs, encoder_last_h, slots_embedding, sent_mask, words_emb, story):

        slots_embedding1 = fluid.layers.transpose(x= slots_embedding, perm=[1,0,2])
        slots_embedding1 = fluid.layers.reshape(x=slots_embedding1, shape=[args['all_slot_num'] * args['batch_size'], -1, args['slot_emb_dim']])
        dec_input = fluid.layers.dropout(slots_embedding1, dropout_prob=args['dropout'])
        hidden = fluid.layers.expand(encoder_last_h, expand_times=[args['all_slot_num'], 1])
        

        # hidden2gate = fluid.layers.create_parameter(shape=[args['slot_emb_dim'], args['gate_kind']], dtype='float32')
        # hidden2pgen = fluid.layers.create_parameter(shape=[args['slot_emb_dim'] * 3, 1],dtype='float32')

        # all_point_outputs_list = fluid.Tensor()
        # all_point_outputs_list = LodTensor_to_Tensor(all_point_outputs_list)
        # all_point_outputs_list.set(np.zeros(shape=(args['all_slot_num'], args['batch_size'], 10 ,data_processor.get_vocab_size('utterances')), 
                                    # dtype='float32'),
                                # fluid.CPUPlace())
        
        all_point_outputs_list = fluid.LoDTensorArray()
        
        out_gate_probs = []
        i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)     
        n = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
        cond = fluid.layers.less_than(x=i, y=n)
        while_op = fluid.layers.While(cond = cond)
        with while_op.block():
        # fluid.layers.arr
        # i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list
        # i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list = \
        # fluid.layers.while_loop(cond=while_cond, 
        #                         body=while_body,
        #                         loop_vars=[i, n, hidden, dec_input, encoder_outs, words_emb, story, all_point_outputs_list])
        # for i in range(10):
            # cell = fluid.layers.GRUCell(hidden_size=args['slot_emb_dim'])
            # dec_outs, hidden = fluid.layers.rnn(cell = cell,
            #                                     inputs = dec_input,
            #                                     initial_states= hidden)
            init_c = fluid.layers.fill_constant(shape=[1, args['batch_size'] * args['all_slot_num'], args['slot_emb_dim']], dtype='float32', value=0.0 )
            dec_outs, hidden, _ = fluid.layers.lstm(input = dec_input,
                                                init_h= fluid.layers.unsqueeze(hidden, axes=[0]),
                                                init_c=init_c,
                                                max_len=400,
                                                hidden_size=args['slot_emb_dim'],
                                                num_layers=1)
            hidden = fluid.layers.squeeze(hidden, axes=[0])
            enc_out = fluid.layers.expand(encoder_outs, expand_times=[args['all_slot_num'], 1, 1])
            context, prob = attend(enc_out, hidden, sent_mask)
            gate_probs = fluid.layers.fc(context, size = args['gate_kind'], act='softmax')
            # gate_probs = fluid.layers.softmax(fluid.layers.matmul(context, hidden2gate))

            if i == 0:
                out_gate_probs = gate_probs
            
            
            p_vocab = attend_vocab(words_emb, hidden)
            p_gen_vec=  fluid.layers.concat([fluid.layers.squeeze(dec_outs, axes=[1]), context, fluid.layers.squeeze(dec_input, axes=[1])], axis=-1)
            vocab_pointer_switches = fluid.layers.fc(p_gen_vec, size=1,act='sigmoid')
            # vocab_pointer_switches = fluid.layers.sigmoid(fluid.layers.matmul(p_gen_vec, hidden2pgen))
            p_context_ptr = fluid.layers.fill_constant(shape=[args['batch_size'] * args['all_slot_num'], data_processor.get_vocab_size('utterances')],dtype='float32',value=0.0)
            

            story_index = fluid.layers.unsqueeze(fluid.layers.expand(story, expand_times=[args['all_slot_num'], 1]),axes=[2])
            updates = fluid.layers.expand(fluid.layers.unsqueeze(prob,axes=[2]), expand_times=[1,1,data_processor.get_vocab_size('utterances')])
            p_context_ptr = fluid.layers.scatter_nd_add(ref = p_context_ptr, index = story_index, updates=updates)

            # print('vocab_pointer_switches size: %s'%(str(vocab_pointer_switches.shape)))
            # print('p_context_ptr size: %s'%(str(p_context_ptr.shape)))
            # print('p_vocab size: %s'%(str(p_vocab.shape)))


            final_p_vocab = fluid.layers.expand((1 - vocab_pointer_switches), expand_times=[1, data_processor.get_vocab_size('utterances')]) * p_context_ptr + \
                            fluid.layers.expand(vocab_pointer_switches, expand_times=[1, data_processor.get_vocab_size('utterances')]) * p_vocab

            # p_final_word = fluid.layers.argmax(final_p_vocab, axis=1)

            npfw = fluid.layers.reshape(final_p_vocab,shape=[args['all_slot_num'], args['batch_size'], -1 ,data_processor.get_vocab_size('utterances')])
            # if i == 0:
            #     c = npfw
            # else:
            #     c = fluid.layers.concat([c, npfw], axis=2)
            all_point_outputs_list.append(npfw)
            i=fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=n, cond=cond)
            # all_point_outputs_list.append(npfw)
            # all_point_outputs_list[:, :, i, :] = npfw
            # dec_input =
        # gate_probs = all_point_outputs_list[0] 
        # all_point_outputs = fluid.layers.concat(all_point_outputs_list, axis=2)
        c = fluid.layers.reshape(x=c, shape=[args['all_slot_num'], args['batch_size'], -1 ,data_processor.get_vocab_size('utterances')])
        # all_point_outputs_list
        return out_gate_probs, c

    def _net_conf_at(all_word, word, sent_mask, intent_label, gates_label, slots, sent_mask1, generates_label):
        """
        Configure the network
        """
        # tensor_all_word = fluid.Tensor()
        # tensor_all_word.set(np.array([i for i in range(data_processor.get_vocab_size('utterances'))]), fluid.CPUPlace())
        
        # all_word = fluid.layers.fill_constant(shape=[data_processor.get_vocab_size('utterances')], dtype='float32', value=tensor_all_word)

        all_word_emb = fluid.embedding(
            input=all_word,
            size=[data_processor.get_vocab_size('utterances'), args['word_emb_dim']],
            param_attr=fluid.ParamAttr(
                name='all_word_emb',
                initializer=fluid.initializer.Normal(0., args['word_emb_dim']**-0.5)))
        cat_list = []
        for i in range(args['batch_size']):
            cat_list.append(fluid.layers.gather(input=all_word_emb, index=word[i]))
        word_emb = fluid.layers.reshape(fluid.layers.concat(cat_list, axis=0),shape=[args['batch_size'], -1, args['slot_emb_dim']])
        print('word_emb size: %s'%(str(word_emb.shape)))
        word_emb = fluid.layers.scale(x=word_emb, scale=args['word_emb_dim']**0.5)
        if args['dropout'] > 0.00001:
            word_emb = fluid.layers.dropout(word_emb, dropout_prob=args['dropout'], seed=None, is_test=False)

        input_feature = word_emb 
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
            size=data_processor.get_vocab_size('domain'),
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
            size=[data_processor.get_vocab_size('slot'), args['slot_emb_dim']],
            param_attr=fluid.ParamAttr(
                name='slot_emb',
                initializer=fluid.initializer.Normal(0., args['slot_emb_dim']**-0.5)))
        slot_emb = fluid.layers.scale(x=slot_emb, scale=args['slot_emb_dim']**0.5)
        # words = [i for i in range(data_processor.get_vocab_size('utterance'))]

        gate_prob, generate_prob = _slot_gate(encoder_outs=bigru_output,
                                encoder_last_h=bigru_last_h,
                                slots_embedding=slot_emb,
                                sent_mask=sent_mask1,
                                words_emb=all_word_emb,
                                story=word)
        gates_label1 = fluid.layers.transpose(gates_label, perm=[1, 0])     
        gates_label1 = fluid.layers.reshape(gates_label1, shape=[args['batch_size'] * args['all_slot_num'], -1])
        
        
        generates_label_ = fluid.layers.reshape(generates_label, shape=[args['batch_size'],  args['all_slot_num'], -1])
        generates_label_ = fluid.layers.transpose(generates_label_, perm=[1, 0, 2])
        generates_label_ = fluid.layers.reshape(generates_label_, shape=[args['batch_size'] * args['all_slot_num'], -1])
        # generate_prob = fluid.layers.transpose(generate_prob, perm=[1, 0, 2])
        ############## slot end #########################
        
        # loss = fluid.layers.mean(x=ce_loss)
        # accuracy = fluid.layers.accuracy(input=intent_probs, label=intent_label)
        # if args['debug']:
        #     print ('loss: %s,  intent_probs: %s' % (str(loss.shape), str(intent_probs.shape)))
        #     intent_probs = fluid.layers.Print(intent_probs, message='intent_probs: ', summarize=-1)
        
        gate_acc, gate_loss, generate_acc, generate_loss = get_slot_acc(gate_prob=gate_prob,
                                                                        gates_label=gates_label1,
                                                                        generate_prob=generate_prob,
                                                                        generates_label=generates_label_)

        ########## chose loss and acc###############
        # loss = gate_loss
        # accuracy = gate_acc
        ########## chose loss and acc end ##########

        return gate_loss, gate_acc, intent_probs, gate_prob, generate_loss, generate_acc

    word = fluid.data(name='word', shape=[args['batch_size'], None], dtype='int64', lod_level=0)
    sent_mask = fluid.data(name='sent_mask', shape=[None, None], dtype='int64', lod_level=0)
    intent_label = fluid.data(name="intent_label", shape=[None, 1], dtype='int64', lod_level=0)
    gates_label = fluid.data(name="gates_label", shape=[args['batch_size'], args['all_slot_num']], dtype='int64', lod_level=0)
    slots = fluid.data(name="slots", shape=[args['batch_size'], args['all_slot_num']], dtype='int64', lod_level=0)
    # context_len = fluid.layers.data(name='context_len', shape=[None, 1], dtype='int64', lod_level=0)
    sent_mask1 = fluid.data(name='sent_mask1', shape=[None, None], dtype='float32', lod_level=0)
    generates_label = fluid.data(name = 'generates_label', shape=[args['batch_size'], None], dtype='int64', lod_level=0)
    
    all_word = fluid.data(name='all_word', shape=[None], dtype='int64', lod_level=0)

    loader = fluid.io.DataLoader.from_generator(feed_list=[all_word, word, sent_mask, intent_label, gates_label, slots, sent_mask1, generates_label], capacity=16, iterable=False)
    gate_loss, gate_acc, intent_probs, gate_probs, generate_loss, generate_acc = _net_conf_at(all_word, word, sent_mask, intent_label, gates_label, slots, sent_mask1, generates_label)
    return loader, gate_loss, gate_acc, intent_probs, intent_label, gate_probs, generate_loss, generate_acc


