#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from config import *
import argparse
import paddle
import paddle.fluid as fluid
import reader
import model
import optimization as opt
import time
import numpy as np


if not args['is_train']:
    print ('No train, exit')
    exit -1

#set env
if args['use_cuda']:
    #place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    place = fluid.CUDAPlace(0)
    dev_count = 1
    print ('GPU used, dev_count = %d' % dev_count)
else:
    place = fluid.CPUPlace()
    dev_count = 1
    print ('CPU used')

# place = fluid

#load data
train_data_processor = reader.DataProcessor(
                                batch_size=args['batch_size'],
                                epoch=args['train_epoch_num'],
                                mode='train')
train_batch_reader = train_data_processor.data_generator()
print ('load data succ!')

#set executor
startup_prog = fluid.default_startup_program()
# startup_prog = fluid.Program
#if hasattr(self, _random_seed):
#    startup_prog.random_seed = self._random_seed
train_prog = fluid.Program()

with fluid.program_guard(train_prog, startup_prog):
    with fluid.unique_name.guard():
        train_reader, loss, acc, intent_probs, intent_labels, gate_probs, generate_loss, generate_acc = model.create_model(train_data_processor)
        optimizer = fluid.optimizer.Adagrad(learning_rate=args['base_lr'])
        #optimizer = opt.optimization(
        #                loss=loss,
        #                learning_rate=args['base_lr'],
        #                train_program=train_prog,
        #                startup_prog=startup_prog)
        optimizer.minimize(loss + generate_loss)
train_reader.set_batch_generator(train_batch_reader, places=place)
exe = fluid.Executor(place)
exe.run(startup_prog)

#exec_strategy = fluid.ExecutionStrategy()
#exec_strategy.use_experimental_executor = True
#exec_strategy.num_threads = dev_count
#train_exe = fluid.ParallelExecutor(
#    use_cuda=args['use_cuda'],
#    loss_name=loss.name,
#    exec_strategy=exec_strategy,
#    main_program=train_prog)
train_exe = exe
        

#run train
train_reader.start()
step_count = 0
epoch_count = 0
train_loss_list = []
train_acc_list = []
train_generate_loss_list = []
train_generate_acc_list = []
# i = 0
while True:
    try:
        # i += 1
        # if i >= 10:
            # break
        # step_begin_time = time.time()
        #np_loss, np_acc = train_exe.run(program=train_prog, fetch_list=[loss.name, acc.name])
        #np_loss, np_acc, np_intent_probs = train_exe.run(program=train_prog, fetch_list=[loss.name, acc.name, intent_probs.name])
        np_loss, np_acc, np_intent_probs, np_gate_probs, np_generate_loss, np_generate_acc = train_exe.run(program=train_prog, fetch_list=[loss.name, acc.name, intent_probs.name, gate_probs.name, generate_loss.name, generate_acc.name])
        # step_end_time = time.time()

        fout = open('temp.txt', mode='a+', encoding='utf8')
        stra = str(np.argmax(np_gate_probs, axis=-1).tolist()) + '\n'
        fout.write(stra)
        fout.close()

        train_acc_list.append(np_acc)
        train_loss_list.append(np_loss)
        train_generate_loss_list.append(np_generate_loss)
        train_generate_acc_list.append(np_generate_acc)
        step_count += 1
        if step_count % args['show_step_num'] == 0:
            # print ('epoch: %d, step: %d, avg_loss: %f, acc: %f, time: %f' % \
            #       (epoch_count, step_count,
            #        np.array(train_loss_list).mean(),
            #        np.array(train_acc_list).mean(),
            #        step_end_time - step_begin_time))
            print ('epoch: %d, step: %d, slot_loss: %f, slot_acc: %f, gene_loss: %f, gene_acc: %f' % \
                  (epoch_count, step_count,
                   np.array(train_loss_list).mean(),
                   np.array(train_acc_list).mean(),
                   np.array(train_generate_loss_list).mean(),
                   np.array(train_generate_acc_list).mean()))

        #save model and do validation
        if epoch_count < train_data_processor.get_cur_epoch_idx():
            save_path = os.path.join(args['save_model_dir'], 'epoch_' + str(epoch_count))
            fluid.save(train_prog, save_path)
            train_acc_list = []
            train_loss_list = []
            step_count = 0
            print ('############################# save model for epoch [%d] ################################' % epoch_count)

            if args['is_valid']:
                pass
                #pro_p, pro_r, pro_f1 = infer(save_path, args['valid_file'])
                #print ('############################# epoch [%d]: avg_f1 = %f ##############################\n\n' \
                #        % (epoch_count, pro_f1))
        epoch_count = train_data_processor.get_cur_epoch_idx()

    except fluid.core.EOFException as e:
        print ('111' + str(e))
        save_path = os.path.join(args['save_model_dir'],'fail_' + str(step_count))
        fluid.io.save_persistables(exe, save_path, train_prog)
        train_reader.reset()
        break

print ('train finished!')
    
def infer(self, model_path, test_data_dir, test_file):
    print ('Start test, model: %s, test file: %s/%s' % (model_path, test_data_dir, test_file))
    if args['use_cuda']:
        place = fluid.CUDAPlace(1)
        #dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = 1  
    
    test_data_processor = reader.DataProcessor(
                                    data_dir=test_data_dir,
                                    batch_size=args['test_batch_size'],
                                    filename=test_file, epoch=1)
    test_batch_reader = test_data_processor.data_generator()

    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_reader, loss, acc, intent_probs, intent_labels = model.create_model()
            optimizer = fluid.optimizer.SGD(learning_rate=args['base_lr'])
            optimizer.minimize(loss)
    test_reader.set_batch_generator(test_batch_reader, places=place)

    test_prog = test_prog.clone(for_test=True)
    test_exe = fluid.Executor(place)
    test_exe.run(startup_prog)
    fluid.load(test_prog, model_path, test_exe, None)

    tp, recall_num, pos_num = 0, 0, 0
    acc_list = []
    test_reader.start()
    while True:
        try:
            np_acc, np_intent_probs, np_intent_labels = test_exe.run(
                    program=test_prog, fetch_list=[acc.name, intent_probs.name, intent_labels.name])
        except fluid.core.EOFException as e:
            print ('222' + str(e))
            test_reader.reset()
            break
        intent_args = np.argsort(np_intent_probs)
        if len(intent_args) != len(np_intent_probs):
            print ('ERROR: intent_arg length != intent_prob length')
            continue
        for pred, label in zip(intent_args, np_intent_labels):
            if pred[-1] == label == 1:
                tp += 1
            if pred[-1] == 1:
                recall_num += 1
            if label == 1:
                pos_num += 1
        acc_list.append(np_acc)

    precision = tp * 1.0 / recall_num
    recall = tp * 1.0 / pos_num
    f1 = 2.0 * precision * recall / (precision + recall)
    print ('############### true_positive = %d, recall_num = %d, positive_num = %d ###################' % (tp, recall_num, pos_num))
    print ('############### precision = %f, recall = %f, f1 = %f, acc = %f ####################' % \
            (precision, recall, f1, np.array(acc_list).mean()))
    print ('Finish test, model: %s, test file: %s/%s' \
            % (model_path, test_data_dir, test_file))
    return precision, recall, f1


