import os
import logging 
import argparse

encoding = 'utf8'

builtin_words = ['PAD', 'EOS', 'START', 'SEP-SYS', 'SEP-USER', 'UNK']
domain_list = ['hotel', 'restaurant', 'taxi', 'attraction', 'train']
slots_list = [['name','type', 'parking', 'pricerange', 'internet', 'day', 'stay', 'people', 'area', 'stars'],
                ['food', 'pricerange', 'area', 'name', 'time', 'day', 'people'],
                ['leaveat', 'destination', 'departure', 'arriveby'], 
                ['name', 'type', 'area'],
                ['people', 'leaveat', 'destination', 'day','arriveby', 'departure']
                ]
raw_slots = ['hotel-name','hotel-type', 'hotel-parking', 'hotel-pricerange', 'hotel-internet', 'hotel-day', 'hotel-stay', 'hotel-people', 'hotel-area', 'hotel-stars',
        'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'restaurant-name', 'restaurant-time', 'restaurant-day', 'restaurant-people',
        'taxi-leaveat', 'taxi-destination', 'taxi-departure', 'taxi-arriveby', 
        'attraction-name', 'attraction-type', 'attraction-area',
        'train-people', 'train-leaveat', 'train-destination', 'train-day','train-arriveby', 'train-departure']
gate2index = {
    'UPDATE':   0,
    'DONTCARE': 1,
    'NONE':     2
}
special_slot_value={
    'dontcare':['any', 'does not care', 'dont care'],
    'none':['not men', 'not mentioned', 'fun', 'art', 'not', 
            'not mendtioned', '']
}
parser = argparse.ArgumentParser(description='dst')

# Training Setting
parser.add_argument('--debug', help='', required=False, type=int, default=0)
parser.add_argument('--use_cuda', help='', required=False, type=int, default=1)
parser.add_argument('--save_model_dir', help='', required=False, type=str, default='./save_model')
parser.add_argument('--is_train', help='', required=False, type=int, default=1)
parser.add_argument('--train_file', help='', required=False, type=str, default='./data/train_dials.json')
parser.add_argument('--is_valid', help='', required=False, type=int, default=1)
parser.add_argument('--valid_file', help='', required=False, type=str, default='./data/dev_dials.json')
parser.add_argument('--is_test', help='', required=False, type=int, default=0)
parser.add_argument('--test_file', help='', required=False, type=str, default='./data/test_dials.json')
parser.add_argument('--test_model', help='', required=False, type=str, default='.')
parser.add_argument('--save_epoch', help='', required=False, type=int, default=1)
parser.add_argument('--show_step_num', help='', required=False, type=int, default=100)
parser.add_argument('--train_epoch_num', help='', required=False, type=int, default=20)
parser.add_argument('--data_path', help='', required=False, type=str, default='./data')

#params
parser.add_argument('--word_emb_dim', help='', required=False, type=int, default=400)
parser.add_argument('--enc_hidden_dim', help='', required=False, type=int, default=400)
parser.add_argument('--base_lr', help='', required=False, type=float, default=0.01)
parser.add_argument('--dropout', help='', required=False, type=float, default=0.15)
parser.add_argument('--batch_size', help='', required=False, type=int, default=32)
parser.add_argument('--test_batch_size', help='', required=False, type=int, default=1)

parser.add_argument('--all_slot_num', help='', required=False, type=int, default=30)
parser.add_argument('--gate_kind', help='', required=False, type=int, default=3)
parser.add_argument('--slot_emb_dim', help='', required=False, type=int, default=400)



args = vars(parser.parse_args())
'''
if args["load_embedding"]:
    args["hidden"] = 400
    print("[Warning] Using hidden size = 400 for pretrained word embedding (300 + 100)...")
if args["fix_embedding"]:
    args["addName"] += "FixEmb"
if args["except_domain"] != "":
    args["addName"] += "Except"+args["except_domain"]
if args["only_domain"] != "":
    args["addName"] += "Only"+args["only_domain"]
'''
print ('config:  %s' % str(args))


