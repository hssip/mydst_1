import numpy as np 
import json, pickle
from collections import OrderedDict 
from config import *

def fix_general_label_error(labels):
    ontology = json.load(open("data/multi-woz/MULTIWOZ2_2/ontology.json", 'r'))
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in domain_list])
    slots = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    label_dict = dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels])
    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
            
            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict

def process_belief_state(belief_state):
    result = {}
    for i in raw_slots:
        arr = i.split('-')
        domain = arr[0]
        slot = arr[1]
        if domain not in result.keys():
            result[domain] = {}
        result[domain][slot] = ''

    label_dict = fix_general_label_error(belief_state)

    if belief_state:
        for belief in belief_state:
            arr = belief['slots'][0][0].lower().split('-')
            if arr[0] not in domain_list:
                # print(belief_state)
                return []
            temp = arr[1]
            if 'book' in temp:
                temp = temp.split(' ')[1]
            # value = belief['slots'][0][1].lower()
            # process value
            value = label_dict[belief['slots'][0][0]]
            result[arr[0]][temp] = value
    
    # result = fix_general_label_error(result)

    return result

def slots2gate(slots):
    gates = []
    if not slots:
        return ['NONE'] * 30
    # print(slots)
            #none
    for i in raw_slots:
        arr = i.split('-')
        domain = arr[0]
        slot = arr[1]
        if slots[domain][slot]  == 'none' or slots[domain][slot] == '':
            gates.append('NONE')
        elif slots[domain][slot]  == 'dontcare':
            gates.append('DONTCARE')
        else:
            gates.append('UPDATE') 
            # #update
            # elif slots[domain][slot] not in special_slot_value['dontcare'] and \
            #     slots[domain][slot] not in special_slot_value['none']:
            #     gates.append('UPDATE')
            # #dontcare
            # elif slots[domain][slot] in special_slot_value['dontcare']:
            #     gates.append('DONTCARE')
            # else:
            #     gates.append('NONE') 
    # tokens = str(gates) + '\n'
    # fout = open('tokens.txt', mode='a+', encoding='utf8')
    # fout.write(tokens)
    # fout.close()

    return gates

def slots2generate(slots):
    result = []
    max_res_length = 10
    for key in raw_slots:
        arr = key.split('-')
        domain = arr[0]
        slot = arr[1]
        value_ = slots[domain][slot].split()
        raw_len = len(value_)
        value_ += ['PAD'] * (max_res_length - raw_len)
        result.append(value_)
    
    result = [x for y in result for x in y]

    # result = ['none' if x == '' else x for x in temp]
    return result

def belief2gate_generate(belief_state):
    slots = process_belief_state(belief_state)
    gate = slots2gate(slots)
    generate = slots2generate(slots)
    return gate, generate