import pandas as pd
import numpy as np
import json
from bardapi import Bard
import os

from translate import Translator
from tqdm.auto import tqdm
import re
from difflib import SequenceMatcher


from deep_translator import GoogleTranslator
#translated = GoogleTranslator(source='auto', target='de').translate("keep it up, you are awesome")
#print(translated)


def translate_dataset(dataset):
    translator = Translator(from_lang="fr", to_lang="en")
    translated = GoogleTranslator(source='fr', target='en') #.translate("keep it up, you are awesome")
    translated_dataset = []
    
    for item in tqdm(dataset):
        translated_item = {}        
        translated_item['text'] = translator.translate(item['text'])
        if "MYMEMORY WARNING:".lower() in translated_item['text'].lower():
            translated_item['text'] = translated.translate(item['text'])        
            translated_item['entities'] = []
            for entity in item['entities']:
                try:
                    entityy = translated.translate(entity['entity'])
                except:
                    entityy = entity['entity']
                try:
                    valuee = translated.translate(entity['value'])
                except:
                    valuee = entity['value']
                translated_entity = {
                    'entity': entityy,
                    'value': valuee
                }
                translated_item['entities'].append(translated_entity)

        else:
            translated_item['entities'] = []
            for entity in item['entities']:
                translated_entity = {
                    'entity': translator.translate(entity['entity']),
                    'value': translator.translate(entity['value'])
                }
                translated_item['entities'].append(translated_entity)


        translated_dataset.append(translated_item)
    return translated_dataset


def split_data(data):
    np.random.shuffle(data)
    spl = int(len(data) * 0.2)
    print(spl)
    prompt = data[:spl]
    eval = data[spl:]
    return prompt, eval


def prepare_prompt(prompt, list_of_keys):
    #prompt = prompt.copy()
    prp = f""""Extract the relevant information from the following user query, these are the possible categories: \n {list_of_keys}\n please just give me the answer for the last query only. give me the answer directly by following the pattern bellow, and make sure to follow the same writing pattern \nhere are some examples of the query and the relevant information to extract: \n"""
    for i in prompt:
        prp += f"""\nQuery: {i['text'].strip()}\n"""
        for key in i['entities']:
            prp += f"""**{key['entity'].strip()}** {key['value'].strip()}\n"""
    return prp
def prep_fewshot(prompt, ele):
    prp = prompt
    prp += f"""\nQuery: {ele['text'].strip()}\n **Category** """
    return prp