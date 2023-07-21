import pandas as pd
import numpy as np
import json
from bardapi import Bard
import os

from translate import Translator
from tqdm.auto import tqdm
import re
from difflib import SequenceMatcher
import argparse

from processing import translate_dataset, split_data, prepare_prompt, prep_fewshot


def evaluation(response, ele):
    correctElements = 0
    totalElements = 0
    response = response.splitlines()
    response = response[1:-1]
    response = "".join(response)
    response_tokens = response.replace('**',':').replace('\n',':').split(':')
    #print("response_tokens: ", response_tokens)
    #print("ele['entities']: ", ele['entities'])
    for entity in ele['entities']:
        if entity['value'].lower() in response.lower():
            correctElements += 1
        else:
             for token in response_tokens:
                if SequenceMatcher(None, entity['value'].lower(), token.lower()).ratio() > 0.5:
                    correctElements += 1
                    break
        totalElements += 1
    return correctElements, totalElements


def main(filename2, filename3, bardapi):
    with open(filename3) as f:
        data = json.load(f)
    print("Translating data...")
    translated_dataset = translate_dataset(data)
    print("Splitting data...")
    prompt, eval = split_data(translated_dataset)
    with open(filename2) as f:
        WebCategs = json.load(f)
    translatedKeys = ["Category", "subcategory", "Brand", "Price", "volume","shoe size"]

    val = prepare_prompt(prompt, translatedKeys)

    os.environ['_BARD_API_KEY'] = bardapi

    correctElements = 0
    totalElements = 0
    for ele in tqdm(eval):
        #print("ele: ", ele)
        response = Bard().get_answer(prep_fewshot(val, ele))
        correct, total = evaluation(response['content'], ele)
        correctElements += correct
        totalElements += total
    print(f"Accuracy: {correctElements/totalElements}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--filename2', type=str, default='WebsiteCategoriesDataNoLink.json', help='Name of the file to save the data in')
    parser.add_argument('--filename3', type=str, default='TestData.json', help='Name of the file to save the data in')
    parser.add_argument('--bardapi', type=str, help='api key for bard api')
    args = parser.parse_args()

    filename2 = args.filename2
    filename3 = args.filename3
    bardapi = args.bardapi
    main(filename2, filename3, bardapi)