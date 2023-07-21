import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import os
import json
import datetime
import argparse

def get_subcategories(link):
    subPage = requests.get(link)
    subSoup = BeautifulSoup(subPage.content, 'html.parser')
    filterSection = subSoup.find('div', id='search_filters').find('div', class_='tvserach-filter-wrapper').find('div', class_='tvsearch-filter-content-wrapper').find_all('section')
    subCateg = {}
    subCategNoLinks = {}
    for section in filterSection:
        name = section.find('p').text
        subCateg[name] = []
        subCategNoLinks[name] = []
        subcategories = section.find('ul', class_='tvfilter-search-types-dropdown').find_all('li')
        if  (section.find('ul', class_='tvfilter-search-types-dropdown').has_attr('data-slider-min')) == False:
            for subcategory in subcategories:
                subCateg[name].append([subcategory.find('a').text.split('\n')[1].strip(), subcategory.find('a')['href']])
                subCategNoLinks[name].append(subcategory.find('a').text.split('\n')[1].strip())
        else:
            subPriceMax = section.find('ul', class_='tvfilter-search-types-dropdown')['data-slider-max']
            subPriceMin = section.find('ul', class_='tvfilter-search-types-dropdown')['data-slider-min']
            subCateg[name].append(subPriceMin)
            subCateg[name].append(subPriceMax)
            subCategNoLinks[name].append(subPriceMin)
            subCategNoLinks[name].append(subPriceMax)
    return subCateg, subCategNoLinks


def get_categories(website, data, dataNoLink):
    page = requests.get(website)
    soup = BeautifulSoup(page.content, 'html.parser')
    categories = soup.find('div',class_='container-fluid main-categories-home').find('div',class_='row').find_all('div',class_='row')
    elementsLis = []
    for a in categories:
        elems = a.findChildren(recursive=False)
        for elem in elems:
            elementsLis.append(elem)
    for category in elementsLis:
        link = category.find('a')['href']
        name = category.find('div', class_="tvcategory-slider-info-box").text
        subcategories, subcategoriesNoLink = get_subcategories(link)
        data['categories'].append({
            'name': name,
            'link': link,
            'subcategories': subcategories
        })
        dataNoLink['categories'].append({
            'name': name,
            'subcategories': subcategoriesNoLink
        })
    return data, dataNoLink
def transform_to_ner_dataset(query):
    ner_dataset = []
    
    # Split the query into different parts
    parts = query.split("\n\n")
    
    for part in parts:
        lines = part.split("\n")
        entity_dict = {}
        
        for line in lines:
            if line.startswith("Query:"):
                text = line.split("Query:")[1].strip()
            else:
                key, value = line.split(":")
                entity_type = key.strip()
                entity_value = value.strip()
                entity_dict[entity_type] = {
                    "entity": entity_type,
                    "value": entity_value,
                }
        entities = list(entity_dict.values())
        ner_data = {
            "text": text,
            "entities": entities
        }
        ner_dataset.append(ner_data)
    return ner_dataset

def save_jsons(data, dataNoLink, NerdataSet, filename1= 'WebsiteCategoriesData.json', filename2= 'WebsiteCategoriesDataNoLink.json', filename3= 'TestData.json'):
    with open(filename1, 'w') as outfile:
        json.dump(data, outfile)
    with open(filename2, 'w') as outfile:
        json.dump(dataNoLink, outfile)
    with open(filename3, 'w') as outfile:
        json.dump(NerdataSet, outfile)

def main(filename1, filename2, filename3, queryExamples):
    website = 'https://www.lamode.tn/'
    data = {}
    data['categories'] = []
    dataNoLink = {}
    dataNoLink['categories'] = []
    print("Getting data from website...")
    data, dataNoLink = get_categories(website, data, dataNoLink)
    with open(queryExamples, 'r') as file:
        TxtData = file.read()
    NerdataSet = transform_to_ner_dataset(TxtData)
    save_jsons(data, dataNoLink, NerdataSet, filename1, filename2, filename3)
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data from website')
    parser.add_argument('--filename1', type=str, default='WebsiteCategoriesData.json', help='Name of the file to save the data in')
    parser.add_argument('--filename2', type=str, default='WebsiteCategoriesDataNoLink.json', help='Name of the file to save the data in')
    parser.add_argument('--filename3', type=str, default='TestData.json', help='Name of the file to save the data in')
    parser.add_argument('--queryExamples', type=str, default='testData.txt', help='Name of the file to save the data in')
    args = parser.parse_args()
    filename1 = args.filename1
    filename2 = args.filename2
    filename3 = args.filename3
    queryExamples = args.queryExamples
    main(filename1, filename2, filename3, queryExamples)