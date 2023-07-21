import pandas as pd
import numpy as np
import json
from bardapi import Bard
import os

from translate import Translator
from tqdm.auto import tqdm
import re
from difflib import SequenceMatcher
import spacy

from bs4 import BeautifulSoup
import requests
import urllib.request
import time

from tqdm.auto import tqdm

import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

import argparse

from processing import translate_dataset, split_data, prepare_prompt, prep_fewshot
from deep_translator import GoogleTranslator


def extract_entities(response):
    entities = []
    responseSplited = response['content'].split("\n")
    responseFiltered = []
    for index,ele in enumerate(responseSplited):
        if "**" in ele:
            responseFiltered.append(ele)
    responseFiltered = list(filter(None, responseFiltered))
    for index,ele in enumerate(responseFiltered):
        elements = ele.split("**")
        entities.append({
            'entity': elements[1].strip(),
            'value': elements[-1].strip()
        })
    return entities
def extract_value_relation(value_string):
    pattern = r"\d+\.?\d*"  # Regular expression pattern to match numbers
    numbers = re.findall(pattern, value_string)
    numbers = [int(num) for num in numbers]  
    if len(numbers) == 2:
        return "between", numbers
    pattern = r"\b(greater|more|less|below|bellow|between)\b"
    matches = re.search(pattern, value_string, re.IGNORECASE)

    if matches:
        relation = matches.group(1)
        if relation.strip().lower() in ["greater", "more"]:
            # Logic for greater or more
            value1 = re.findall(r"[\d.,]+", value_string)
            return f"greater", value1[0] if value1 else None
        elif relation.strip().lower() in ["less", "below", "bellow"]:
            # Logic for less, below, or bellow
            value1 = re.findall(r"[\d.,]+", value_string)
            return f"less", value1[0] if value1 else None
        elif relation.strip().lower() == "between":
            # Logic for between
            values = re.findall(r"[\d.,]+", value_string)
            return f"between", values[:2] if len(values) >= 2 else None
        else:
            return "Invalid value relation.", 0
        
    else:
        return "No value relation found.", 0
    
def extract_price_max_min(entities):
    for i in entities:
        if "price" in i['entity'].lower():
            return extract_value_relation(i['value'])
    return "No price found", 0
        
def prepare_web_search_query(entities):
    query = ""
    translator = Translator(from_lang="en", to_lang="fr")
    translated = GoogleTranslator(source='en', target='fr') 
    for i in entities:
        if "price" not in i['entity'].lower():
            valeur = translator.translate(i['value'])
            if "MYMEMORY WARNING:".lower() in valeur.lower():
                valeur = translated.translate(i['value'])
            query += f"{valeur} "
    return query

def get_recommended_articles(entities):
    val = len(entities)
    while val>0:
        query = prepare_web_search_query(entities[:val])
        query = query.replace(" ", "+")
        url = f"https://www.lamode.tn/recherche?controller=search&s={query}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        try:
            articles = soup.find("div", class_="products").find_all("article")
            links = []
            for article in articles:
                links.append(article.find("a")['href'])
            return links
        except:
            val = val-1
            continue
    return []

def get_articles_descriptions(links):
    descriptions = []
    for link in tqdm(links):
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        try:
            descriptions.append((link,soup.find("div", class_="product-description").text))
        except:
            try:
                descriptions.append((link,soup.find("div", class_="tvproduct-page-decs").text))
            except:
                descriptions.append((link,"None"))
    return descriptions

def filter_based_on_price(links, symb, vall):
    filtered = []    
    for link in links:
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        prr = soup.find("span", class_="price")['content']
        if symb.strip().lower() == "between":
            if float(vall[0]) <= float(prr) <= float(vall[1]):
                filtered.append(link)
        elif symb.strip().lower() == "greater":
            if float(prr) >= float(vall):
                filtered.append(link)
        elif symb.strip().lower() == "less":
            if float(prr) <= float(vall):
                filtered.append(link)
    return filtered


def compare_article_query_SentenceTransformer(query, article):
    model = SentenceTransformer("dangvantuan/sentence-camembert-base")
    query_embedding = model.encode(query)
    article_embeddings = model.encode(article)
    cos_scores = util.pytorch_cos_sim(query_embedding, article_embeddings)[0]
    cos_scores = cos_scores.numpy()[0]
    return cos_scores


def get_recommended_articles_sentence_transformer(query, descs):
    recommended_articles = []
    for link, desc in tqdm(descs):
        score = compare_article_query_SentenceTransformer(query, desc)
        recommended_articles.append((link, score))
    recommended_articles.sort(key=lambda x: x[1], reverse=True)
    return recommended_articles

def compare_article_query_Spacy(query, article, pipeline="fr_core_news_sm"):
    nlp = spacy.load(pipeline)
    query_doc = nlp(query)
    article_doc = nlp(article)
    return query_doc.similarity(article_doc)


def get_recommended_articles_spacy(query, descs, pipeline="fr_core_news_sm"):
    recommended_articles = []
    for link, desc in tqdm(descs):
        score = compare_article_query_Spacy(query, desc, pipeline)
        recommended_articles.append((link, score))
    recommended_articles.sort(key=lambda x: x[1], reverse=True)
    return recommended_articles

def print_recommended_articles(recommended_articles, nb_articles=5):
    for i, (link, score) in enumerate(recommended_articles[:nb_articles]):
        print(f"{i+1}. {link} - {score}")



def main(filename3, bardapi, query, nb_articles, recommender):
    os.environ['_BARD_API_KEY'] = bardapi
    with open(filename3) as f:
        data = json.load(f)
    print("Translating data...")
    translated_data = translate_dataset(data)
    translatedKeys = ["Category", "subcategory", "Brand", "Price", "volume","shoe size"]
    prompt, eval = split_data(translated_data)
    prompt = prepare_prompt(prompt, translatedKeys)
    exampQuery = query
    translator = Translator(from_lang="fr", to_lang="en")
    translated = GoogleTranslator(source='en', target='fr')
    translatedQuery = translator.translate(exampQuery)
    if "MYMEMORY WARNING:".lower() in translatedQuery.lower():
        translatedQuery = translated.translate(exampQuery)

    translatedQueryy = {}
    translatedQueryy['text'] = translatedQuery
    prep = prep_fewshot(prompt, translatedQueryy)
    response = Bard().get_answer(prep)
    print("extracting entities...")
    extracted_entities = extract_entities(response)
    print("extracting price preference...")
    symb, vall = extract_price_max_min(extracted_entities)
    print("getting recommended articles links...")
    articlesRecommended = get_recommended_articles(extracted_entities)
    
    if symb not in ["between", "greater", "less"]:
        print("price preference not found")
        print("getting all articles descriptions...")
        descs = get_articles_descriptions(articlesRecommended)
    else:
        print("filtering articles based on price preference...")
        filtered_descs = filter_based_on_price(articlesRecommended, symb, vall)
        print("getting all articles descriptions...")
        descs = get_articles_descriptions(filtered_descs)

    print("number of relevant articles: ", len(rec))
    if recommender == "spacy":
        rec = get_recommended_articles_spacy(exampQuery, descs)
        print("Top 5 articles based on spacy: ")
        print_recommended_articles(rec, min(nb_articles, len(rec)))
    elif recommender == "sentence_transformer":
        rec1 = get_recommended_articles_sentence_transformer(exampQuery, descs)
        print("Top 5 articles based on sentence transformer: ")
        print_recommended_articles(rec1, min(nb_articles, len(rec)))
    else:
        rec = get_recommended_articles_spacy(exampQuery, descs)
        rec1 = get_recommended_articles_sentence_transformer(exampQuery, descs)
        print("Top 5 articles based on spacy: ")
        print_recommended_articles(rec, min(nb_articles, len(rec)))
        print("Top 5 articles based on sentence transformer: ")
        print_recommended_articles(rec1, min(nb_articles, len(rec)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the relevant articles')
    parser.add_argument('--filename3', type=str, default='TestData.json', help='Name of the file to save the data in')
    parser.add_argument('--bardapi', type=str, default='WggIsf09_Prld2KezCtJ9y3xf5KfBcZkg9Kw2zc1v-t8nL2C-kQxK51os9kBFSnRt2YItQ.', help='api key for bard api')
    parser.add_argument('--query', type=str, default='je veux une montre pour femme de la marque Guess avec un prix moins de 700dt', help='query to search for')
    parser.add_argument('--nb_articles', type=int, default=5, help='number of articles to print')
    parser.add_argument('--recommender', type=str, default='spacy', help='recommender to use (spacy or sentence_transformer or both)')
    args = parser.parse_args()

    filename3 = args.filename3
    bardapi = args.bardapi
    query = args.query
    nb_articles = args.nb_articles
    recommender = args.recommender

    main(filename3, bardapi, query, nb_articles, recommender)



