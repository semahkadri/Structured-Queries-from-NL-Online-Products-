# Structured queries from NL (online products)

## Description:

What we want to do in the final output, is with a user query looking to buy a product from a specific website (https://www.lamode.tn in our case), we want to extract the relevant features and use them to query the website and get the most relevant results.

## Project requirements:

```bash
pip install -r requirements.txt
```

## Preprocessing:

We first wanted to check what are the different categories and subcategories of products that are available on the website, so we scraped the website and extracted the categories and subcategories and stored them in a json file.
We also extracted the different features of each product and stored them in a json file.

```bash
python Website_data_scrapper.py
```

### Possible arguments:

- --filename1: name of the file where the categories and subcategories will be stored with the links for each one of them. (default: WebsiteCategoriesData.json)
- --filename2: name of the file where the categories and subcategories will be stored without the links for each one of them. (default: WebsiteCategoriesDataNoLink.json)
- --queryExamples: our evaluation dataset that contains the queries and the expected results. (default: testData.txt)
- --filename3: processed dataset in a json format (default: TestData.json)

## Evaluation:

We first wanted to evaluate the method that we chose to extract the features from the user query, our chosen method is to use [PaLM 2](https://ai.google/discover/palm2) that is used in bard, we used the bard API as described in this repository: https://github.com/dsdanielpark/Bard-API

To be able to use PaLM 2 for inference task. you need to extract the "\_\_Secure-1PSID" like follows:

1. Visit https://bard.google.com/
2. F12 for console
3. Session: Application → Cookies → Copy the value of \_\_Secure-1PSID cookie.

We also tested with different prompts but the one that we found to be the most efficient is few-shot learning.

A problem that we faced is that the model is not able to extract the features of the product if the query is not in english, so we used a translator to translate the query to english and then we used the model to extract the features.

```bash
python Evaluation.py
```

The accuracy that we got at the end is 0.8 most of the time which is pretty decent.

### Possible arguments:

- --filename2: name of the file where the categories and subcategories are stored without the links for each one of them. (default: WebsiteCategoriesDataNoLink.json)

- --filename3: processed dataset in a json format (default: TestData.json)
- --bardapi: api key for bard api (Must be provided like described above)

## Querying:

We extract the features from the user query using the same method as in the evaluation part, then we preprocess and translate those features and use them to query the website and get the most relevant results.

We all of the products that we get from the custom query that we create, and then we extract all of the paragraphs and prices of each product. we use the paragraphs and compare it with the inital user query and get the most relevant product using either ["dangvantuan/sentence-camembert-base"](https://huggingface.co/dangvantuan/sentence-camembert-base) sentence transformer embeddings or spacy embeddings then we compare the cosine similarity of the embeddings and get the most relevant products.

To further narrow the results we use the prices of the products and compare them with the price that the user is willing to pay (if he provided it) and finally we get the most relevant products.

```bash
python getRelevantArticles.py
```

### Possible arguments:

- --filename3: processed dataset in a json format (default: TestData.json)
- --bardapi: api key for bard api (Must be provided like described above)
- --query: query to search for (default: je veux une montre pour femme de la marque Guess avec un prix moins de 700dt)
- --nb_articles: number of articles to print (default: 5)
- --recommender: recommender to use (spacy or sentence_transformer or both) (default: spacy)
