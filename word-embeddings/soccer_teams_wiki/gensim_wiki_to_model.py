from nltk.tokenize import SpaceTokenizer
from gensim.models import Word2Vec
import nltk


nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')

tokenizer = SpaceTokenizer()
wiki_tokenized = []

wiki_files = ["soccer_teams_wiki/resources/wikipedia_corinthians.txt",
              "soccer_teams_wiki/resources/wikipedia_palmeiras.txt",
              "soccer_teams_wiki/resources/wikipedia_portuguesa.txt",
              "soccer_teams_wiki/resources/wikipedia_santos.txt",
              "soccer_teams_wiki/resources/wikipedia_sao_paulo.txt"]

for file in wiki_files:
    with open(file, "r") as wiki_file:
        wiki_text = wiki_file.readlines()

    # TODO text cleanup. Remove stop words and other text treatment for articles
    for line in wiki_text:
        phrase = [word.lower() for word in tokenizer.tokenize(line) if word not in stop_words]
        wiki_tokenized.append(phrase)

our_model = Word2Vec(wiki_tokenized, size=10, window=15, min_count=1, workers=4)

while True:
    query_word = input('Type Word: ')
    query_word = query_word.strip().lower()
    if our_model.__contains__(query_word):
        print(our_model.most_similar(query_word))
        print(our_model[query_word])
    else:
        print('Word not found.')
