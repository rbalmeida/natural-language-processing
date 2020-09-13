from nltk.tokenize import SpaceTokenizer
from gensim.models import Word2Vec
import nltk


nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('portuguese')

tokenizer = SpaceTokenizer()
wiki_tokenized = []

wiki_files = [["corinthians", "soccer_teams_wiki/resources/wikipedia_corinthians.txt"],
              ["palmeiras", "soccer_teams_wiki/resources/wikipedia_palmeiras.txt"],
              ["portuguesa", "soccer_teams_wiki/resources/wikipedia_portuguesa.txt"],
              ["santos", "soccer_teams_wiki/resources/wikipedia_santos.txt"],
              ["sao_paulo", "soccer_teams_wiki/resources/wikipedia_sao_paulo.txt"]]

for team in wiki_files:
    team_name = team[0]
    team_file = team[1]
    with open(team_file, "r") as wiki_file:
        wiki_text = wiki_file.readlines()

    # TODO text cleanup. Remove stop words and other text treatment for articles
    for line in wiki_text:
        phrase = [word.lower() for word in tokenizer.tokenize(line) if word not in stop_words]
        # phrase.append(team_name) TODO would adding this context help in improving model performance?
        wiki_tokenized.append(phrase)

our_model = Word2Vec(wiki_tokenized, size=10, window=20, min_count=1, workers=4)

word_comparison = [
    ["corinthians", "palmeiras"],
    ["palmeiras", "portuguesa"],
    ["portuguesa", "santos"],
    ["corinthians", "preto"],
    ["corinthians", "verde"],
    ["corinthians", "vermelho"],
    ["palmeiras", "preto"],
    ["palmeiras", "verde"],
    ["palmeiras", "vermelho"],
    ["portuguesa", "preto"],
    ["portuguesa", "verde"],
    ["portuguesa", "vermelho"],
    ["peixe", "corinthians"],
    ["peixe", "palmeiras"],
    ["peixe", "santos"],
    ["porco", "corinthians"],
    ["porco", "palmeiras"],
    ["porco", "santos"],
    ["timão", "corinthians"],
    ["timão", "palmeiras"],
    ["timão", "santos"],


]

for pair in word_comparison:
    print("{} -> {}: {}".format(pair[0], pair[1], our_model.wv.distance(pair[0], pair[1])))

