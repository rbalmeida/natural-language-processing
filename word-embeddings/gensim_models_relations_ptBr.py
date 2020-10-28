from gensim.models import Word2Vec, KeyedVectors

pretrained_path = "~/resources/cbow_s50.txt" # download available at http://www.nilc.icmc.usp.br/embeddings
w2v_model = KeyedVectors.load_word2vec_format(pretrained_path)
print('Word2Vec carregado')
print(len(w2v_model.vocab))

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
    print("{} -> {}: {}".format(pair[0], pair[1], w2v_model.wv.distance(pair[0], pair[1])))



# Ref Practical Natural Language Processing.