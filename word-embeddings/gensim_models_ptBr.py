from gensim.models import Word2Vec, KeyedVectors

pretrained_path = "~/resources/cbow_s50.txt" # download available at http://www.nilc.icmc.usp.br/embeddings
w2v_model = KeyedVectors.load_word2vec_format(pretrained_path)
print('Word2Vec carregado')
print(len(w2v_model.vocab))

while True:
    query_word = input('Digite a palavra: ')
    query_word = query_word.strip().lower()
    if w2v_model.__contains__(query_word):
        print(w2v_model.most_similar(query_word))
        print(w2v_model[query_word])
    else:
        print('palavra não encontrada')


# Ref Practical Natural Language Processing.