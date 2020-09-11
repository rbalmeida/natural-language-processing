from gensim.models import Word2Vec, KeyedVectors

pretrained_path = "~/resources/GoogleNews-vectors-negative300.bin"
w2v_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
print('done loading Word2Vec')
print(len(w2v_model.vocab))

while True:
    query_word = input('Type Word: ')
    query_word = query_word.strip().lower()
    print(w2v_model.most_similar(query_word))
    print(w2v_model[query_word])


# Ref Practical Natural Language Processing.