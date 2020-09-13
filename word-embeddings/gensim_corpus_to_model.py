from gensim.models import Word2Vec
from gensim.test.utils import common_texts

our_model = Word2Vec(common_texts, size=10, window=5, min_count=1, workers=4)
print(our_model.wv.most_similar('computer', topn=5))
print(our_model['computer'])
print(common_texts)

