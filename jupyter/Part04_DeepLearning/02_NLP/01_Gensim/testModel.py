from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('zh_cn.jieba.txt.model')

testwords = ['苹果','数学','学术','白痴','篮球']
for i in xrange(5):
    res = en_wiki_word2vec_model.wv.most_similar(testwords[i])
    print (testwords[i])
    print (res)
