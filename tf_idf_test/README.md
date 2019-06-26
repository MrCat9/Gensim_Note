## 说明
基于TF-IDF的文本相似度




## 思路
```
# 读出文章集  # 读 csv

# 文本预处理（如：去除空值）

# 读出停用词  # 考虑动态添加停用词？
f = open('stopwords.txt', encoding='gbk')
lines = f.readline()
stoplst = list(map(lambda x:x.strip('\n'), lines))

# 分词，得到所有文章的分词结果texts（形如：[['文章1的分词结果'], ['文章2的分词结果'], ['文章3的分词结果'], ]）
texts = [[word for word in jieba.lcut(document) if word not in stoplst] for document in documents]

# （统计词频，过滤低频词（过滤整个文章集/每篇文章的低频词）（用TF-IDF的话，去除低频词会影响IDF值））

# 建词袋（一个词对应一个id）
dictionary = corpora.Dictionary(texts)
# dictionary.token2id  # 每个词对应的id
# dictionary.dfs  # 词频
# featurenum=len(dictionary.token2id.keys())

# 将所有文章映射到词袋组成的向量空间 -> 组成语料库（形如：[文章1的映射结果, 文章2的映射结果, ]）
# doc2bow -> doc to bag-of-words
corpus = [dictionary.doc2bow(text) for text in texts]

# （TF-IDF）模型 <- 语料库
tfidf = models.TfidfModel(corpus)

# 所有文章的TF-IDF向量 = 模型[语料库]  # 一篇文章的TF-IDF向量 = 模型[一篇文章的向量]
corpus_tfidf = tfidf[corpus]

# （过滤掉一篇文章TF-IDF值较低的那些纬度）（纬度越低计算余弦相似度越快）

# 计算文章相似度
index = similarities.MatrixSimilarity(corpus_tfidf)  # 传入所有文章的TF-IDF向量

# 一篇文章与所有文章的相似结果
sims = index[一篇文章的TF-IDF向量]
```




## 版本一
```
tf_idf_old.py
text_cosine_similarity.py
main.py
```

## 版本二
```
TopicSimilar.py
```

## 版本三
```
tf_idf.py
```

