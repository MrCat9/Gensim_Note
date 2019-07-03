# -*- coding: utf-8 -*-


import os
import json
import pandas as pd
from jieba import posseg
from gensim import corpora, models, similarities
# from six import iteritems


class TfIdf(object):
    def __init__(self, csv_dir, stop_words_file_path, give_max_sim_num):
        # ================ 传入参数 ================
        self.csv_dir = csv_dir  # csv 文件目录
        self.stop_words_file_path = stop_words_file_path  # 停用词文件路径  # 考虑升级成目录
        self.MAX_SIM_NUM = give_max_sim_num
        # ==========================================

        # ================ 中间变量 ================
        self.df = ''  # 从csv文件读取出来的内容的DataFrame
        self.stpwrd_list = []
        self.words = ''
        # ==========================================

        # ================ 结果 ================
        # ======================================

    @staticmethod
    def save_dict_to_json(dict_data, json_path='result.json'):
        json_str = json.dumps(dict_data, indent=4)
        with open(json_path, 'w') as json_file:
            json_file.write(json_str)

    @staticmethod
    def make_dict_data_serializable(dict_data):
        new_dict_data = {}
        for _key, _value in dict_data.items():  # 把 dict 中的 value 都转成 str
            if _value:
                new_dict_data[str(_key)] = str(_value)
            else:
                new_dict_data[str(_key)] = ''
        return new_dict_data

    def get_stop_words_list(self, file_path):
        # 从文件导入停用词表
        stpwrdpath = file_path
        stpwrd_dict = open(stpwrdpath, 'r', encoding='utf-8')
        stpwrd_content = stpwrd_dict.read()
        # 将停用词表转换为list
        stpwrd_list = stpwrd_content.splitlines()
        # def byte_to_str(text_byte):
        #     return text_byte.decode('utf-8')  # bytes 转 str
        # stpwrd_list = list(map(byte_to_str, stpwrd_list))
        stpwrd_dict.close()

        self.stpwrd_list = stpwrd_list
        return self.stpwrd_list

    def cut_text(self, text_df):
        text_series = text_df.loc[:, 'text']

        def jieba_cut(text_str):
            # seg_list = jieba.cut(text_str)  # 默认是精确模式
            seg_list = posseg.cut(text_str)
            words_list = []
            for word, nature in seg_list:
                # 考虑将分词分出来的新词添加到 stop word 库里  # 可以根据词性来判断是否加入停用词
                print(word, nature)
                words_list.append(word)
            return words_list

        text_series = text_series.map(jieba_cut)  # 分词
        text_df['text'] = text_series

        self.df = text_df
        return self.df

    def text_preprocess(self, df):
        df = df.drop_duplicates(['id'])
        df = df.dropna(axis=0, how='any', subset=['id', 'text'])  # 删除包含空值的行
        # 删除索引
        # df = df.reset_index(drop=True)
        # 重建索引
        # df = df.reindex(index=range(len(df)))

        self.df = df
        return self.df

    def read_csv_to_df(self, csv_dir):
        files_list = os.listdir(csv_dir)  # 获取文件夹下的所有文件和文件夹

        temp_df = pd.DataFrame()  # 用于连接多个DataFrame的临时df
        for _file in files_list:
            # 从每个csv文件中读出DataFrame，并连接在一起
            # file_path = csv_dir+'/'+_file
            file_path = os.path.join(csv_dir, _file)
            temp_df = temp_df.append(pd.read_csv(file_path, header=None, usecols=[0, 11], names=['id', 'text']), ignore_index=True)

        self.df = temp_df
        return self.df

    def calculate_text_similarity(self):
        # 读出文章集  # 读 csv
        df = self.read_csv_to_df(self.csv_dir)

        # 文本预处理（如：去除空值）
        text_df = self.text_preprocess(self.df)

        # 对正文分词
        cuted_text_df = self.cut_text(self.df)

        # ================================================================

        # 分词结果
        texts = self.df.loc[:, 'text'].tolist()  # [['文章1的分词结果'], ['文章2的分词结果'], ['文章3的分词结果'], ]

        # 建词袋（一个词对应一个id）
        dictionary = corpora.Dictionary(texts)  # 词袋

        # dictionary.token2id  # 每个词对应的id
        # featurenum=len(dictionary.token2id.keys())

        # 获取词袋中的低频词 id
        # dictionary.dfs  # 词频
        # MIN_FREQUENCY_OF_WORD = 1
        # low_frequency_word_ids = [token_id for token_id, token_freq in iteritems(dictionary.dfs) if token_freq <= MIN_FREQUENCY_OF_WORD]

        # 获取停用词
        stpwrd_list = self.get_stop_words_list(self.stop_words_file_path)

        # 获取词袋中的停用词 id
        stop_word_ids = [dictionary.token2id[stopword] for stopword in self.stpwrd_list if stopword in dictionary.token2id]

        # 过滤掉词袋中的 低频词 和 停用词
        # dictionary.filter_tokens(stop_word_ids + low_frequency_word_ids)
        dictionary.filter_tokens(stop_word_ids)  # 过滤掉词袋中的 停用词

        # 去除间隙
        dictionary.compactify()  # remove gaps in id sequence after words that were removed

        # 保存 字典
        dictionary.save('tfidf_file/dictionary.dict')  # store the dictionary, for future reference

        # 将所有文章映射到词袋组成的向量空间 -> 组成语料库（形如：[文章1的映射结果, 文章2的映射结果, ]）
        corpus = [dictionary.doc2bow(text) for text in texts]

        # 保存 语料库
        corpora.MmCorpus.serialize('tfidf_file/corpus.mm', corpus)
        # 读出 语料库
        # corpus = corpora.MmCorpus('tfidf_file/corpus.mm')

        # （TF-IDF）模型 <- 语料库
        tfidf = models.TfidfModel(corpus)  # initialize a model
        # 保存 模型
        # tfidf.save('tfidf_file/model.tfidf')  # same for lsi, lda, ...
        # 读出 模型
        # tfidf = models.TfidfModel.load('tfidf_file/model.tfidf')

        # 所有文章的TF-IDF向量 = 模型[语料库]
        corpus_tfidf = tfidf[corpus]

        # （过滤掉一篇文章TF-IDF值较低的那些纬度）（纬度越低计算余弦相似度越快）

        # 计算文章相似度
        index = similarities.MatrixSimilarity(corpus_tfidf)  # 传入所有文章的TF-IDF向量
        # 保存
        # index.save('tfidf_file/index.index')
        # 读出
        # index = similarities.MatrixSimilarity.load('tfidf_file/index.index')

        # 生成文章索引与id的对应list
        text_id_list = self.df.loc[:, 'id']
        text_id_list = text_id_list.tolist()

        # 一篇文章与所有文章的相似结果
        # sim = index[一篇文章的TF - IDF向量]
        result_dict = {}  # 相似结果 dict
        i = 0  # 当前文章的索引
        for tf in corpus_tfidf:
            sim_array = index[tf]  # 一篇文章与所有文章的相似结果
            sim_array[i] = None  # 将当前文章与本身的相似结果置为 空
            sim_series = pd.Series(sim_array)
            sim_series = sim_series.sort_values(ascending=False, na_position='last')  # 降序排列
            sim_series = sim_series.dropna(axis=0, how='any')  # 删除当前文章与本身的相似结果  # 删除最后一个
            top_sim = sim_series.head(n=self.MAX_SIM_NUM)
            top_sim_dict = top_sim.to_dict()  # 文章 索引:相似度

            top_sim_ids_dict = {}  # 文章 id:相似度
            for key, value in top_sim_dict.items():
                new_key = text_id_list[key]
                top_sim_ids_dict[str(new_key)] = value

            result_dict[str(text_id_list[i])] = top_sim_ids_dict
            i += 1

        # ================================================================

        # 处理dict变量，使其能够作为json文件保存
        result_dict = self.make_dict_data_serializable(result_dict)

        # 将 dict 数据保存到 json 文件中
        self.save_dict_to_json(result_dict)


if __name__ == '__main__':
    give_csv_dir = 'temp_csv_data'
    give_stop_words_file_path = 'stopwords.txt'
    give_max_sim_num = 10
    my_tf_idf = TfIdf(give_csv_dir, give_stop_words_file_path, give_max_sim_num)
    my_tf_idf.calculate_text_similarity()

