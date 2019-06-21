# -*- coding: utf-8 -*-

import collections
from functools import reduce


if __name__ == '__main__':

    list1 = [['你好', 'hello'], ['测试', 'test'], ['你', '你好']]  # 分词结果  每个list是一篇文章

    def list_add(x, y):
        return x + y

    list1 = reduce(list_add, list1)

    # 词频统计
    word_counts = collections.Counter(list1)  # 对分词做词频统计
    word_counts_top3 = word_counts.most_common(3)  # 获取前3最高频的词

    print(word_counts)  # Counter({'你好': 2, 'hello': 1, '测试': 1, 'test': 1, '你': 1})
    print(word_counts_top3)  # [('你好', 2), ('hello', 1), ('测试', 1)]
