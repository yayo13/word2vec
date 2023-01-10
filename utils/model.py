import torch.nn as nn

from utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        '''
        inputs_: [N, 2R]
                N表示有多少个序列，是在一个batch中所有句子筛选后截取出来的
                R表示每个序列的半径即CBOW_N_WORDS
                注意inputs_内容是每个词在vocab字典中的序号

        self.embeddings.weights: [K, D]
                K表示该数据集中词的数量，即vocab字典的长度
                D表示每个词向量化后的维度
                因此self.embeddings.weights第i行是vocab中第i个词的向量
        
        self.embeddings操作就是根据inputs_表示的每个词的序号索引得到其向量
                其输出是[N, 2R, D]

        x.mean 就是将上下文（2R）个词的向量取平均

        self.linear 就是做xA+b线性变换，输出为[N, K]

        整体看模型的功能是对输入的每个词先做向量化再做分类，当然算法的意图是获得self.embeddings.weights
        '''
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x