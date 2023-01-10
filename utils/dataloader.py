import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2, WikiText103

from utils.constants import (
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)


def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    英文分词器
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer


def get_data_iterator(ds_name, ds_type, data_dir):
    """
    torch中之前的用法是Dataset+DataLoader或iterableDataset+DataLoader，前者是map-style后者是iterable-style
    map-style适合已知所有样本的情况，同时在ddp中可以用torch.utils.data.Sample来控制采样
    iterable-style适合实时产生的流式数据，但不支持Sample采样，在多进程读取时要单独处理防止数据重复加载

    从torch1.11开始引入了TorchData对标tensorflow的tf.data，其提供DataPipes来替代现有的Dataset
    在新版本中用法是MapDataPipes+DataLoader或IterDataPipes+DataLoader，前者是map-style后者是iterable-style

    更详细分析见https://www.guyuehome.com/39806
    """
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = to_map_style_dataset(data_iter)
    return data_iter


def build_vocab(data_iter, tokenizer):
    """ Builds vocabulary from iterator

        以分词后的列表为输入，统计每个词的频率，保留大于等于min_freq的词，额外添加specials定义的词
        vocab.get_itos()返回按词频从大到小排列的list
        vocab.get_stoi()返回dict，其key为词value是该词在get_itos()返回list的序号
        
        e.g:
        >> tokens = ["e", "d","d", "c", "b", "b", "b", "a"]
        >> vocab = build_vocab_from_iterator(tokens, min_freq=2, specials=["<unk>", "<eos>"],  special_first=False)
        >> print(vocab.get_itos())
        ['b', 'd', '<unk>', '<eos>']
        >> print(vocab.get_stoi())
        {'<eos>': 3, '<unk>': 2, 'b': 0, 'd': 1}
    """
    
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=["<unk>"],
        min_freq=MIN_WORD_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        # 句子过滤，长度小于2R+1的忽略
        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        # 长度大于阈值的句子截断
        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS) # 取中间的词作为输出
            input_ = token_id_sequence                   # 两边的词（上下文）作为输入
            batch_input.append(input_)
            batch_output.append(output)

    # batch_size表示一次处理多少句子，每个句子再按2R+1长度做拆分
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, text_pipeline):
    """
    Collate_fn for Skip-Gram model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=SKIPGRAM_N_WORDS past words 
    and N=SKIPGRAM_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is a middle word.
    Each element in `batch_output` is a context word.
    """
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS) # 取中间的词作为输入
            outputs = token_id_sequence                      # 两边的词（上下文）作为输出

            for output in outputs:
                batch_input.append(input_)
                batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
    model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None
):

    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
        
    # 这里x是data_iter出来的一行文字
    # 首先做分词形成词语列表，vocab(['w1', 'w2',...])返回词w1,w2的索引列表
    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab
    