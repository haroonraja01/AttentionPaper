from torchtext.datasets import IMDB
import torchtext
from torchtext.data import get_tokenizer
import torch.nn as nn
import torch
from collections import defaultdict

# train_iter, valid_iter, test_iter = IWSLT2017(root='/Users/haroonraja/Desktop/language_translation/',
#                                               split=('train', 'valid', 'test'), language_pair=('de', 'en'))
train_data, test_iter = IMDB(root='/Users/haroonraja/Desktop/language_translation/')
train_iter = iter(train_data)

tgt_sentence, src_sentence = next(train_iter)

vocab = set(src_sentence.split())

word_to_ix = defaultdict(int)

j = 0
for i, v in enumerate(vocab):
    if v.lower() not in word_to_ix.keys():
        word_to_ix[v.lower()] = j
        j += 1
print(word_to_ix)

embeds = nn.Embedding(len(vocab), 10)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["rented"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

#
# runs = 10
# run = 1
# while run < runs:
#     tgt_sentence, src_sentence = next(train_iter)
#     tokenizer = get_tokenizer("basic_english")
#     tokens = tokenizer(src_sentence)
#     print(tokens)
#     run += 1

# while train_iter:
#     src_sentence, tgt_sentence = next(train_iter)
#     print(src_sentence)
