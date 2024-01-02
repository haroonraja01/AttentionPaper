from torchtext.datasets import IMDB
from torchtext.transforms import SentencePieceTokenizer

train_iter, test_iter = IMDB()


# def tokenize(label, line):
#     return line.split()


# tokens = []
# for label, line in train_iter:
#     tokens += tokenize(label, line)
#
# tokens_test = []
# for label, line in test_iter:
#     tokens_test += tokenize(label, line)
# print(len(tokens), len(tokens_test))

xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
transform = SentencePieceTokenizer(xlmr_spm_model_path)
#print(transform(["hello world", "attention is all you need!"]))

# Write transforms and data loading from the following link:
# https://pytorch.org/text/0.16.0/tutorials/sst2_classification_non_distributed.html#sphx-glr-tutorials-sst2-classification-non-distributed-py

tokens = []
labels = []
for label, line in train_iter:
    labels.append(label)
    #print(label)
    # tokens += transform(label, line)
print(max(labels), min(labels))
tokens_test = []
for label, line in test_iter:
    tokens_test += transform(label, line)
print(len(tokens), len(tokens_test))
