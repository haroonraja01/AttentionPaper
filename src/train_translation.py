from torchtext.datasets import IWSLT2016


train_iter, valid_iter, test_iter = IWSLT2016(root='/Users/haroonraja/Desktop/language_translation/',
                                              split=('train', 'valid', 'test'), language_pair=('de', 'en'),
                                              valid_set='tst2013', test_set='tst2014')
print(train_iter)
src_sentence, tgt_sentence = next(iter(train_iter))
print(src_sentence, tgt_sentence)
