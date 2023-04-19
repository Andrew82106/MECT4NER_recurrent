# MECT4NER代码的复现和修改

MECT4NER代码：https://github.com/CoderMusou/MECT4CNER

MECT4NER论文：https://arxiv.org/abs/2107.05418


# changeLogs


- StaticEmbedding.py lines 309-313:

```python
# from
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
# to
        if hasattr(self, 'words_to_words'):
            try:
                words[words >= len(self.words_to_words)] = 1  # 增加了这一行，显然会导致数据打标错误，但是没办法，数据不知道为啥要溢出
                words = self.words_to_words[words]
            except Exception as e:
                raise Exception(f"{e}\n\n ------self.word::{words}\n\nself.word-1{words-1} ------\n\n:: Msg:: \n\n ::word={words}\n\n ::len(self.words_to_words)={len(self.words_to_words)}\n\n ::self.words_to_words={self.words_to_words}")
```

- model.py lines 134-142:

```python
# from
        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}

            return result
# to
        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}

            return output
            # 修改上面这行，从而在非训练模式下将output输出。output为未经过线性层的raw数据。
```
