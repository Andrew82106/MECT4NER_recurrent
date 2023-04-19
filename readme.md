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

            return {'pred': output}
            # 修改上面这行，从而在非训练模式下将output输出。output为未经过线性层的raw数据。
```

- AdaptSelfAttention.py lines 37-43:

```python
# from
        if dataset == 'weibo':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 320, 320), requires_grad=True)
        if dataset == 'msra':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 310, 310), requires_grad=True)
        if dataset == 'resume':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 344, 344), requires_grad=True)
        if dataset == 'ontonotes':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 477, 477), requires_grad=True)
        if dataset == 'tc':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 398, 398), requires_grad=True)
# to 
        if dataset == 'weibo':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 320, 320), requires_grad=True)
        if dataset == 'dark_data':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 320, 320), requires_grad=True)
        if dataset == 'msra':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 310, 310), requires_grad=True)
        if dataset == 'resume':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 344, 344), requires_grad=True)
        if dataset == 'ontonotes':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 477, 477), requires_grad=True)
        if dataset == 'tc':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 398, 398), requires_grad=True)
```
