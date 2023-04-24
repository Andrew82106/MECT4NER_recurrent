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
        if dataset == 'thu':
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

- path.py line 19:

```python
# add:
dark_data_path = '/home/ws/NER/DarkData'.replace(old_Loc, rootLoc)
thu_data_path = '/home/ws/NER/THU'.replace(old_Loc, rootLoc)
```

- 添加了BertWordEncoding.py，用于在MECT4CNER的输出后套一层处理，来得到语句中的词向量
- 在predict.py中添加了字向量输出，结合BertWordEncoding.py来获得文本中的词向量
- 在NER数据集中添加了darkData数据集，用于打标
- PosFusionEmbedding.py line 17-29:

```python
# from
    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        max_seq_len = pos_s.size(1)
        pe_ss = self.pe_ss[(pos_ss).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
# to
    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        max_seq_len = pos_s.size(1)
        tensor_ss = (pos_ss).view(-1) + self.max_seq_len
        tensor_ee = (pos_ee).view(-1) + self.max_seq_len
        tensor_ss[tensor_ss >= len(self.pe_ss)] = len(self.pe_ss) - 2
        tensor_ee[tensor_ee >= len(self.pe_ee)] = len(self.pe_ee) - 2
        pe_ss = self.pe_ss[tensor_ss].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[tensor_ee].view(size=[batch, max_seq_len, max_seq_len, -1])
```
