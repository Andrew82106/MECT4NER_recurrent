# 基于MECT4CNER修改的版本

原仓库：[MECT4CNER_Repo](https://github.com/CoderMusou/MECT4CNER)

## dev log

- 2023.8.23 开始尝试对英文词汇添加词根拆分功能。运行中英混搭数据集命令：

```bash
python main.py --dataset demo
```

- 2023.9.3 调试复盘

思考了一下，发现之前处理英文单词的思路不太对。

之前的思路是在CNN偏旁embedding中对英文单词去找词根，达到和偏旁部首类似的效果

但是在整个模型中lattice embedding部分模型也会把英文单词拆开，所以上面单独找词根其实是无用功

而且单独找词根还会出现句子长度在lattice embedding和radical embedding中不一致的情况，一边是把单词直接拆成字母，另一边是把单词当做一个汉字

所以正确的做法应该是将单词拆成字母直接放进去

所以应该一开始在处理数据集的时候就把单词拆开

哎搞了这么久原来是这个的原因，有点气的

不过这个自己做的数据集总是会报错说``Invalid instance which ends at line:xxx has been dropped``

找了找，应该是数据集的问题，数据集中有爬取的时候没处理好的nbsp文件，把数据集处理一下就好了
