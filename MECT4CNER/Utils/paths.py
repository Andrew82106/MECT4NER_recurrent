rootLoc = "/Users/andrewlee/Desktop/Projects/hmn/MECT4CNER-master/MECT4NER_NEW"
old_Loc = "/home/ws"


yangjie_rich_pretrain_unigram_path = '/home/ws/data/gigaword_chn.all.a2b.uni.ite50.vec'.replace(old_Loc, rootLoc)
yangjie_rich_pretrain_bigram_path = '/home/ws/data/gigaword_chn.all.a2b.bi.ite50.vec'.replace(old_Loc, rootLoc)
yangjie_rich_pretrain_word_path = '/home/ws/data/ctb.50d.vec'.replace(old_Loc, rootLoc)

# this path is for the output of preprocessing
yangjie_rich_pretrain_char_and_word_path = '/home/ws/data/yangjie_word_char_mix.txt'.replace(old_Loc, rootLoc)

# This is the path of the file with radicals
radical_path = '/home/ws/data/char_info.txt'.replace(old_Loc, rootLoc)

ontonote4ner_cn_path = "/home/ws/NER/People's Daily".replace(old_Loc, rootLoc)
msra_ner_cn_path = '/home/ws/NER/MSRA'.replace(old_Loc, rootLoc)
resume_ner_path = '/home/ws/NER/ResumeNER'.replace(old_Loc, rootLoc)
weibo_ner_path = '/home/ws/NER/Weibo'.replace(old_Loc, rootLoc)
