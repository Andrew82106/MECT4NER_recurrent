# to wiki.bmes
import tqdm

cnt = 0
with open("input_wiki.txt", "r", encoding="utf-8") as f:
    text_context = f.read()
    text_context = text_context.replace("0", "").replace("\n", "").replace(" ", "")
    with open("wiki.bmes", "w", encoding="utf-8") as f1:
        for i in tqdm.tqdm(text_context):
            if i == "，" or i == "、" or i == " " or i == ' ':
                f1.write(", O\n")
            elif i == '。':
                f1.write(i+" O\n\n")
            elif i != "\n":
                f1.write(i+" O\n")
            else:
                f1.write("。 O\n\n")
            cnt += 1
"""            if cnt > 32477:
                break"""