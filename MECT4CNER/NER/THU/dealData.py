import tqdm

with open("input_THUCNews.txt", "r", encoding="utf-8") as f:
    text_context = f.read()
    text_context = text_context.replace("/0", "")
    with open("THUCNews.bmes", "w", encoding="utf-8") as f1:
        for i in tqdm.tqdm(text_context):
            if i == " ":
                f1.write(", O\n")
            elif i != "\n":
                f1.write(i+" O\n")
            else:
                f1.write("。 O\n\n")