# https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

count = 0

with open('./dataset/Sarcasm_Headlines_Dataset.json', 'r') as f:
    for line in f:
        print(line.strip())
        count += 1
        if count==5:
            break
