import nltk
data_set_name = input("Введите название файла без расширения:")
with open(f"{data_set_name}.txt", "r", encoding="UTF-8") as test:
    list_of_text = []

    for count, row in enumerate(test):
        F = True
        for stopword in nltk.corpus.stopwords.words("russian"):
            if stopword in row and len(row.split()) < 2:
                F = False
        if F:
            list_of_text.append(row)

    list_of_text = list(set(list_of_text))

    with open(f"{data_set_name}.txt", "w", encoding="UTF-8") as test:
        for row in list_of_text:
            test.write(row)

