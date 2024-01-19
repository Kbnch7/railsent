import nltk

data_set_name = input("Введите название файла без расширения:")
text = ''

with open(f"{data_set_name}.txt", "r", encoding="UTF-8") as data:
    for line in data:
        text += line
    list_of_text = nltk.tokenize.sent_tokenize(text, language="russian")

with open("new_data.txt", "a", encoding="UTF-8") as new_data:
    for row in list_of_text:
        new_row = row
        try:
            if (row[0] == '\n' or row[0] == "\r") and len(row)>1:
                new_row = row[1:]
        except Exception:
            continue

        try:
            if row[-1] == "." or row[-1] == "!" or row[-1] == "?":
                new_row = row.capitalize() + "\n"
            else:
                new_row = row.capitalize() + ".\n"
        except:
                new_row = row.capitalize() + ".\n"
        F = True
        for stopword in nltk.corpus.stopwords.words("russian"):
            if stopword.lower() in row.lower() and len(row.split()) < 2:
                F = False
        if F:
            new_data.write(new_row)
