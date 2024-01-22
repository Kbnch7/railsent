
import requests
from bs4 import BeautifulSoup

data = set()
link = input("Ссылка на сайт: ")
html_class = input("Класс html: ")
html_tag = input("Тэг html: ")  # p, a, div, etc


def crawler():
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")

    result = soup.findAll(html_tag, {"class": html_class})
    for elem in result:
        data.add(elem.getText())

    print(data)


if __name__ == '__main__':
    crawler()
    
