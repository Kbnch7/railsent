
import requests
from bs4 import BeautifulSoup

data = set()


def crawler():
    response = requests.get('https://www.kp.ru/daily/23458/36682/')
    soup = BeautifulSoup(response.text, "html.parser")

    result = soup.findAll("p", {"class": "sc-1wayp1z-16 dqbiXu"})
    for elem in result:
        data.add(elem.getText())

    print(data)


if __name__ == '__main__':
    crawler()
