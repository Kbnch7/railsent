FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN ["python", "-m", "nltk.downloader", "stopwords"]

CMD ["python", "CostomCCN_GRUModel.py"]
