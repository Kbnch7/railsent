FROM python:3.8

WORKDIR /app

COPY . /app

# RUN pip install --no-cache-dir pandas numpy tensorflow scikit-learn matplotlib seaborn tqdm transformers nltk

RUN pip install --no-cache-dir -r requirements.txt

RUN ["python", "-c", "import nltk; nltk.download('stopwords')"]

EXPOSE 80


CMD ["python", "./CustomGRUModel.py"]
