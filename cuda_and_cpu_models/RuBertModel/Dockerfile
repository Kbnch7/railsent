FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -m nltk.downloader stopwords

COPY . /app

CMD ["sh", "-c", "python RuBertModel.py && tail -f /dev/null"]
