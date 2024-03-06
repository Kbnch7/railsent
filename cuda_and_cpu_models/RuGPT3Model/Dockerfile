FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install --upgrade transformers && \
    pip install --no-cache-dir -r requirements.txt
    
CMD ["python", "RuGPT3model.py"]
