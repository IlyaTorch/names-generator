FROM python:3.7-slim

ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt
COPY . /app

RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

WORKDIR /app
