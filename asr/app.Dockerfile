FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3-distutils \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /asr

COPY ./requirements.txt /asr/requirements.txt
RUN pip3 install -r requirements.txt

RUN mkdir -p /data/
COPY ./asr/src /asr/src/
COPY ./asr/conf /asr/conf/
COPY ./asr/.env /asr/.env
COPY ./asr/main.py /asr/main.py

LABEL service="asr-api"
EXPOSE 8001

ENTRYPOINT ["python3", "main.py"]