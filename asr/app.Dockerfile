FROM python:3.9-slim

ARG REPO_DIR="."
ARG CONDA_ENV_NAME="htx_app"
ARG CONDA_HOME="/miniconda3"
ARG MINICONDA_SH="Miniconda3-${CONDA_VER}-${CONDA_ARCH}.sh"
RUN apt-get update && apt-get install -y \
	build-essential \
	libopencv-dev \
	python3-pip \
        && apt-get clean && rm -rf /tmp/* /var/tmp/* \
    build-essential wget curl \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN curl -O https://repo.anaconda.com/miniconda/${MINICONDA_SH} && \
    chmod +x ${MINICONDA_SH} && \
    ./${MINICONDA_SH} -u -b -p ${CONDA_HOME} && \
    rm ${MINICONDA_SH}  # Clean-up the Miniconda installer
ENV PATH="${CONDA_HOME}/bin:$PATH"
RUN conda create --name htx_app python=3.9 -y
SHELL ["/bin/bash", "-c", "source /opt/miniconda3/bin/activate htx_app &&"]

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt
RUN conda install --yes pip && pip install -r /app/requirements.txt

COPY ./asr/src 	 /app/src/
COPY ./asr/conf  /app/conf/
COPY ./asr/src /app/src/
COPY ./asr/conf /app/conf/
COPY ./asr/main.py /app/main.py

ENV APP_ENV=production
LABEL service="asr-api"
EXPOSE 8000

ENTRYPOINT ["python3", "main.py"]