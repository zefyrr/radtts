FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN apt-get update -y
ENV TZ=America/Los_Angeles
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y ffmpeg libsndfile1 sox locales vim
RUN pip3 install -U numpy
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /src