FROM tensorflow/tensorflow:1.14.0-py3-jupyter

RUN apt update && apt install \
    wget

RUN pip3 install \
    nltk \
    pandas \
    seaborn \
    sklearn \
    tensorflow-hub \
    requests \
    tensorflow-serving-api