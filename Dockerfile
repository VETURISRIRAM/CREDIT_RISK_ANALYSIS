FROM python:3

RUN mkdir -p /application
COPY 'data.csv' /application/

ADD main.py /application

RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install sklearn
RUN pip install imblearn
RUN pip install statsmodels

CMD [ "python", "./application/main.py" ]