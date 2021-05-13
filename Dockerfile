FROM python:3.8.10-buster

COPY TaxiFareModel / TaxiFareModel

RUN pip install --upgrade pip
RUN pip install -r TaxiFareModel/requirements.txt

CMD uvicorn TaxiFareModel.api.fast:app --host 0.0.0.0
