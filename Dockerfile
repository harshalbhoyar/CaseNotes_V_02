FROM python:3.11.4-slim-buster

WORKDIR /home/ubuntu/CaseNotes_V_02

RUN pip install flask

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "app.py" ]

