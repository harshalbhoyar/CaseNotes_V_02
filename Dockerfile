FROM python:3.11.4-slim-buster

WORKDIR /home/ec2-user/CaseNotes_V_02

RUN pip install --upgrade pip

RUN python3 -m venv venv

RUN source venv/bin/activate

RUN pip install flask

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000 

CMD [ "python", "app.py" ]