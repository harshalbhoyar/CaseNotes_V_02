FROM python:3.11.4-slim-buster

WORKDIR /home/ec2-user/CaseNotes_V_02

RUN python3 -m venv venv

RUN source venv/bin/activate

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "routes.py" ]