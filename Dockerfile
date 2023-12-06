FROM python:3.10-slim
 
WORKDIR /app
 
COPY . /app
RUN pip install --upgrade pip
RUN pip install virtualenv
RUN pip install --no-dependencies transformers==4.10.0
RUN pip install -r requirements.txt
 
EXPOSE 5000
 
ENV NAME World
 
CMD ["python", "app.py"]