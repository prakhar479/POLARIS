FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 3001 8089 5000
CMD ["python3", "Node.py"]
