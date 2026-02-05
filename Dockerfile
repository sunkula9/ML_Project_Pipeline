FROM python:3.12-slim-buster

WORKDIR /app
# Install system dependencies
RUN apt-get update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python", "app.py"]