FROM nvcr.io/nvidia/tensorrt:23.01-py3

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt 
ENV LISTEN_PORT=5000
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
CMD ["python", "app.py"]
