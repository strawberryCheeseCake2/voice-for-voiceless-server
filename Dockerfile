FROM python:3.12.4

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# docker build -t vfv .
# docker run -d --name vfvcontainer -p 80:80 vfv

# some setting in azure 

# docker build -t {SERVER_NAME}.azurecr.io/{CONTAINER_NAME}:build-tag-1 .
# docker push {SERVER_NAME}.azurecr.io/{CONTAINER_NAME}:build-tag-1