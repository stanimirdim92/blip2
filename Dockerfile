FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install  --upgrade -r /code/requirements.txt

#RUN useradd -m -u 1000 user

#USER user

#ENV HOME=/home/user \
#    PATH=/home/user/.local/bin:$PATH


COPY . /code/app

EXPOSE 5000
ENV PORT 5000

#CMD exec uvicorn --port $PORT --workers 1 --host app.main:app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0",  "--port", "5000",  "--workers", "1"]