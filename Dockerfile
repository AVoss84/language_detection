FROM tiangolo/uwsgi-nginx-flask:python3.7-alpine3.7
#FROM tiangolo/uwsgi-nginx-flask:python3.7

ENV LISTEN_PORT=5000
EXPOSE 5000
RUN apk --update add --no-cache g++
RUN apk --update add bash nano

#RUN apk update \
#  && apk add --update-cache --no-cache libgcc libquadmath musl \
#  && apk add --update-cache --no-cache libgfortran \
#  && apk add --update-cache --no-cache lapack-dev

# Needed for numpy, sklearn, scipy installation... 
RUN apk update && apk add gfortran build-base openblas-dev libffi-dev

ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY ./requirements.txt /var/www/requirements.txt

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r /var/www/requirements.txt

CMD ["flask", "run", "--host", "0.0.0.0"]
