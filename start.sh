#!/bin/bash
app="language-identifier"

docker build -t ${app} .

docker run -d -p 56733:80 \
  --name=${app} \
  -v $PWD:/app ${app}
