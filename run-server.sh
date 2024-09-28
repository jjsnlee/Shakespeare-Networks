#!/bin/bash

cd py-server
IP_ADDRESS='localhost'

if [ -z "${RNLP_PORT}" ];
then
  #echo "var is unset";
  RNLP_PORT="8401"
fi

export PYTHONDONTWRITEBYTECODE=1

if [ "$1" = "--daemon" ];
then
  nohup python manage.py runserver "$IP_ADDRESS:$RNLP_PORT" &>> ../../logs/server.log&
  #cd ../; nohup uwsgi py-server/cfg/wsgi.ini &>> logs/server.log&
else
  python manage.py runserver "$IP_ADDRESS:$RNLP_PORT"
  #python -Wd manage.py runserver "$IP_ADDRESS:$RNLP_PORT"
  # cd ../; uwsgi py-server/cfg/wsgi.ini
fi
