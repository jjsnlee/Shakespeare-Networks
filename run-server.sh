#!/bin/bash

cd py-server
#export PYTHONPATH=../../mod-ext-agtm/mod-agtm/src:\
#../../py-momc/src:\
#../../py-impls/src:\
#../../py-builder/src:\
#$PYTHONPATH

#IP_ADDRESS=`hostname -I`
# from http://stackoverflow.com/questions/369758/how-to-trim-whitespace-from-a-bash-variable
#IP_ADDRESS="$(sed -e 's/[[:space:]]*$//' <<<${IP_ADDRESS})"

# http://www.cyberciti.biz/faq/how-to-find-out-the-ip-address-assigned-to-eth0-and-display-ip-only/
#IP_ADDRESS=`/sbin/ifconfig eth0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'`

#IP_ADDRESS=`/sbin/ifconfig wlp2s0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'`
IP_ADDRESS='localhost'

if [ -z "${RNLP_PORT}" ];
then
  #echo "var is unset";
  RNLP_PORT="8401"
fi

if [ "$1" = "--daemon" ];
then
  nohup python manage.py runserver "$IP_ADDRESS:$RNLP_PORT" &>> ../../logs/server.log&
  #cd ../; nohup uwsgi py-server/cfg/wsgi.ini &>> logs/server.log&
else
  python manage.py runserver "$IP_ADDRESS:$RNLP_PORT"
  #python -Wd manage.py runserver "$IP_ADDRESS:$RNLP_PORT"
  # cd ../; uwsgi py-server/cfg/wsgi.ini
fi
