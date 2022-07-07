#/bin/bash

for client_num in 2 3 4 5 6
do
  bash Scripts/setup-clients.sh $client_num && \
  bash Scripts/setup-online.sh $client_num 128 128 && \
  bash Scripts/setup-ssl.sh $client_num 128 128 && \
  bash Scripts/setup-online.sh $client_num 192 128 && \
  bash Scripts/setup-ssl.sh $client_num 192
done