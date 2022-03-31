#!/bin/bash
# This script copy the required workloads, request json, and request client.py to the clients

shell_path=$(cd "$(dirname "$0")";pwd)

while getopts "p:n:" opt
do
    case $opt in
            n)
                client_num=${OPTARG}
                ;;
            *)
                echo "Usage: ./switch_n_client.sh -n <client_num>"
            exit 1;;
    esac
done

cp Programs/Source/multiversion_client_num/lime_$client_num.mpc Programs/Source/lime.mpc && \
cp Programs/Source/multiversion_client_num/linear_regression_$client_num.mpc Programs/Source/linear_regression.mpc && \
cp Programs/Source/multiversion_client_num/logistic_regression_$client_num.mpc Programs/Source/logistic_regression.mpc && \
cp Programs/Source/multiversion_client_num/vfl_decision_tree_$client_num.mpc Programs/Source/vfl_decision_tree.mpc && \
./compile.py Programs/Source/logistic_regression.mpc && \
./compile.py Programs/Source/linear_regression.mpc && \
./compile.py Programs/Source/lime.mpc && \
./compile.py Programs/Source/vfl_decision_tree.mpc && \
bash Scripts/setup-clients.sh $client_num && \
bash Scripts/setup-online.sh $client_num 128 128 && \
bash Scripts/setup-ssl.sh $client_num 128 128 && \
bash Scripts/setup-online.sh $client_num 192 128 && \
bash Scripts/setup-ssl.sh $client_num 192
