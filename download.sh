#!/bin/bash

period1=1514815200
period2=1709586000
interval="1d";

for symbol in $(cat data/mutual_funds.json | jq -r '.[]'); do

  echo "downloading symbol: $symbol";
  echo "command: go-stock-price-prediction -symbol $symbol -period1 $period1 -period2 $period2 -interval $interval";
  go-stock-price-prediction -symbol $symbol -period1 $period1 -period2 $period2 -interval $interval;

done

for symbol in $(cat data/symbols.json | jq -r '.[]'); do

  echo "downloading symbol: $symbol";
  echo "command: go-stock-price-prediction -symbol $symbol -period1 $period1 -period2 $period2 -interval $interval";
  go-stock-price-prediction -symbol $symbol -period1 $period1 -period2 $period2 -interval $interval;

done
