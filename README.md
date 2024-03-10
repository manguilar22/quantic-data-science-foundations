# Quantic Data Science Foundations - Capstone Project

### Data Sources 

https://www.slickcharts.com/sp500


#### Stock Index ETFs 

| Symbol | Name              |
|--------|-------------------|
| SPY    | S&P 500 ETF       |	   
| QQQ    | Nasdaq 100 ETF    |
| DIA    | Dow Jones ETF     |
| ONEQ   | Nasdaq Comp ETF   |


#### Creating ML models for stock price prediction

The following command will get the stock symbols from what the portfolio that has been built. The output will be a text file containing the arguments requiered for a python program to build the ML model for predicting future stock prices. 

```bash
for symbol in $(awk -F ',' '{print $1}' data/portfolio.csv); do 

echo "python ml.py --symbol $symbol --period1 $period1 --period2 $period2 --interval $interval" >> jobs.txt

done
```

##### Execute Python jobs

**Note:** Make sure the correct Python environment is being used in your shell session. The *quantic.yml* file contains all libraries required to build the ML modles for each stock. 

```bash
while IFS= read -r job; do 
    eval $job; 
done < jobs.txt
```