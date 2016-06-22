# Bitcoin trading bot

Bayesian regression model for predicting changes in the price of Bitcoin,
implemented with Python and sklearn.

The model implemented is similar to the one described in _Bayesian Regression and Bitcoin_ by Shah, Zhang ([arxiv:1410.1231](http://arxiv.org/pdf/1410.1231v1.pdf))

### Usage

#### Training

    python train.py data/training_data.csv

#### Trade

    python trade.py data/testing_data.csv

