# Time_Series_Survey

TODO:

hyperparameter tuning and more datasets(more sampling method to M4)

We currently focus on selecting small(limited history) univariate(currently) time series datasets, on which GBDTs outperform RNN and N-BEATS in one-step(currently) forecasting.

For example: lookback window size = 50, 1000 instances in total.

## performance

MAPE/sMAPE

|            | LSTM+Linear     | N-BEATS         | XGBM            | LGBM            | CATB            |
| ---------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| M4(sample) | 3.16278/3.51589 | 3.56738/3.86817 | 2.38993/2.76590 | 4.91277/4.73043 | 2.69940/3.00849 |
|            |                 |                 |                 |                 |                 |
|            |                 |                 |                 |                 |                 |

## dataset info

