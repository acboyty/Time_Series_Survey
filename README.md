# Time_Series_Survey

TODO:

Neural Prophet: is it a unsupervised method??

We currently focus on selecting small(limited history) univariate(currently) time series datasets, on which GBDTs outperform RNN and N-BEATS in one-step(currently) forecasting.

For example: lookback window size = 50, 1000 instances in total.

## Performance

MAPE/sMAPE

|             | LSTM+Linear             | N-BEATS                   | XGBM                    | LGBM                    | CATB                    | Neural Prophet |
| ----------- | ----------------------- | ------------------------- | ----------------------- | ----------------------- | ----------------------- | -------------- |
| M4(sample)  | 3.16278/3.51589         | 3.56738/3.86817           | **2.38993**/**2.76590** | 4.91277/4.73043         | 2.69940/3.00849         |                |
| M4(sample1) | 4.11026/4.06729         | **3.64101**/**3.59241**   | 3.69822/3.77689         | 5.43398/4.60801         | 14.31157/8.83668        |                |
| M4(sample2) | 4.06871/4.14208         | 3.09411/3.03003           | **2.23061**/2.24960     | 3.97615/3.11812         | 2.24813/**2.16569**     |                |
| ACSF1       | **0.23669**/**0.23674** | 0.63974/0.62632           | 0.27732/0.27852         | 0.50666/0.49931         | 0.24889/0.24873         |                |
| Car         | 9.52988/6.21009         | 8.44799/6.33472           | 8.13189/8.02840         | **6.70177**/**5.61442** | 10.27571/8.06611        |                |
| Fish        | **2.98889**/**2.99318** | 3.98505/4.38288           | 9.19697/8.86786         | 9.97310/9.84769         | 10.48386/8.26835        |                |
| Yoga        | **7.50614**/**7.78231** | 9.15068/10.60429          | 10.41571/9.60754        | 9.63378/9.21118         | 19.19672/17.87246       |                |
| Worms       | 57.81779/22.06355       | 42.61992/**20.48697**     | 46.86538/25.30363       | 46.51857/24.20151       | **34.72143**/23.18958   |                |
| Symbols     | **2.99689**/**2.88291** | 4.10728/4.16476           | 6.90836/6.53102         | 6.99339/6.52555         | 9.56463/8.67842         |                |
| Trace       | 13.11834/13.58730       | 9.85311/9.04429           | 10.75702/10.17314       | 9.72184/10.16684        | **7.46065**/**7.20619** |                |
| Plane       | 20.17367/17.79917       | **15.36909**/**15.84318** | 21.14664/20.62732       | 20.50483/20.07373       | 17.24489/18.10027       |                |
| Meat        | 5.45354/5.58601         | 2.22012/2.26013           | **1.91326**/**1.92622** | 1.98756/2.01198         | 2.66461/2.68946         |                |
| Wine        | 4.37374/3.83479         | 4.10914/2.78318           | 3.89487/2.56973         | **3.02594**/**1.98494** | 6.23243/4.22749         |                |
| ShapesAll   | **9.68632**/**9.10743** | 16.42129/12.75353         | 15.41340/11.54903       | 17.76827/11.59538       | 31.29247/15.63691       |                |

* seems XGBM works well on small TS datasets

## Dataset Info

UCR: [Welcome to the UCR Time Series Classification/Clustering Page](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

|             | all 1000x(50+1), lookback=50, lookforward=1                  | 6-2-2                                   |
| ----------- | ------------------------------------------------------------ | --------------------------------------- |
| M4(sample)  | daily, first 200 rows, 5 step-by-step instances per row      |                                         |
| M4(sample1) | daily, first 200 rows, 5 instances per rows, sampled by 5 steps |                                         |
| M4(sample2) | daily, first 1000 rows, 1 instance per row                   |                                         |
| ACSF1       | 100 rows in total, 10 instances per row, sampled by 100 steps |                                         |
| Car         | first 50 rows, 20 instances per row, sampled by 25 steps     |                                         |
| Fish        | first 100 rows, 10 instances per row, sampled by 40 steps    | datasets from UCR are sampled similarly |

