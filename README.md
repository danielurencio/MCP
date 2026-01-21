# MCP (Model Construction Planner)

## Time-oriented ML design
A correct machine learning formulation for financial time series problems requires careful handling of out-of-sample data. This is critical for both model selection and the correct interpretation of performance.

Any maintainable solution for training and evaluating machine learning models in this context must, at its core, support the proper splitting of training, validation, and testing sets such that:

* Observations are ordered in time: training data precedes validation data, which in turn precedes test data.
* Overlapping forward returns are addressed by introducing appropriate gaps when using holdout sets for evaluation.

Failure to respect time ordering can introduce unnecessary biases and lead to misleading results. For example, if a dataset is randomly shuffled before being split into training and test sets, observations from the future may end up in the training set while earlier observations are used for testing. This setup does not reflect how time series problems are encountered in real-world financial settings. Financial contexts are inherently path-dependent, relying on information available up to a given point in time. As a result, such data leakage can artificially inflate model performance by incorporating information that would not have been available under normal operating conditions.

### Evaluation and model selection considerations
If targets, $Y$, within a a dataset are expressed as forward returns that overlap in time, one can erroneously count periods that contributed to the calculation of returns more than once. This can misleadingly inflate performance if adjacent observations consider overlapping returns to calculate a loss for the function to be optimized.

Consider the next image, where three observations, denoted by $x_{i}$, have as its target a value, or return, that spans across four time steps in the future. Using this data "as is" for performance evaluation would be similar to tripling a student's grades for answering three question even though these were part of one single exam.
<p align="center">
<img src="imgs/overlapping_targets.jpeg" alt="Description of image" style="width:50%; max-width:200px;">
</p>

If targets $Y$ within a dataset are expressed as forward returns that overlap in time, the same underlying periods can be implicitly counted multiple times. This can lead to misleadingly inflated performance when adjacent observations rely on overlapping returns to compute the loss function being optimized.

Consider the image below, where three observations, denoted by $x_i$, each have a target return spanning four future time steps. Using this data as is for performance evaluation would be analogous to tripling a student’s grade for answering three questions that all belong to a single exam.

<p align="center">
<img src="imgs/gaping_targets.jpeg" alt="Description of image" style="width:85%; max-width:200px;">
</p>

$$
\sum_{i=1}^n X_i
$$


Usage
```python
import warnings
from MCP import Dataset, TaskHandler
warnings.filterwarnings("ignore")


train_start = '2010-01-05'
train_end = '2018-03-27'
test_end = '2021-09-28'

dataset = Dataset.get()
th = TaskHandler('PoC').train_with_best_params(dataset, train_start, train_end, test_end)
```

### Sentiment Scores
```python
commodities_datasets = {k:dataset[dataset.TRADE_12 == k] for k in dataset.TRADE_12.unique()}
commodity_list = list(commodities_datasets.keys())

arr = list()
start = '2010-01-05'
end = '2011-04-12'

for i in range(len(commodity_list)):
    print(commodity_list[i])
    kpis_df = TaskHandler('sentiment_scores').simple_retrain(dataset, start, end)
    arr.append(kpis_df)

pd.concat(arr).to_csv('sentiment_scores_v0.csv', index=False)
```


#### To do:
Introduce gaps in train and test fold sizes.
```python
from datetime import datetime
from dateutil.relativedelta import relativedelta

from MCP import TimeFolder


fmt = '%Y-%m-%d'
dates = list()
date = datetime.strptime(start_date, fmt)
space_to_fill = 0.5
cv_folds = 5

while date <= datetime.strptime(end_date, fmt):
    dates.append(date)
    date += relativedelta(weeks=1)

space_to_fill_index = int(space_to_fill*len(dates))
test_size = int(len(dates[:space_to_fill_index]) / cv_folds)
train_size = len(dates[space_to_fill_index:])

print(train_size, test_size)

time_folder = TimeFolder(train_size=train_size, test_size=test_size, period_type='weeks')
folds = time_folder.get_time_splits(start_date, end_date)

assert cv_folds == len(folds)
```
