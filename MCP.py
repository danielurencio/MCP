import warnings
warnings.filterwarnings("ignore")

import yaml
from pathlib import Path
from functools import reduce
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

from sklearn.metrics import (
    precision_recall_curve,
    auc,
    mean_absolute_error,
)


class Dataset:
    # Files
    path = Path('data')
    features_file = path / 'features_data_ml_lab.parquet'
    targets_file = path / 'target_data_ml_lab.parquet'
    # Relevant columns
    index = 'TRADE_ID'
    time_col = 'DATE'
    date_fmt = '%Y-%m-%d'
    original_target = 'TARGET_VOL_NORM_RETURN_OIM_5D'

    @classmethod
    def preprocessing(cls):
        features = pd.read_parquet(cls.features_file) 
        labels = pd.read_parquet(cls.targets_file)

        # Join targets
        dataset = features.join(labels[[cls.original_target]])

        # Find records with null targets and discard them
        null_target_records = dataset[dataset[cls.original_target].isna()]
        dataset.drop(null_target_records.index, inplace=True)
        
        cls.dataset = dataset.reset_index().set_index(cls.index)
  
    @staticmethod
    def norm_to_classes(a, b):
        def fn(x):
            if x < a:
                return 0
            elif x >= a and x <= b:
                return 1
            elif x > b:
                return 2
            else:
                return None
                
        return fn

    @classmethod
    def init_transformations(cls):
        # Target transformations
        cls.y_transformations = dict(
            norm_distribution_classes={
                'fn':cls.norm_to_classes,
                'input_args': [-0.43, 0.043]
            }
        )

    @classmethod
    def target_transformations(cls):

        for item in cls.y_transformations.items():
            name, fn_info = item
        
            if fn_info.get('input_args'):
                fn = fn_info['fn'](*fn_info['input_args'])
            else:
                fn = fn_info['fn']
        
            cls.dataset.loc[:, name] = cls.dataset[cls.original_target].map(fn)

    @classmethod
    def get(cls):
        cls.preprocessing()
        cls.init_transformations()
        cls.target_transformations()
        return cls.dataset


def AUCPR_with_threshold(y_true, y_pred, n_classes=2, percentile=70):
    def fn(ix):
        y_true_ = (y_true == ix).astype(int)
        y_pred_ = y_pred[:, ix]
    
        threshold = np.percentile(y_pred_, percentile)
        mask = y_pred_ >= threshold
    
        y_true_filtered = y_true_[mask]
        y_pred_filtered = y_pred_[mask]
    
        precision, recall, _ = precision_recall_curve(y_true_filtered,
                                                      y_pred_filtered)
        return auc(recall, precision)

    auc_per_class = list(map(fn, range(n_classes)))
    avg_classes_auc = np.mean(auc_per_class)
    std_classes_auc = np.std(auc_per_class)

    return avg_classes_auc - std_classes_auc


loss_fns = dict(
    AUCPR_with_threshold=dict(fn=AUCPR_with_threshold, maximize=True),
    mean_absolute_error=dict(fn=mean_absolute_error, maximize=False)
)


class ModelObject:
    def __init__(self, algorithm_choice, problem_type, hyperparams):
        # Support for other libraries can be expaned by modifying this dict 
        self.options = {
            'xgboost': {
                    'object': xgb,
                    'regression': {
                        'attribute': 'XGBRegressor',
                        'objective': 'reg:squarederror',
                        'predict_fn': 'predict'
                    },
                    'binary_classification': {
                        'attribute': 'XGBClassifier',
                        'objective': 'binary:logistic',
                        'predict_fn': 'predict_proba'
                    },
                    'multiclass_classification': {
                        'attribute': 'XGBClassifier',
                        'objective': 'multi:softprob',
                        'predict_fn': 'predict_proba'
                    }
            }
        }
        # We'll need these for the __str__ dunder method
        self.algorithm_choice = algorithm_choice
        self.problem_type = problem_type
        self.hyperparams = hyperparams

        # These are the arguments, or inputs, for the model instance
        obj = self.options[algorithm_choice]['object']
        obj_attrs = self.options[algorithm_choice][problem_type]
        params = {'objective': obj_attrs['objective'], **hyperparams}

        # Necessary for fit and predict methods
        self.predict_fn = obj_attrs['predict_fn']
        self.model = getattr(obj, obj_attrs['attribute'])(**params)

    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return getattr(self.model, self.predict_fn)(X)

    def __repr__(self):
        obj = f'ModelObject("{self.algorithm_choice}", "{self.problem_type}", hyperparams)'
        return f'hyperparams={self.hyperparams}\n{obj}'



class TimeFolder:
    def __init__(self,
                 train_size=6,
                 test_size=2,
                 gap=0,
                 fmt='%Y-%m-%d',
                 period_type='months'):
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.fmt = fmt
        self.period_type = period_type
    
    def convert_to_datetime(self, date):
        return datetime.strptime(date, self.fmt)
    
    def add_periods(self, date, units):
        return date + relativedelta(**{self.period_type:units})
    
    def get_time_splits(self, start_date, end_date, arr=None):
        if arr is None:
            arr = []
        
        start_date_dt = self.convert_to_datetime(start_date)
        end_date_dt = self.convert_to_datetime(end_date)
        train_period_end = self.add_periods(start_date_dt, self.train_size)
        test_period_start = self.add_periods(train_period_end, self.gap)
        test_period_end = self.add_periods(test_period_start, self.test_size)
        
        if test_period_end > end_date_dt:
            return arr
        
        arr.append([start_date_dt, train_period_end, test_period_end])
        new_start_date_dt = self.add_periods(start_date_dt, self.test_size).strftime(self.fmt)
        
        return self.get_time_splits(new_start_date_dt, end_date, arr)

    @classmethod
    def sizes_calculator(cls,
                         start_date,
                         end_date,
                         cv_folds,
                         space_to_fill,
                         period_type,
                         periods,
                         fmt='%Y-%m-%d',
                         dates=None):

        if not dates:
            dates = list()

        date = datetime.strptime(start_date, fmt)

        while date <= datetime.strptime(end_date, fmt):
            dates.append(date)
            date += relativedelta(**{period_type:periods})

        space_to_fill_index = int(space_to_fill*len(dates))
        test_size = int(len(dates[:space_to_fill_index]) / cv_folds)
        train_size = len(dates[space_to_fill_index:])

        return train_size, test_size


class ModelTrainer:
    def __init__(self, time_col, feature_columns, target, ix):
        self.time_col = time_col
        self.feature_columns = feature_columns
        self.target = target
        self.ix = ix

    @staticmethod
    def get_datasets(df, time_col, ix):
        def fn(d):
            train = df[(df[time_col] >= d[0]) & (df[time_col] < d[1])].reset_index()
            valid = df[(df[time_col] >= d[1]) & (df[time_col] < d[2])].reset_index() # Important!
            return train.set_index(ix), valid.set_index(ix)
        return fn
    
    def prepare_dataset(self, df):
        X = df[self.feature_columns].copy()
        y = df[self.target].copy()
        X = X.fillna(-1)
    
        return X, y
    
    def forward_pass(self, model, datasets):
        train_dataset, valid_dataset = datasets
        
        X_train, y_train = self.prepare_dataset(train_dataset)
        X_valid, y_valid = self.prepare_dataset(valid_dataset)
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        return y_valid, y_pred
    
    
    def retrain(self, df_train, folds, model, evaluator_fn, n_classes=None):
        metrics, predictions, index = [], [], []

        for datasets in map(self.get_datasets(df_train, self.time_col, self.ix), folds):
            if any(not len(d) for d in datasets):
                continue

            if n_classes and datasets[0][self.target].nunique() < n_classes:
                continue
                
            model_fold = ModelObject(
                model.algorithm_choice,
                model.problem_type,
                model.hyperparams
            )
            
            y_true, y_pred = self.forward_pass(model_fold, datasets)
           
            if model.problem_type == 'regression':
                metric = evaluator_fn(y_true, y_pred)
            else:
                metric = evaluator_fn(y_true, y_pred, n_classes=n_classes)
                          
            metrics.append(float(metric))
            predictions += y_pred.tolist()
            index += [list(d) for d in datasets[-1].index.tolist()] # Validation Set's index
    
        return metrics, folds, index, predictions


class TimeCrossValidator:
    def __init__(self,
                 n_trials,
                 algorithm_choice,
                 problem_type,
                 folds,
                 h_ranges,
                 time_col,
                 target,
                 ix,
                 feature_columns):

        self.n_trials = n_trials
        self.algorithm_choice = algorithm_choice
        self.problem_type = problem_type
        self.folds = folds
        self.h_ranges = h_ranges
        self.time_col = time_col
        self.target = target
        self.ix = ix
        self.feature_columns = feature_columns

    @staticmethod
    def hyperparam_setter(trial, name, data_type, lower_bound, upper_bound):
        if data_type == 'int':
            return getattr(trial, f'suggest_{data_type}')(name, lower_bound, upper_bound)
        elif data_type == 'float':
            return getattr(trial, f'suggest_{data_type}')(name, lower_bound, upper_bound, log=True)
        else:
            raise ValueError(f'Unsupported DataType: {data_type}')
    
    def cross_validation(self, df_train, evaluator_fn):
        is_regression = self.problem_type == 'regression'
        multiplier = 1 if is_regression else -1
        n_classes = None if is_regression else df_train[self.target].nunique()
        
        def objective(trial):
            # NOTE: We parameterize hyperparams to support many algorithms
            hyperparams = {
                h['name']: self.hyperparam_setter(trial,
                                                  h['name'],
                                                  h['data_type'],
                                                  h['lower_bound'],
                                                  h['upper_bound'])
                for h in self.h_ranges
            }    
    
            model = ModelObject(self.algorithm_choice,
                                self.problem_type,
                                hyperparams)
            
            trainer = ModelTrainer(time_col=self.time_col,
                                   feature_columns=self.feature_columns,
                                   target=self.target,
                                   ix=self.ix)
        
            metrics, _, _, _ = trainer.retrain(df_train,
                                               self.folds,
                                               model,
                                               evaluator_fn,
                                               n_classes=n_classes)
    
            avg_ = np.mean(metrics)
            std_ = np.std(metrics)
    
            # NOTE: In regression we mustn't subtract std.
            return avg_ + (multiplier * std_)
            
        return objective
    
    def perform_optuna_optimization(self, df_train, eval_fn):
        direction = 'minimize' if self.problem_type == 'regression' else 'maximize'
        
        study = optuna.create_study(direction=direction,
                                    sampler=optuna.samplers.TPESampler())

        study.optimize(self.cross_validation(df_train, eval_fn),
                       n_trials=self.n_trials,
                       catch=(ValueError, RuntimeError),
                       show_progress_bar=True)
    
        return study




class TaskHandler:
    def __init__(self, model_name):
        yaml_file = open(f'models/{model_name}.yaml').read()
        self.config = yaml.safe_load(yaml_file)
        # ---------------------------------------------------------
        # GLOBAL CONFIG
        # ---------------------------------------------------------
        self.algorithm_choice = self.config['algorithm_choice']
        self.problem_type = self.config['problem_type']
        self.target = self.config['target']
        self.time_col = self.config['time_col']
        self.ix = self.config['ix']
        self.feature_columns = self.config['feature_columns']
        self.evaluator_fn = loss_fns[self.config['loss']]['fn']

        # ---------------------------------------------------------
        # TIME FOLDS
        # ---------------------------------------------------------
        if self.config.get('retrain_info'):
            self.train_size = self.config['retrain_info']['train_size']
            self.test_size = self.config['retrain_info']['test_size']
            self.period_type = self.config['retrain_info']['period_type']
            self.gap = self.config['retrain_info'].get('gap')
            self.gap = self.gap if self.gap else 0

            tf_kwargs = dict(train_size=self.train_size,
                             test_size=self.test_size,
                             gap=self.gap,
                             period_type=self.period_type)

            self.time_folder = TimeFolder(**tf_kwargs)

    @staticmethod
    def idempotent_list(x):
        return x if isinstance(x, list) else [x]

    @staticmethod
    def concat_arrs(fn):
        def inner_fn(a, b):
            return fn(a) + fn(b)
        return inner_fn
    
    @staticmethod
    def range_indexer(i, arr):
        len_arr = len(arr)
        return [arr[j][i] for j in range(len_arr)]
    
    
    def parse_kpis_df(self, kpis):
        n_kpis = len(kpis)
        kpis_content = [reduce(self.concat_arrs(self.idempotent_list),
                               self.range_indexer(i, kpis))
                        for i in range(len(kpis[0]))]
        
        return pd.DataFrame(kpis_content)


    def cross_validation_through_time(self, dataset, start_date, end_date):

        folds = self.time_folder.get_time_splits(start_date, end_date)
        assert self.config.get('retrain_info'), 'Please set retrain info!'

        h_ranges = self.config['cross_validation']['hyperparams_ranges']
        n_trials = self.config['cross_validation']['n_trials']

        tcv_kwargs = dict(n_trials=n_trials,
                          algorithm_choice=self.algorithm_choice,
                          problem_type=self.problem_type,
                          folds=folds,
                          h_ranges=h_ranges,
                          target=self.target,
                          time_col=self.time_col,
                          ix=self.ix,
                          feature_columns=self.feature_columns)
        
        study = TimeCrossValidator(**tcv_kwargs).perform_optuna_optimization(dataset,
                                                                             self.evaluator_fn)
        return study.best_trial.params

    
    def train_with_best_params(self, dataset, train_start, train_end, test_end):
        df_train = dataset[(dataset[self.time_col] >= train_start) &
                           (dataset[self.time_col] < train_end)]
        df_test = dataset[(dataset[self.time_col] >= train_end) &
                          (dataset[self.time_col] <= test_end)]

        best = self.cross_validation_through_time(df_train, train_start, train_end)

        model = ModelObject(self.algorithm_choice,
                            self.problem_type,
                            best)
        
        trainer = ModelTrainer(time_col=self.time_col,
                               feature_columns=self.feature_columns,
                               ix=self.ix,
                               target=self.target)

        return trainer.forward_pass(model, (df_train, df_test))


    def simple_retrain(self, dataset, start_date, end_date, hyperparams={}):
        folds = self.time_folder.get_time_splits(start_date, end_date)
        n_classes = dataset[self.target].nunique() if self.problem_type != 'regression' else None
        
        model = ModelObject(self.algorithm_choice,
                            self.problem_type,
                            hyperparams)
        
        trainer = ModelTrainer(time_col=self.time_col,
                               feature_columns=self.feature_columns,
                               ix=self.ix,
                               target=self.target)
    
        kpis = trainer.retrain(dataset,
                               folds,
                               model,
                               self.evaluator_fn,
                               n_classes=n_classes)

        return self.parse_kpis_df(kpis[-2:])
