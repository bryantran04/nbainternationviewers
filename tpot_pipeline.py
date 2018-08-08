import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-30175928.47307623
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    OneHotEncoder(minimum_fraction=0.05, sparse=False),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.9, tol=0.0001)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.65, tol=0.001)),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="ls", max_depth=5, max_features=0.25, min_samples_leaf=9, min_samples_split=9, n_estimators=100, subsample=0.8500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
