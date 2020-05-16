import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load
import argparse


### PARSE ARGUMENTS ###
parser = argparse.ArgumentParser(description='Create model for making predictions.')
parser.add_argument('--model', type=str, choices=['rf', 'svm', 'stacking', 'majority', 'uniform'], 
        default='rf', help='Name of the model to use.')
parser.add_argument('--action', type=str, choices=['create', 'eval-tts', 'eval-cv'], 
        default='create', help='Create prediction model and save for later use or evaluate model \
                using either a train-test split or 5-fold cross-validation.')
args = parser.parse_args()
#######################

# Parse processed features and target variable values.
data = np.load('./data/features_data.npy')
target = np.load('./data/features_target.npy')

# Initialize pipeline template.
clf_pipeline = Pipeline([('scaling', RobustScaler())])

# Select and initialize model.
if args.model == 'rf':
    model = RandomForestRegressor()
    model.name = 'rf'
elif args.model == 'svm':
    model.name = 'svm'
elif args.model == 'stacking':
    model.name = 'stacking'
elif args.model == 'xgboost':
    model.name == 'xgboost'
elif args.model == 'majority':
    model.name == 'majority'
elif args.model == 'uniform':
    model.name == 'uniform'

# Add prediction model to pipeline.
clf_pipeline.steps.append(['clf', model])

if args.action == 'create':
    # If creating prediction model for general use.
    
    # Load indices of samples to exclude (correspond to grid).
    # exclude = np.load('./data/exclude_indices.npy')

    # Train prediction model.
    clf_pipeline.fit(data, target)

    # Persist fitted model.
    dump(clf_pipeline, './models/' + args.model + '.joblib')


if args.action == 'eval-tts':
    # If evaluating model using train-test split.
    pass


if args.action == 'eval-cv':
    # If evaluating model using cross-validation.
    pass




