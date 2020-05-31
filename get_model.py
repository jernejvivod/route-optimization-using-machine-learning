import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from models.gboostreg import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sklearn.metrics
from joblib import dump, load
import argparse


### PARSE ARGUMENTS ###
parser = argparse.ArgumentParser(description='Create model for making predictions.')
parser.add_argument('--model', type=str, choices=['rf', 'svm', 'gboosting', 'stacking', 'majority', 'uniform'], 
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

# root mean square logarithmic error
def rmsle(y_true, y_pred):
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


# Select and initialize model.
if args.model == 'rf':
    model = RandomForestRegressor()
    model.name = 'rf'
elif args.model == 'svm':
    model = SVR(kernel='rbf', C=1e3, gamma='auto')
    model.name = 'svm'
elif args.model == 'stacking':
    model.name = 'stacking'
elif args.model == 'gboosting':
    model = GradientBoostingRegressor()
    model.name = 'gboosting'
    with open('./features_all.txt', 'r') as f:
        f_names = list(map(lambda x: x.strip(), f.readlines()))
        f_to_name = {'f' + str(idx) : f_names[idx] for idx in range(len(f_names))}
elif args.model == 'logreg':
    clf = LogisticRegression(max_iter=1000)
    clf.name = 'logreg'
    with open('./features_all.txt', 'r') as f:
        f_names = list(map(lambda x: x.strip(), f.readlines()))
        f_to_name = {'f' + str(idx) : f_names[idx] for idx in range(len(f_names))}
elif args.model == 'majority':
    model.name = 'majority'
elif args.model == 'uniform':
    model.name = 'uniform'

# Add prediction model to pipeline.
clf_pipeline.steps.append(['clf', model])


### CREATING MODEL FOR COMPUTING DISTANCES ###
if args.action == 'create':
    # If creating prediction model for general use.

    # Train prediction model.
    clf_pipeline.fit(data, target)

    # Persist fitted model.
    dump(clf_pipeline, './models/' + args.model + '.joblib')


### TRAIN-TEST SPLIT ###
if args.action == 'eval-tts':

    # Get training and testing sets.
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

    # Train model.
    clf_pipeline.fit(data_train, target_train)

    # Score predictions.
    pred = clf_pipeline.predict(data_test)
    mae = sklearn.metrics.mean_absolute_error(target_test, pred)
    evs = sklearn.metrics.explained_variance_score(target_test, pred)
    r2 = sklearn.metrics.r2_score(target_test, pred)
    
    # Print scores.
    print('mae: {0:.4f}'.format(mae))
    print('evs: {0:.4f}'.format(evs))
    print('r2: {0:.4f}'.format(r2))
    
    # If using gradient boosting method, evaluate and display feature importance scores.
    if model.name in {'gboosting', 'logreg'}:
        feature_scores = model.score_features(f_to_name)
        print('Features sorted by estimated importance:')
        for feature, score in [(feature, score) for feature, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)]:
            print('{0}: {1:.4f}'.format(feature, score))


### CROSS-VALIDATION ###
if args.action == 'eval-cv':
    raise(NotImplementedError('Evaluation using cross-validation not yet implemented'))


