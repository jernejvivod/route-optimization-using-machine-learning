from gboostreg import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sklearn.metrics

boston = load_boston()
data = boston.data
target = boston.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

model = GradientBoostingRegressor()
ref_model = RandomForestRegressor()
model.fit(data_train, target_train)
ref_model.fit(data_train, target_train)

pred = model.predict(data_test)
ref_pred = ref_model.predict(data_test)

mae = sklearn.metrics.mean_absolute_error(target_test, pred)
ref_mae = sklearn.metrics.mean_absolute_error(target_test, ref_pred)

