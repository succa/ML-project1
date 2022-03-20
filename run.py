import numpy as np

from MLscripts.proj1_helpers import *
# from MLscripts.helpers import *
from MLscripts.plot import *
from MLscripts.implementations import *

from hotgrad.sequential import Sequential
from hotgrad.variable import Variable
from hotgrad.sequential import Sequential
from hotgrad.functions.layers import Linear
from hotgrad.functions.activations import ReLU, Tanh
from hotgrad.functions.losses import MSE
from hotgrad.optimizers import SGD

print("Loading Dataset...")
yb_train, input_data_train, ids_train = load_csv_data('data/train.csv', sub_sample=False)
preprocessed_data_train, all_ys_train = clean_data(input_data_train, yb_train, feature_expansion=False)

yb_test, input_data_test, ids_test = load_csv_data('data/test.csv', sub_sample=False)
preprocessed_data_test, _ = clean_data(input_data_test, yb_test, feature_expansion=False)

X_train0_V = Variable(preprocessed_data_train[0])
X_train1_V = Variable(preprocessed_data_train[1])
X_train2_V = Variable(preprocessed_data_train[2])
X_train3_V = Variable(preprocessed_data_train[3])

X_test0_V = Variable(preprocessed_data_test[0])
X_test1_V = Variable(preprocessed_data_test[1])
X_test2_V = Variable(preprocessed_data_test[2])
X_test3_V = Variable(preprocessed_data_test[3])

y_train0_V = Variable(all_ys_train[0].reshape(-1,1))
y_train1_V = Variable(all_ys_train[1].reshape(-1,1))
y_train2_V = Variable(all_ys_train[2].reshape(-1,1))
y_train3_V = Variable(all_ys_train[3].reshape(-1,1))

print("Fitting model 0")
model0 = Sequential([Linear(100), ReLU(), Linear(50), ReLU(), Linear(1), Tanh()], MSE(), SGD(lr=0.001))
model0.fit(X_train0_V, y_train0_V, batch_size=20, epochs=100, verbose=True)
pred0 = model0.predict(X_test0_V)

print("Fitting model 1")
model1 = Sequential([Linear(100), ReLU(),  Linear(50), ReLU(), Linear(1), Tanh()], MSE(), SGD(lr=0.001))
model1.fit(X_train1_V, y_train1_V, batch_size=20, epochs=100, verbose=True)
pred1 = model1.predict(X_test1_V)

print("Fitting model 2")
model2 = Sequential([Linear(100), ReLU(), Linear(50), ReLU(), Linear(1), Tanh()], MSE(), SGD(lr=0.001))
model2.fit(X_train2_V, y_train2_V, batch_size=20, epochs=100, verbose=True)
pred2 = model2.predict(X_test2_V)

print("Fitting model 3")
model3 = Sequential([Linear(100), ReLU(), Linear(50), ReLU(), Linear(1), Tanh()], MSE(), SGD(lr=0.001))
model3.fit(X_train3_V, y_train3_V, batch_size=20, epochs=100, verbose=True)
pred3 = model3.predict(X_test3_V)

test_predictions = [pred0, pred1, pred2, pred3]

all_test_predictions = np.zeros(input_data_test.shape[0])
all_test_predictions, all_test_predictions.shape

for i, test_prediction in enumerate(test_predictions):
    all_test_predictions[input_data_test[:, 22] == i] = test_prediction

create_csv_submission(ids_test, all_test_predictions, 'submissionNN.csv')

print("Finised")