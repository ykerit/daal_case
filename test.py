import daal4py as d4p
import numpy as np

data = np.array([[1] * 27])

file = open('bst_wf_test.txt', 'rb')
model = file.read()
train_result = d4p.gbt_regression_training_result()
train_result.__setstate__(model)

result = d4p.gbt_regression_prediction().compute(data, train_result.model)
