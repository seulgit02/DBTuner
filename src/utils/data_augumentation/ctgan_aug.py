from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

print("start data augumentation")
print("real_data_columns: ", real_data.columns)
print("real_data:", real_data)


# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)

print("synthetic_data: ", synthetic_data)
print("finish augumentation")