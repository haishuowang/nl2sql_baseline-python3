from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
print(mnist)
import pandas as pd
pd.to_pickle(mnist, '/home/haishuowang/PycharmProjects/nl2sql_baseline-python3/data/mnist.pkl')
