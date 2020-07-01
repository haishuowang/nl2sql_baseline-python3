import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def kdeplot(data_test, data_train, save_path='.'):
    for x in data_test.columns:
        g = sns.kdeplot(data_train[x], color="Red", shade=True)
        g = sns.kdeplot(data_test[x], ax=g, color="Blue", shade=True)
        g.set_xlabel(x)
        g.set_ylabel("Frequency")
        g.legend(["train", "test"])
        # plt.show()
        plt.savefig(f'{save_path}/{x}.png')
        plt.close()

def scatter_matrix_plot(data_train):
    scatter_matrix(data_train, figsize=(20, 16))
    plt.show()

def hist_plot(data):
    data.iloc[:, :10].hist(bins=50, figsize=[20, 15])
    plt.show()