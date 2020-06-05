import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('winequality-white.csv',sep=';')

count_data = wine.groupby('quality')['quality'].count()

print(count_data)

count_data.plot()
plt.show()