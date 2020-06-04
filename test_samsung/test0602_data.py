import numpy as np
import pandas as pd

samsung = pd.read_csv('./data/csv/삼성전자 주가.csv',index_col=0, header=0, encoding='CP949')
hite = pd.read_csv('./data/csv/하이트 주가.csv',index_col=0, header=0, encoding='CP949')

print(samsung.head())
print(hite.head())

samsung = samsung.dropna(axis=0)

hite = hite.fillna(method='bfill') # back_fill
hite = hite.dropna(axis=0)

hite = hite[:509]
hite.iloc[0,1:] = ['10','20','30','40']
# hite.loc['2020-06-02','고가':'거래량'] = [10,20,30,40]
print(hite)

# 오름차순으로 변경 <- 날짜문자열을 어떻게 인식한거지....?!
samsung = samsung.sort_values(['일자'], ascending='True')
print(samsung)

for i in range(len(samsung)) :
    samsung.iloc[i,0] = int(samsung.iloc[i,0].replace(',',''))

for i in range(len(hite)) :
    for j in range(len(hite.iloc[i])) :
        hite.iloc[i,j] = int((hite.iloc[i,j]).replace(',',''))

print(samsung.shape)
print(hite.shape)

samsung = samsung.values
hite = hite.values

np.save('./data/samsung.npy', arr=samsung)
np.save('./data/hite.npy', arr=hite)


