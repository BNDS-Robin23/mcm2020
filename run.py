
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("E://mcm//datas//microwave.csv",encoding = 'ansi')
df=df.iloc[:,1:16]
df.dropna(axis=0, how='any', inplace=True)
df.index=np.arange(len(df))
df=df.replace('y',1)
df=df.replace('Y',1)
df=df.replace('n',-1)
df=df.replace('N',-1)
df=df.replace('?',0)

#print(df)

d=df.iloc[:,7:12]
x=preprocessing.scale(d.T)
#print(d)
#x=np.matrix(d.as_matrix(columns=None))
pca=PCA()
reduced_x=pca.fit_transform(x)
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels=['PC'+str(y) for y in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()
print(reduced_x)