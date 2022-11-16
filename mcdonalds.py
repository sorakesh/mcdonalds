#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np


# In[3]:


dt= pd.read_csv('https://homepage.boku.ac.at/leisch/MSA/datasets/mcdonalds.csv')
dt.sample(10)


# In[4]:


dt.info()


# In[5]:


dt.describe()


# In[6]:


dt['Age'].value_counts()


# In[7]:


dt['Gender'].value_counts()


# In[8]:


dt['Like'].value_counts()


# In[9]:


dt['VisitFrequency'].value_counts()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns 


# In[11]:


labels = ['Female', 'Male']
size = dt['Gender'].value_counts()
explode = [0,0.2]
colors = ['Pink', 'blue'] ## color Genders

plt.rcParams['figure.figsize'] = (7,7)
plt.pie(size, labels = labels, explode = explode, colors = colors, autopct = '%.3f%%', startangle = 270,shadow=True)
plt.legend(title="Gender",fontsize= 15)
plt.show() ## Female % is greater than male 


# In[12]:


#Age
plt.rcParams['figure.figsize'] = (25, 8)
f = sns.countplot(x=dt['Age'],palette = 'Set1')
f.bar_label(f.containers[0])
plt.title('Age distribution of customers', fontsize= 30)
plt.show()
# Mcdonalds recieve more customers of age between 50-60 and 35-40 less customers above 68 and below 21
            


# In[13]:


#Customer segmentation - based on pyschographic segmentation

#For convinence renaming the category
dt['Like']= dt['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
#Like 

sns.boxenplot(data=dt, x="Like", y="Age",scale="linear",saturation=0.75, width=0.8, k_depth="trustworthy")            
plt.title('Likelyness of McDonald w.r.t Age', fontsize=20)
plt.show()


# In[14]:




sns.boxplot(
    data=dt, x="VisitFrequency", y="Age",
    notch=True, showcaps=True,
    flierprops={"marker": "x"},
    boxprops={"facecolor": (.8, .12, .1, .21)},
    medianprops={"color": "coral"},
    
)
plt.title('VisitFrequency of McDonald w.r.t Age', fontsize=20)
plt.show()


# In[23]:


from sklearn.preprocessing import LabelEncoder
def replace1(x): ## yes replace to 1 other wise zero
    dt[x] = LabelEncoder().fit_transform(dt[x])
    return dt

ctg = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in ctg:
    replace1(i)
dt


# In[24]:


#Histogram of the each attributes
plt.rcParams['figure.figsize'] = (12,14)
dt.hist()
plt.show()


# In[25]:


#Considering only first 11 attributes
dt_ctg = dt.loc[:,ctg]
dt_ctg


# In[26]:


#Considering only the 11 cols and converting it into array
ar = dt.loc[:,ctg].values
ar


# In[27]:


#Principal component analysis

from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_dt = preprocessing.scale(ar)

pca = PCA(n_components=11)
pc = pca.fit_transform(ar)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
df = pd.DataFrame(data = pc, columns = names)
df


# In[28]:


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


# In[29]:


np.cumsum(pca.explained_variance_ratio_)


# In[30]:


# correlation coefficient between original variables and the component

ld = pca.components_ #//ld loding 
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
ld_dt = pd.DataFrame.from_dict(dict(zip(pc_list, ld)))
ld_dt['variable'] = dt_ctg.columns.values
ld_dt = ld_dt.set_index('variable')
ld_dt


# In[31]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,20)
ax = sns.heatmap(ld_dt, annot=True)
plt.show()


# In[32]:


get_ipython().system('pip install bioinfokit')


# In[33]:


#Scree plot (Elbow test)- PCA
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(12,5))


# In[34]:


# get PC scores
pca_scores = PCA().fit_transform(ar)

# get 2D biplot
cluster.biplot(cscore=pca_scores, loadings=ld, labels=dt.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(12,7))


# EXTRACTING SEGMENTS

# In[35]:


pip install yellowbrick


# In[36]:


#Extracting segments

#Using k-means clustering analysis
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)).fit(dt_ctg)
visualizer.show()


# In[40]:


#K-means clustering 

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(dt_ctg)
dt['cluster_num'] = kmeans.labels_ #adding to dt
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster. 


# In[38]:


#To see each cluster size
from collections import Counter
Counter(kmeans.labels_)


# In[39]:


#Visulazing clusters
sns.scatterplot(data=df, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# DESCRIBING SEGMENTS

# In[41]:



#DESCRIBING SEGMENTS

from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
crosstab =pd.crosstab(dt['cluster_num'],dt['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[42]:


#Mosaic plot gender vs segment
crosstab_gender =pd.crosstab(dt['cluster_num'],dt['Gender']) # 0= Femle, 1= male
crosstab_gender


# In[43]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[44]:


#box plot for age

sns.boxplot(x="cluster_num", y="Age",data=dt)


# Selecting target segment

# In[45]:


#Calculating the mean
#Visit frequency
dt['VisitFrequency'] = LabelEncoder().fit_transform(dt['VisitFrequency'])
visit = dt.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[46]:


#Like
dt['Like'] = LabelEncoder().fit_transform(dt['Like'])
Like = dt.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[47]:


#Gender
dt['Gender'] = LabelEncoder().fit_transform(dt['Gender'])
Gender = dt.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[48]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[49]:


#Target segments

plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[50]:


import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics


# In[51]:


#create demogram and find the best clustering value
dt_model = dt.drop(['Gender'],axis=1)
merg = shc.linkage(dt_model,method="ward")
plt.figure(figsize=(25,10))
shc.dendrogram(merg,leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# In[52]:


dt_model = dt.drop(['Gender'],axis=1)
plt.figure(figsize=(25, 15))  
plt.title(" Dendograms")  
dend = shc.dendrogram(shc.linkage(dt_model, method='ward')) 


# In[53]:


hiyerartical_cluster = AgglomerativeClustering(n_clusters = 4,affinity= "euclidean",linkage = "ward")
hiyerartical_cluster.fit_predict(dt_model)


# In[54]:


#create model
kmeans = KMeans(n_clusters=4)
data_predict = kmeans.fit_predict(dt_model)

plt.figure(figsize=(15,10))
plt.scatter( x = 'Age' ,y = 'Like' , data = dt_model , c = data_predict , s = 200 )
plt.xlabel("Age")
plt.ylabel("Like")
plt.show()


# In[ ]:





# In[ ]:




