from re import X
import h2o
import pykalman as pyk
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.cluster import KMeans

sns.set()

do_grid_search = False
dependent_var = ['RemainingUsefulLife']
index_columns_names =  ["UnitNumber","Cycle"]
operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,6)]
operational_mode = ["operating mode"]
sensor_measure_columns_names =["SensorMeasure"+str(i) for i in range(1,13)]

input_file_column_names = dependent_var + index_columns_names + operational_settings_columns_names+ operational_mode + sensor_measure_columns_names
h2o.init()
train = h2o.upload_file("file_name2.csv")
test  = h2o.upload_file("file_name2.csv")
train.set_names(input_file_column_names);
test.set_names(input_file_column_names);
train[train['UnitNumber'] == 1].head(5)
train[train['UnitNumber'] == 1].tail(5)
def add_remaining_useful_life(h2o_frame):
    # Get the total number of cycles for each unit
    grouped_by_unit = h2o_frame.group_by(by=["UnitNumber"])
    max_cycle = grouped_by_unit.max(col="Cycle").get_frame()
    
    # Merge the max cycle back into the original frame
    result_frame = h2o_frame.merge(max_cycle)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_Cycle"] - result_frame["Cycle"]
    result_frame["RemainingUsefulLife"] = remaining_useful_life
    
    # drop the un-needed column
    result_frame = result_frame.drop("max_Cycle")
    return result_frame

train_with_predictor = add_remaining_useful_life(train)
train_with_predictor[train_with_predictor['UnitNumber'] == 1].head(5)
train_pd = train_with_predictor.as_data_frame(use_pandas=True)
g = sns.PairGrid(data=train_pd.query('UnitNumber < 10') ,
                 x_vars=dependent_var,
                 y_vars=sensor_measure_columns_names + operational_settings_columns_names,
                 hue="UnitNumber", height=3, aspect=2.5)

pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(train_pd)
##////////////////////////////// PCA Analysis ////////////////////////////////////////////////////////////////////
# creating dataframe 
df2 = train_pd 
df = df2.reindex(df2.index.drop(1)).reset_index(drop=True)
# checking head of dataframe 
df.head()

print(df)
from sklearn.preprocessing import StandardScaler 
  
scalar = StandardScaler() 
  
# fitting 
scalar.fit(df) 
scaled_data = scalar.transform(df) 
  
# Importing PCA 
from sklearn.decomposition import PCA 
  
# Let's say, components = 3 
pca = PCA(n_components = 3) 
pca.fit(scaled_data) 
x_pca = pca.transform(scaled_data) 

print(df)
print(x_pca)
########///////////////Clustering ////////////////////  
#print(iris)
# Create a dataframe
df.sample(4)
# Instantiate Kmeans
km = KMeans(5)
clusts = km.fit_predict(x_pca)


#Plot the clusters obtained using k means


x_pca.shape 
print(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],km.cluster_centers_[:, 2])
########///////////////Graphing  ////////////////////  
x_pca.shape 


colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(df)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2] ,c = colors, cmap='viridis', linewidth=1)
#ax.scatter(km.cluster_centers_[:, 0],km.cluster_centers_[:, 1],km.cluster_centers_[:, 2],s = 3500,marker='o',c='red',label='centroids')

ax.set_xlabel('First Principal  component')
ax.set_ylabel('Second Principal  component')
ax.set_zlabel('Third Principal  component')


plt.show()

##////////////////////////////// clustering ////////////////////////////////////////////////////////////////////



