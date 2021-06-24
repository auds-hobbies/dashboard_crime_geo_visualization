#!/usr/bin/env python
# coding: utf-8

# #  SFO CRIME VISUALIZATION (DASHBOARD IN DJANGO)
# 
# 
# 
# <div class="alert alert-danger" style="margin: 10px"><strong>NOTE!</strong> This is a fictional project.</div>
# 
# <div class="alert alert-block alert-info">
#   
#     
# <b>Summary:</b> 
# * <span style='font-family:Georgia'> Visualization of crime within SFO via dashboard built using Django, HTML, and CSS
# * <span style='font-family:Georgia'> This is the first stage of the dashboard and will be deployed to PythonAnywhere.
# * <span style='font-family:Georgia'> Concentration of locations with high crime determined clustering using DBSCAN
#  
# 
# <b>Contents:</b> 
# * <span style='font-family:Georgia'> Data preprocessing
# * <span style='font-family:Georgia'> Exploratory data analysis
# * <span style='font-family:Georgia'> Feature engineering
# * <span style='font-family:Georgia'> Geo - visualization 
#  

# In[1]:


import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import LineString, Point, shape, Polygon, MultiPoint
import shapely.geometry as geom
from descartes import PolygonPatch
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium import Map
from folium.map import Layer, FeatureGroup,LayerControl,Marker
from folium.plugins import MarkerCluster,FastMarkerCluster,FeatureGroupSubGroup,Fullscreen 
#from libpysal.weights.distance import get_points_array
from matplotlib import cm
import mplleaflet as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import numpy
from scipy.spatial import cKDTree
import mapclassify

import pandas as pd
import numpy as np
from numpy import mean, std, NaN, array
from stop_words import get_stop_words
import scipy.stats as stats
import random 
#from random import seed
#seed(1)
from collections import Counter

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer#, SnowballStemmer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk import pos_tag, word_tokenize, RegexpParser 
#from gensim import models, corpora
import string
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

from pandas.plotting import scatter_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from plotly import graph_objects as go 
from plotly.graph_objs import * 
from dash.dependencies import Input, Output, State
import plotly.express as px 

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline #as imbPipeline

from joblib import dump
from joblib import load
import pickle 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score,KFold, GridSearchCV
from sklearn.decomposition import NMF, LatentDirichletAllocation
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, PowerTransformer, RobustScaler,StandardScaler
from sklearn.feature_selection import SelectFromModel
#pip install libpysal
#pip install pygeos

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.linear_model import LogisticRegression,LinearRegression, Lasso, ElasticNet
from xgboost import XGBClassifier, XGBRegressor
from sklearn.dummy import DummyClassifier 
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier , GradientBoostingClassifier,RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor, ExtraTreesRegressor

from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report, confusion_matrix, auc, accuracy_score,log_loss,roc_auc_score,                             roc_curve, make_scorer,r2_score,jaccard_score, mean_squared_error
from imblearn.metrics import geometric_mean_score 

import json


pd.options.display.max_columns = 35
pd.options.display.float_format ='{:.2f}'.format 

import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter('ignore')


# <div class="alert alert-block alert-info">
#   
#     
# ### <b>Load Data, Pre-processing:</b> 
# 
# - load data 
# - check features for missing values, duplication etc
# - merge duplicate crime rows on crime description, resolution etc
# - clean text columns

# In[2]:


sfo_file = r"C:...incidents.csv"  

sf = pd.read_csv(sfo_file, parse_dates=True)

print('Shape = ',sf.shape)
print(f"Missing ={[col for col in sf.columns if sf[col].isna().sum()>0]} ")

sf.info(3)


# In[3]:


# Missing values 

sf['PdDistrict'] = sf['PdDistrict'].fillna('no_name_pd') 
print('Features with missing values = ', [col for col in sf.columns if sf[col].isna().sum()>0])


# In[4]:


# checking for duplicates in ID 

print(f"ID ={sf.IncidntNum.dtype}\n")
sf.IncidntNum.value_counts(dropna=False).head(10)


# Duplication in incident numbers -- all features, except category and descript differ in values. We propose the theory that
# the duplication is as a result of multiple crimes committed in relation to the specific incident number.

# In[5]:


# checking duplication on a specific incident number
sf.loc[sf.IncidntNum == 160886390][:3]


# In[6]:


# concatenating duplicate rows 
sf = sf.groupby('IncidntNum').agg({'Category':', '.join, 'Descript':', '.join,'Date':'first', 'Time':'first',
               'PdDistrict':'first','Resolution':', '.join,'Address':', '.join, 'X':'first','Y':'first',
                'Location':'first','PdId':'first' }).reset_index()

#sf.head(2)
sf.loc[sf.IncidntNum == 160886390][:3]


# #### Text & Date Columns

# In[7]:


stop_words = set(stopwords.words('english'))


corpus_category = []
for i in range(len(sf['Category'])):
    #create a list of common stopwords
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","from"]

    stop_words = stop_words.union(new_words)
    # remove special characters 
    text = re.sub("(\\W)+", " ", sf['Category'][i])
    text = text.lower()
    text = text.split()
    text = set(text)
    text2 = [t for t in text if not t in set(stop_words)]
    text2 = " ".join(text)
    corpus_category.append(text2) 
sf['category'] = corpus_category


corpus_resolution = []
for i in range(len(sf['Resolution'])):
    #create a list of common stopwords
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","from"]

    stop_words = stop_words.union(new_words)
    # remove special characters 
    text = re.sub("(\\W)+", " ", sf['Resolution'][i])
    text = text.lower()
    text = text.split()
    text = set(text)
    text2 = [t for t in text if not t in set(stop_words)]
    text2 = " ".join(text)
    corpus_resolution.append(text2)
sf['resolution'] = corpus_resolution

corpus = []
for i in range(len(sf['Descript'])):
    #create a list of common stopwords
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","from"]

    stop_words = stop_words.union(new_words)
    # remove special characters 
    text = re.sub("(\\W)+", " ", sf['Descript'][i])
    text = text.lower()
    text = text.split()
    text = set(text)
    text2 = [t for t in text if not t in set(stop_words)]
    text2 = " ".join(text)
    corpus.append(text2)
sf['descript'] = corpus

#fetch wordcount for each abstract
sf['word_count'] = sf['descript'].apply(lambda x: len(str(x).split(" ")))
#sf[['descript', 'word_count']].head(3)

sf['inc_c'] = 1
sf.inc_c = sf.inc_c.astype(int)
sf['date'] = pd.to_datetime(sf['Date'])#, errors='coerce')
#sf.dtypes

sf.set_index('date',inplace=True)
#sf.head(1)

sf['year'] = sf.index.year
sf['month'] = sf.index.month # df['Month']=df.index.strftime('%B')
sf['day'] = sf.index.day
sf.head(1)


# <div class="alert alert-block alert-info">
#   
#     
# ### <b>Exploratory Data Analysis:</b> 
# 
# * <span style='font-family:Georgia'> Text data
# * <span style='font-family:Georgia'> Date / time  

# In[8]:


#Months with the least value of incidents after June 1, 2016
sf2 = sf[['inc_c']]
sf2.loc['2016-06-01':].idxmin()


# In[9]:


# monthly sum
sf2[['inc_c']].resample('M').sum().plot.bar(figsize=(15,3));


# In[10]:


#visualizing top bi-grams
def get_top_n_words(corpus,n=None):
    vec = CountVectorizer(ngram_range=(5,5), # 5 words
                         max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, 
                      idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                       reverse=True)
    return words_freq[:n]

#convert most frequent words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=10) # 10 sentences/bar graphs
top_df = pd.DataFrame(top_words)
top_df.columns = ['n_gram', 'freq'] 
top_df[:10]

fig3 = px.bar(top_df, x="n_gram", y="freq")#, color="smoker", barmode="group")
fig3.show()


# <div class="alert alert-block alert-info">
#   
#     
# ### <b>Feature Engineering, Spatial Feature Engineering:</b> 
# 
# * <span style='font-family:Georgia'> New features e.g., crime labels, spatial features 
# * <span style='font-family:Georgia'> Spatial feature engineering e.g., mergine dataframe with shapefile

# In[11]:


# FEATURE ENGINEERING 
h0 = sf.copy()


h0corp = []
for i in range(len(h0['Category'])):
    #create a list of common stopwords
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","from"]

    stop_words = stop_words.union(new_words)
    # remove special characters 
    text = re.sub("(\\W)+", " ", h0['Category'][i])
    text = text.lower()
    text = text.split()
    #text = set(text)
    text2 = [t for t in text if not t in set(stop_words)]
    text2 = " ".join(text)
    h0corp.append(text2)
h0['category0'] = h0corp
#h0.head(1)


# In[12]:


#patterns
#- summary of items for string matching
p1_theft ='(larceny theft)|(vehicle theft)|(burglary)|(robbery)|(stolen property)' #theft
p2_assault_vandalism = '(assault)|(vandalism)|(arson)|(disorderly conduct)|(suspicious)|(trespass)' #assault_vandalism
p3_missing_suicide = '(missing person)|(kidnapping)|(loitering)|(family offenses)|(suicide)|(runaway)' # missing_suicide 
p4_fraud = '(fraud)|(forgery)|(counterfeiting)|(embezzlement)|(bribery)|(extortion)' # fraud
p5_narcotics_drunk_sex = '(sex offenses)|(prostitution)|(pornography)|(obscene)|(drug)|(narcotic)|                           (drunkenness)|(driving under the influence)' # narcotics drunk sex diu
    
p6_gambling_warrants_other ='(gambling)|(liquor laws)|(bad checks)|(weapon laws)|(law)|(warrants)|(secondary codes)|                              (other offences)|(other)' # gambling_warrants_other 
    
p7_noncriminal = '(recovered vehicle)|(non criminal)' # noncriminal vehicle non criminal
#p8_unknown                      


h0['category1'] = h0['category0'].replace({
                                        #p0_noreason:'noreason',
                                        p1_theft:'- theft - ',
                                        p2_assault_vandalism:'- assault_vandalism - ',
                                        p3_missing_suicide:'- missing_suicide - ',
                                        p4_fraud:'- fraud - ',
                                        p5_narcotics_drunk_sex:'- narcotics_diu_sex - ',
                                        p6_gambling_warrants_other:'- gambling_warrants_other -',
                                        p7_noncriminal:'- noncriminal - '
                                        },regex = True)

h1 = h0.reset_index()

#splitting along '-' 
cat1split = []
for i in range(0,len(h1)):
    text = h1['category1'][i]
    if '-' in text:
        text2 = text.split('-')
        text2 = filter(None,text2)
       # text2 = set(text2)
    else:
        text2 = text
    text2 = ''.join(text2)
    cat1split.append(text2)     

h1['category2'] = cat1split 

# extract labels or give "unknown" label 
reasons_known = ['theft','assault_vandalism','missing_suicide',
                    'fraud','narcotics_diu_sex','gambling_warrants_other','noncriminal']
elements = []
for i in range(0,len(h1)):
    text = h1['category2'][i]
    check4knownlabels = any([x in reasons_known for x in text.split()])
    if check4knownlabels:
        text2 = text
    else:
        text2 = re.sub(text, 'unknown', text)
    elements.append(text2)

h1['category3'] = elements

elements_cat = []
crime_categories = set(reasons_known)
for i in range(0,len(h1)):
    text = h1['category2'][i]
    text2 = text.split() #set(text.split())
    text3 = list( crime_categories.intersection(text2) )
    text3_len = len(text3)
    if text3_len == 0:
        text2 = 'unknown'
        text3 = text2
    else:
        text3int = text3
        textrem = []
        for j in text2:
            if j in text3int:
                textrem.append(j)
            else:
                pass
        text3 = textrem
        text3 = ' '.join(text3)
    elements_cat.append(text3)
h1['cat3v1'] = elements_cat

h1[['Category','category0','cat3v1']][:10]#.value_counts()


# In[ ]:


# 
""" 
crime categories = ['theft','assault_vandalism','missing_suicide',
                    'fraud','narcotics_diu_sex','gambling_warrants_other','noncriminal']
"""
category_label_patterns = "(theft)|(assault_vandalism)|(missing_suicide)|(fraud)|(narcotics_diu_sex)|(gambling_warrants_other)|(noncriminal)|(unknown)"

# 
h1['labels'] = h1[['cat3v1']].apply(lambda r: " ,".join(set(r.str.extractall(category_label_patterns).stack())), axis=1)

#extracting labels
h1['theft'] = h1.labels.str.extract('(theft)', expand=True)
h1['assault_vandalism'] = h1.labels.str.extract('(assault_vandalism)', expand=True)
h1['missing_suicide'] = h1.labels.str.extract('(missing_suicide)', expand=True)
h1['fraud'] = h1.labels.str.extract('(fraud)', expand=True)
h1['narcotics_diu_sex'] = h1.labels.str.extract('(narcotics_diu_sex)', expand=True)
h1['gambling_warrants_other'] = h1.labels.str.extract('(gambling_warrants_other)', expand=True)
h1['noncriminal'] = h1.labels.str.extract('(noncriminal)', expand=True)
h1['unknown'] = h1.labels.str.extract('(unknown)', expand=True)


# In[ ]:


str_cols = ['theft','assault_vandalism','missing_suicide',
                    'fraud','narcotics_diu_sex','gambling_warrants_other','noncriminal','unknown']
str_cols = h1.columns[h1.dtypes==object]
h1[str_cols] = h1[str_cols].fillna(0)
h1.fillna(0,inplace=True)


# In[ ]:


h1['theft'] = h1['theft'].copy()
h1['assault_vandalism'] = h1['assault_vandalism'].copy()
h1['missing_suicide'] = h1['missing_suicide'].copy()
h1['fraud'] = h1['fraud'].copy()
h1['narcotics_diu_sex'] = h1['narcotics_diu_sex'].copy()
h1['gambling_warrants_other'] = h1['gambling_warrants_other'].copy()
h1['noncriminal'] = h1['noncriminal'].copy()
h1['unknown'] = h1['unknown'].copy()

h1.theft.replace(('theft','0'),(1,0), inplace = True)
h1.assault_vandalism.replace(('assault_vandalism','0'),(1,0), inplace = True)
h1.missing_suicide.replace(('missing_suicide','0'),(1,0), inplace = True)
h1.fraud.replace(('fraud','0'),(1,0), inplace = True)
h1.narcotics_diu_sex.replace(('narcotics_diu_sex','0'),(1,0), inplace = True)
h1.gambling_warrants_other.replace(('gambling_warrants_other','0'),(1,0), inplace = True)
h1.noncriminal.replace(('noncriminal','0'),(1,0), inplace = True)
h1.unknown.replace(('unknown','0'),(1,0), inplace = True)

h1[[#'cat3v1',
    'labels','theft','assault_vandalism','missing_suicide','fraud', 
   'narcotics_diu_sex','gambling_warrants_other','noncriminal','unknown']][-5:]


# In[ ]:


cols_to_keep = ['date',  'year', 'month', 'day','IncidntNum', 'Time', 'PdDistrict','X', 'Y', 'Location', 'PdId', 
                'category', 'resolution', 'descript', 'inc_c','cat3v1', 'labels', 'theft', 'assault_vandalism', 
                'missing_suicide','fraud', 'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal','unknown' ]

cols_to_delete = ['Category', 'Descript', 'Date', 'Resolution', 'Address', 
                 'word_count', 'category0', 'category1', 'category2', 'category3']

h1 = h1[cols_to_keep] #h1.drop(columns=cols_to_delete)

print('Shape ={}'.format(h1.shape))


# ### Spatial Features

# In[ ]:


# geopandas 
# shape file
sfopd = "...\san-francisco.geojson"

dfp = gpd.read_file(sfopd)
print(f"police district = {dfp.shape}")
#dfp


c000v0 = h1.copy() #sf.copy()
c000 = c000v0.reset_index()
c000.to_csv("C:...\sfopd_001.csv")

c000 = pd.read_csv("C:..._001.csv")

# converting to geodataframe
gc000 = gpd.GeoDataFrame(
    c000, 
    geometry=gpd.points_from_xy(c000.X, c000.Y)
)


# convert geodataframe to shape file 
gc000.to_file("C:....shp"
,crs={'init' :'epsg:4326'})
#

missing_items = [col for col in c000.columns if c000[col].isna().sum()>0]
print(f"missing = {missing_items}, \npd df shape = {c000.shape}")
print(f"gc000 shape = {gc000.shape}")

gc000.head(1)


# In[ ]:


# loading shape file 

shp_file = "C:..."

gc000shp = gpd.read_file(shp_file)

print(f"gc000shp = {gc000shp.shape}")


gc000shp_nb = gpd.sjoin(gc000shp,  dfp[['DISTRICT','COMPANY','geometry']], how="inner", op='intersects')
print(f'police district = {dfp.shape}, gc000shp_nb {gc000shp_nb.shape}')


#merging incidents with location
gc000shp_dfp = dfp2.merge(gc000shp, on='PdDistrict', how='left')


gc000shp_nb = gc000shp_nb.rename(columns={ 'assault_va':'assault_vandalism', 'missing_su':'missing_suicide', 
            'narcotics_':'narcotics_diu_sex', 'gambling_w':'gambling_warrants_other', 'noncrimina':'noncriminal'})

print(f"gc000shp_nb = {gc000shp_nb.shape}")


# # total crime counts per sub-region 
# dfp2 = dfp.copy()
# dfp2['PdDistrict'] = dfp2['DISTRICT'].copy()
# 
# district_crime_counts0 = gc000shp_dfp[['DISTRICT','inc_c']].groupby(by='DISTRICT').count()
# dfp2 = dfp2.merge(district_crime_counts0, on='DISTRICT')
# 
# dfp2 = dfp2[['OBJECTID', 'DISTRICT', 'COMPANY', 'geometry', 'PdDistrict', 'inc_c_x']]
# dfp2.columns = ['OBJECTID', 'DISTRICT', 'COMPANY', 'geometry', 'PdDistrict', 'inc_c']
# print(dfp2.shape)
# dfp2

# In[ ]:


# 
tracts = dfp.copy()

# summing up the problem types (labels) by NAME_2 .....NOTE:-name2 seems expanded list of municipalities
sum_labels_by_inc = gc000shp_nb.groupby('DISTRICT')[['inc_c']].sum().reset_index()


# percentage of incidents
sum_labels_by_inc['incident_perc'] = (sum_labels_by_inc['inc_c']/sum_labels_by_inc['inc_c'].sum() )*100
#sum_labels_by_inc


#merging with the North Brabant shape file 
tracts = tracts.merge(sum_labels_by_inc, on='DISTRICT', how='left')

print(tracts.shape)
tracts


# In[ ]:


# summing up the problem types (labels) by NAME_2 .....NOTE:-name2 seems expanded list of municipalities
sum_labels_by_name2 = gc000shp_nb.groupby('DISTRICT')[[ 'theft', 'assault_vandalism', 'missing_suicide', 'fraud', 
            'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal', 'unknown']].sum().reset_index()

sum_labels_by_name2[:5]


# In[ ]:


# count the total number of problems in each cbs region
sum_labels_by_name2['sum_total_labels'] = sum_labels_by_name2[['theft', 'assault_vandalism', 'missing_suicide', 'fraud', 
            'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal', 'unknown']].sum(axis=1)

sum_labels_by_name2['regional_perc'] = round(sum_labels_by_name2['sum_total_labels']/sum_labels_by_name2['sum_total_labels'].sum()*100, 2) 
#sum_labels_by_name2[:5]


#merging with the North Brabant shape file 
tracts = tracts.merge(sum_labels_by_name2, on='DISTRICT', how='left')

print(tracts.shape)

tracts.head(2)


# <div class="alert alert-block alert-info">
#   
#     
# ### <b>Geo - Visualization:</b> 
# 
# * <span style='font-family:Georgia'> Visualization of spatial features 
# 

# In[ ]:


import matplotlib

fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'aspect':'equal'})
dfp2.plot(column='inc_c', scheme='Quantiles', k=5, cmap='GnBu', legend=True, ax=ax)
#ax.set_xlim(150000, 160000)
#ax.set_ylim(208000, 215000)
ax.set_title('Crime per SFO area')


# In[ ]:


# GEO-CODING SFO LOCATION 

address = "San Francisco, United States of America"
geolocator = Nominatim(user_agent="my_app")
mycoordinates = geolocator.geocode(address)
#print("Latitude = {}, Longitude = {}".format(mycoordinates.latitude, mycoordinates.longitude))
#mycoordinates.raw
latitude = mycoordinates.latitude
longitude = mycoordinates.longitude
print("latitude: {}, longitude: {}".format(latitude, longitude))


# In[ ]:


#RUN DBSCAN 

selectRegion = 'BAYVIEW'
trafficmap_region = gc000shp_nb[gc000shp_nb['DISTRICT']==selectRegion]

# traffic map
g4 = trafficmap_region.copy() 

coordsg6 = g4[['IncidntNum','PdDistrict','category','resolution','labels','Y','X']]
coordsg6array = g4[['IncidntNum','PdDistrict','category','resolution','labels','Y','X']].values

coordsg6_lonlat = coordsg6[['Y','X']].values 

max_distance = 0.075  
## minimum number of cluster members
min_coordinate_pts = 2
## calculate epsilon parameter using the user defined distance
kms_per_radian = 6371.0088
epsilon = max_distance / kms_per_radian  
## perform clustering
db_coordsg6 = DBSCAN(eps=epsilon, min_samples=min_coordinate_pts,
                algorithm='ball_tree', metric='haversine').fit(np.radians(coordsg6_lonlat))
#generate cluster labels
cluster_labels = db_coordsg6.labels_
coordsg6['cluster_labels'] = cluster_labels # or db_coordsg6.labels_

coordsg6_clusters = coordsg6[coordsg6['cluster_labels'] != -1].reset_index()
coordsg6_nonclusters = coordsg6[coordsg6['cluster_labels'] == -1].reset_index()

# cluster labels
cluster_labels_realclusters = np.array(coordsg6_clusters.cluster_labels.values)
cluster_labels_nonclusters = np.array(coordsg6_nonclusters.cluster_labels.values)
print(f'Clusters: {len(cluster_labels_realclusters)}\nNon clusters: {len(cluster_labels_nonclusters)}\nDistance (radius): {max_distance}km ')


# In[ ]:


# number of clusters for non -1_one type clusters 
num_real_clusters = len(set(coordsg6_clusters.cluster_labels.values))
# number of clusters for -1_one type clusters 
num_non_clusters = len(set(coordsg6_nonclusters.cluster_labels.values))

# clustr labels
cluster_labels = set(coordsg6_clusters.cluster_labels.values.tolist())

#clusters relating to coordinates 
coordsg6_latlon = coordsg6_clusters[['Y','X']].values   
clusters_latlon = pd.Series([coordsg6_latlon[cluster_labels_realclusters == n] for n in range(num_real_clusters)])
center_lat = []
center_lon = []
center_geom_lst = []
cluster_size_lst = []

for i in range(len(clusters_latlon)):
    ## filter empty clusters
    #if clusters_latlon[ii].any():
    ## get centroid and magnitude of cluster
    center_lon.append(MultiPoint(clusters_latlon[i]).centroid.x) #latitude: 37.7790262 -y, longitude: -122.419906-x
    center_lat.append(MultiPoint(clusters_latlon[i]).centroid.y)
    center_geom = (MultiPoint(clusters_latlon[i]).centroid.x, MultiPoint(clusters_latlon[i]).centroid.y)
    center_geom_lst.append(center_geom)
    cluster_size_lst.append(len(clusters_latlon[i]))   
    
    

cluster_df_coords = pd.DataFrame(zip(center_geom_lst,center_lat,center_lon,#lst_category,lst_resolut,\
                                     cluster_size_lst, cluster_labels),
            columns = ['center_geom','center_lat','center_lon','cluster_size',#'lst_resolut','cluster_size', \
                       'cluster_labels'])
cdc = cluster_df_coords.copy()

####
coordsg6_clusters_array_noncoords = coordsg6_clusters[['category','resolution','labels']].values

clusters_array_all_grp = pd.Series([coordsg6_clusters_array_noncoords[cluster_labels_realclusters == n] for n in range(num_real_clusters)])

from itertools import zip_longest

lst_i_concat = []
lst_i_concat_category = []
lst_i_concat_resoluti = []
lst_i_labels = []
for i in range(len(clusters_array_all_grp)):
    i= clusters_array_all_grp[i]
    
    #i_concat = [' '.join(   set(x.split())  ) for x in map(', '.join, zip_longest(*i,fillvalue=''))] # removing set
    i_concat = [' '.join(   x.split()  ) for x in map(', '.join, zip_longest(*i,fillvalue=''))] # enabling full no
    i_category = i_concat[0]
    i_resoluti = i_concat[1]
    i_labels = i_concat[2]
    lst_i_concat.append(i_concat)
    lst_i_concat_category.append(i_category)
    lst_i_concat_resoluti.append(i_resoluti)
    lst_i_labels.append(i_labels)

# clustr labels
cluster_labels = set(coordsg6_clusters.cluster_labels.values.tolist())
clusters_all_grp_df = pd.DataFrame(zip(lst_i_concat,lst_i_concat_category,lst_i_concat_resoluti,lst_i_labels), 
                                   columns=['concat','comb_category','comb_resolution','comb_labels'])
cag = clusters_all_grp_df.copy()
#cag[:5]


# In[ ]:


res_refinded = []

#r = result.copy()
for rr in range(len(cag)):
    rr = cag['comb_resolution'].values[rr]
    s = rr.split(',')
    t =  list(filter(None,s))
    u = ''.join(t)
    v = u.split(' ')
    w = list(set(v))
    x = ' '.join(w)
    res_refinded.append(x)
    
cag['res_refined'] = res_refinded
    
#cag['resolution_refined'] = res_refinded
#cag.resolution_refined[0]
problem_labels_cleaned = []
for i in range(len(cag)):
    a = cag['comb_labels'].values.tolist()[i]
    b = ''.join(a)
    c = b.split(',')
    d = ''.join(c)
    e = d.split(' ')
    f = list(filter(None,e))
    f2 = ' '.join(f)
    problem_labels_cleaned.append(f2)
    #print(f)
cag['cluster_problem_labels_cleaned'] = problem_labels_cleaned

problem_labels_summary = []
for c in range(len(cag)):
    c = cag['cluster_problem_labels_cleaned'].values.tolist()[c]
    d = c.split(',')
    d = list(filter(None,d))
    problem_labels_summary.append(d)
    #print(d)
cag['problem_labels_summary'] = problem_labels_summary

#FOR THE DIFFERENT KINDS OF PROBLEMS AT LOCATION, NO COUNT-------
problem_labels_summaryset = []
for c in range(len(cag)):
    c = cag['cluster_problem_labels_cleaned'].values.tolist()[c]
    d = c.split(',')
    e = set(d)
    f = list(filter(None,e))
    problem_labels_summaryset.append(f)
    #print(d)
cag['problem_labels_summaryset'] = problem_labels_summaryset
#------

problem_labels_summary_dict = []
for s in range(len(cag)):
    s = cag['problem_labels_summary'].values[s]
    t = ' '.join(s)
    u = t.split(' ')
    u = list(filter(None,u))
    v = Counter(u)
    #dict((x,l.count(x)) for x in set(l))
    w = dict((x,u.count(x)) for x in u)  #dict((x,u.count(x)) for x in set(u))
    #print(v) #Counter({'other_reason': 1, 'limited_visibility': 1, 'fast_cars': 1})
    #print(w)
    problem_labels_summary_dict.append(w)
cag['problem_labels_summary_dict'] = problem_labels_summary_dict

lst_numprobs = [] # number of problems per cluster
for i in range(len(cag)):
    i2 = cag['problem_labels_summary_dict'].values[i]
    numprobs = sum(list(i2.values()))
    lst_numprobs.append(numprobs)
cag['total_number_problems'] = lst_numprobs
    

result = pd.concat([cdc,cag],axis=1)
result = result.drop(columns=['concat'])
print(result.shape)

#result.comb_resolution[0]
final_columns = ['center_geom', 'center_lat', 'center_lon', 'cluster_size',
       'cluster_labels', 'comb_category', 'comb_labels', 'res_refined','problem_labels_summary_dict',
                'total_number_problems','cluster_problem_labels_cleaned' ,'problem_labels_summaryset'#
                ]
result = result[final_columns]
result[['center_geom', 'cluster_size',
       'cluster_labels', 'problem_labels_summary_dict','total_number_problems','problem_labels_summaryset']][:2]


# In[ ]:


traffic_map = folium.Map(location=[latitude, longitude], zoom_start=10)


def color(cluster_size): 
    if cluster_labels != -1:
        if cluster_size in range(0,10): 
            col = 'cyan'
        elif cluster_size in range(10,50): 
            col = 'yellow'
        else: # cluster_size in range(6,100e6): 
            col = 'red'
    else: 
        col = 'grey' # 
    return col 

for center_lat, center_lon, cluster_labels,cluster_size,labelsum,labelsumset,totalno in zip(result['center_lat'], 
    result['center_lon'], result['cluster_labels'], result['cluster_size'],result['problem_labels_summary_dict']
                                             ,result['problem_labels_summaryset'],result['total_number_problems']
                                                                             ):
    
    folium.CircleMarker(
        [center_lon, center_lat],
        #radius=.15*bike, # number of labels at that point? 
        popup = ('Cluster ID #: ' + str(cluster_labels).capitalize() + '<br>'
                 #'Crime types: ' + str(labelsumset) + '<br>'
                 'No. of Incidents:' + str(cluster_size) + '<br>'
                  'Crime Type:' + str(labelsum) + '<br>'
                'Total # Crime Types: ' + str(totalno) #+'%' # 'cluster_size: ' + str(cbs) +'%'
                ),
        color=color(cluster_size), #color='b',
        key_on = cluster_labels,#cluster_size,
        threshold_scale=[0,1],#,2,3],
        #fill_color=colordict[cluster_size],
        fill=True,
        fill_opacity=0.7
        ).add_to(traffic_map)
    

traffic_map


# In[ ]:


cluster_variables =  [ 'theft', 'assault_vandalism', 'missing_suicide', 'fraud',
       'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal', 'unknown']



f, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# loop over all the variables of interest
for i, col in enumerate(cluster_variables):
    # axis where the map will go
    ax = axs[i]
    tracts.plot(column=col, ax=ax, scheme='Quantiles', 
            linewidth=0, cmap='RdPu')
    # Remove axis clutter
    ax.set_axis_off()
    # axis title to the name of variable being plotted
    ax.set_title(col)
# Display the figure
#plt.title('Spread of crime types per district')
plt.show()


# In[ ]:


tracts[['DISTRICT','theft', 'assault_vandalism', 'missing_suicide', 'fraud',
      # 'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal',
       #'unknown'
                   ]].plot(x="DISTRICT", kind="bar", figsize=(16,4))

plt.xlabel('District')
plt.ylabel('Number of problems')
plt.title('Analysis per police district')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'aspect':'equal'})
tracts.plot(column='theft', scheme='Quantiles', k=5, cmap='GnBu', legend=True, ax=ax)
#ax.set_xlim(150000, 160000)
#ax.set_ylim(208000, 215000)
#ax.set_xlabel('rate per 100')
ax.set_title('Theft across SFO police district', fontsize=16)


# In[ ]:


zz = tracts[['theft', 'assault_vandalism', 'missing_suicide', 'fraud',
       'narcotics_diu_sex', 'gambling_warrants_other', 'noncriminal', 'unknown']]


categories = list(zz.columns.values)
sns.set(font_scale = 2)
plt.figure(figsize=(15,6))
ax= sns.barplot(categories, zz.iloc[:,:].sum().values)
plt.title("Crime labels per category", fontsize=18)
plt.ylabel('Number of labels', fontsize=12)
plt.xlabel('Label type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = zz.iloc[:,:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
    
plt.xticks(rotation=69)
#plt.yticks(rotation=90)
# plt.savefig("sample.jpg")

plt.show()


# <div class="alert alert-block alert-info">
#   
#     
# # <b>Thank You | Dank u wel !!</b> 
