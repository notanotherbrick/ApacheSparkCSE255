
# coding: utf-8

# In[ ]:

# Name: Kishan Kumar Sachdeva
# Email: ksachdev@ucsd.edu
# PID: A53104678
from pyspark import SparkContext
sc = SparkContext()


# In[1]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

from string import split,strip

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel

from pyspark.mllib.util import MLUtils


# ### Higgs data set
# * **URL:** http://archive.ics.uci.edu/ml/datasets/HIGGS#  
# * **Abstract:** This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
# 
# **Data Set Information:**  
# The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.
# 
# 

# In[2]:

#define feature names
feature_text='lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb'
features=[strip(a) for a in split(feature_text,',')]
# print len(features),features


# In[3]:

# # create a directory called higgs, download and decompress HIGGS.csv.gz into it

# from os.path import exists
# if not exists('higgs'):
#     print "creating directory higgs"
#     !mkdir higgs
# %cd higgs
# if not exists('HIGGS.csv'):
#     if not exists('HIGGS.csv.gz'):
#         print 'downloading HIGGS.csv.gz'
#         !curl -O http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
#     print 'decompressing HIGGS.csv.gz --- May take 5-10 minutes'
#     !gunzip -f HIGGS.csv.gz
# !ls -l
# %cd ..


# ### As done in previous notebook, create RDDs from raw data and build Gradient boosting and Random forests models. Consider doing 1% sampling since the dataset is too big for your local machine

# In[3]:



# In[8]:

# Import data, sample and create labeledPoints
path='/HIGGS/HIGGS.csv'
# cluster
#path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path).sample(False,0.1).cache()
Data=inputRDD.map(lambda x:x.split(','))            .map(lambda x:LabeledPoint(x[0],x[1:])).cache()



# In[6]:

cols_txt='class_label,lepton pT,lepton eta,lepton phi,missing energy magnitude,missing energy phi,jet 1 pt,jet 1 eta,jet 1 phi,jet 1 b-tag,jet 2 pt,jet 2 eta,jet 2 phi,jet 2 b-tag,jet 3 pt,jet 3 eta,jet 3 phi,jet 3 b-tag,jet 4 pt,jet 4 eta,jet 4 phi,jet 4 b-tag,m_jj,m_jjj,m_lv,m_jlv,m_bb,m_wbb,m_wwb'


# In[7]:

Columns=cols_txt.split(',')


# In[9]:

# create a train and test data split
(trainingData,testData)=Data.randomSplit([0.7,0.3])
# counts=testData.map(lambda lp:(lp.label,1)).reduceByKey(lambda x,y:x+y).collect()
# counts.sort(key=lambda x:x[1],reverse=True)
# counts


# In[10]:

trainingData=trainingData.cache()
testData=testData.cache()


# In[ ]:

print 'Gradient Boosted Trees'
from time import time
errors={}
for depth in [10]:
    start=time()
    model=GradientBoostedTrees.trainClassifier(trainingData,
                                             categoricalFeaturesInfo={}, numIterations=35, learningRate=0.15,
                                            maxDepth=depth)
    #print model.toDebugString()
    errors[depth]={}
    dataSets={'train':trainingData,'test':testData}
    #print 'training completed for ', depth
    for name in dataSets.keys():  # Calculate errors on train and test sets
        data=dataSets[name]
        Predicted=model.predict(data.map(lambda x: x.features))
        LabelsAndPredictions=data.map(lambda lp: lp.label).zip(Predicted)
        Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
        errors[depth][name]=Err
    print depth,errors[depth],int(time()-start),'seconds'
# print errors

