# -*- coding: utf-8 -*-
# Name: Kishan Kumar Sachdeva
# Email: ksachdev@ucsd.edu
# PID: A53104678
from pyspark import SparkContext
sc = SparkContext()

def print_count(rdd):
    print 'Number of elements:', rdd.count()


from pyspark.sql import Row
import numpy as np

with open('../Data/hw2-files-1gb.txt') as f:
    files = [l.strip() for l in f.readlines()]
data = sc.textFile(','.join(files))\
          .map(lambda text: text.encode('utf-8')).cache()        
print_count(data)