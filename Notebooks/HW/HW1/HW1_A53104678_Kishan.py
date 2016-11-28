
# coding: utf-8

# In[2]:

# Name: Kishan Kumar Sachdeva
# Email: ksachdev@ucsd.edu
# PID: A53104678

from pyspark import SparkContext
sc = SparkContext()

textRDD = sc.newAPIHadoopFile('/data/Moby-Dick.txt',
                              'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',
                              'org.apache.hadoop.io.LongWritable',
                              'org.apache.hadoop.io.Text',
                              conf={'textinputformat.record.delimiter': "\r\n\r\n"}) \
.map(lambda x: x[1])

sentences=textRDD.map(lambda x:x.replace('\r\n',' '))        .flatMap(lambda x: x.split(". "))


# In[3]:

# Function for cleaning punctuation
def cleanData(inputstr):
    s=list(inputstr) # convert string to list of characters
    for i in range(len(s)):
        o=ord(s[i])             # ASCII value of each string
        if o==46 or (o>=48 and o<=57) or (o>=97 and o<=122) : #space, numbers and lowercase left as is
            continue 
        elif (o>=65 and o<=90): # convert uppercase to lowercase
            s[i]=chr(o+32)  
        else:               # replace with space for anything else 
            s[i]=" " 
        
    return str("".join(s)) # str to convert unicode string to string


# In[4]:

sentences_rm_punct=sentences.map(cleanData)#sentences_rm_rn.map(cleanData) # clean data of punctuation, uppercase


# In[5]:

sentence2words=sentences_rm_punct.map(lambda x: x.split(" "))    .filter(lambda x: x != ['']) # split by space and remove blank sentences


# In[6]:

def removeBlankWords(inputlist):
    y=[]
    for x in inputlist:
        if x != '':
            y.append(x)
    return y
sentence2words_clean=sentence2words.map(removeBlankWords) # remove blank words of form ''


# In[7]:

def printOutput(n,freq_ngramRDD):
    top=freq_ngramRDD.take(5)
    print '\n============ %d most frequent %d-grams'%(5,n)
    print '\nindex\tcount\tngram'
    for i in range(5):
        print '%d.\t%d: \t"%s"'%(i+1,top[i][0],' '.join(top[i][1]))


# In[8]:

for n in range(1,6):
    if n==1:
        pairs=sentence2words_clean.flatMap(lambda x:x)
        freq_ngramRDD=pairs.map(lambda word: (word, 1))             .reduceByKey(lambda a, b: a + b)             .sortBy(lambda x:x[1],False)             .map(lambda x: (x[1],[x[0]]))
    else:
        if n==2:
            pairs=sentence2words_clean.flatMap(lambda x:zip(x,x[1:])) # create bigram
        elif n==3:
            pairs=sentence2words_clean.flatMap(lambda x:zip(x,x[1:],x[2:])) # create trigram
        elif n==4:
            pairs=sentence2words_clean.flatMap(lambda x:zip(x,x[1:],x[2:],x[3:])) #create 4-grams
        elif n==5:
            pairs=sentence2words_clean.flatMap(lambda x:zip(x,x[1:],x[2:],x[3:],x[4:])) #create 5-grams

        freq_ngramRDD=pairs.map(lambda word: (word, 1))                  .reduceByKey(lambda a, b: a + b)                 .sortBy(lambda x:x[1],False)                 .map(lambda x: (x[1],x[0]))
        
    printOutput(n,freq_ngramRDD)

