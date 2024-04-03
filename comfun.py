# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:38:16 2021

@author: dell
"""
import para

def read_allPOIs_to_zonePOIs(path):
    #把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573', 'zondID': '429'}
    f = open(path,encoding='utf-8') #打开all_POI
    f.readline() #读出表头信息
    
    zone_dic_wordname={}
    zone_dic_wordcordinate={}
    for i in range(0,para.zone_count): #初始化
        zone_dic_wordname[i]=[]
        zone_dic_wordcordinate[i]=[]
    while 1:
        temp = f.readline().strip('\n') #去除首尾空格

        if not temp:
            break    
        temp = temp.split(',')
        zone_dic_wordname[eval(temp[4])].append(temp[1])
        zone_dic_wordcordinate[eval(temp[4])].append([eval(temp[2]),eval(temp[3])])
    return zone_dic_wordname, zone_dic_wordcordinate   

def read_allPOIs(path):
    #把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573', 'zondID': '429'}
    f = open(path,encoding='utf-8') #打开all_POI
    f.readline() #读出表头信息
    
    dic_wordname=[]
    dic_wordcordinate=[]
    
    while 1:
        temp = f.readline().strip('\n')
        if not temp:
            break    
        temp = temp.split(',')
        dic_wordname.append(temp[1])
        dic_wordcordinate.append([eval(temp[2]),eval(temp[3])])
    print(len(dic_wordname))
    return dic_wordname, dic_wordcordinate



from scipy import spatial 
def generate_corups_place2vec(zone_dic_wordname, zone_dic_wordcordinate,path,K):
    f = open(path,"w",encoding = "utf-8")
    zoneCount=len(zone_dic_wordname)
    for i in range(0,zoneCount):# 对每一个区域    
        zone_wordname=zone_dic_wordname[i]
        poiCount=len(zone_wordname)
        if(poiCount==0):
            continue
        if(poiCount<K+2):
            for n in range(0,poiCount):
                f.write(zone_wordname[n]+' ')
            f.write('\n')
            continue
        dist_mat = spatial.distance_matrix(zone_dic_wordcordinate[i],zone_dic_wordcordinate[i])
        dist_mat=dist_mat.tolist()    
        for j in range(0,poiCount): #对区域中的每一个POI找出最近的K个邻居           
            strNN=''
            dist=dist_mat[j]
            minIndex=dist.index(min(dist))                
            dist[minIndex]=10000
            for m in range(0,K):
                minIndex=dist.index(min(dist))                
                dist[minIndex]=10000                        
                strNN+=zone_wordname[minIndex]+' '
                if( m==K/2-1):  #--------------------------------中间位置跟K有关
                    strNN+=zone_wordname[j]+' '
            f.write(strNN+'\n')        
    f.close()
    print('write over: zoneDoc_place2vec.txt')
    
def generate_corups_random(zone_dic_wordname,path=r"D:\word2vec\data\zoneDoc_randomWord.txt"):
    f = open(path,"w",encoding = "utf-8")   
    for i in range(0,para.zone_count):
        line=''
        zone=zone_dic_wordname[i]
        for poi in zone:
            line+=poi+" "
        line+="\n"
        f.write(line)
    f.close()
    print('write over: zoneDoc_randomWord.txt')

def saveVectors(A,path,label=''):
    f=open(path,"w",encoding = "utf-8")
    
    rowCount=len(A)
    colCount=len(A[0])
    
    headline=str(rowCount)+' '+str(colCount)+'\n'
    f.write(headline)
    
    
    for i in range(0,rowCount):
        line=label+str(i)+' '
        for j in range(0,colCount):
            line+=str(A[i][j])
            if j!=(colCount-1):
                line+=' '
        f.write(line+'\n')
    f.close()
    print('save successful:'+path)

def saveList(A,path,label=''):
    f=open(path,"w",encoding = "utf-8")
    rowCount=len(A)
    for i in range(0,rowCount):
        line=label+str(i)+' '+str(A[i])
        f.write(line+'\n')
    f.close()
    print('save successful:'+path)

def saveArray(A,path,label=''):
    f=open(path,"w",encoding = "utf-8")
    f.write('NO HEAD\n')
    rowCount=len(A)
    for i in range(0,rowCount):
        line=label+str(i)+' '
        colCount=len(A[i])
        j=0
        while j<colCount:
            line+=str(A[i][j])
            if j!=(colCount-1):
                line+=' '
            j=j+1
        f.write(line+'\n')
    f.close()
    print('save successful:'+path)

def readVectors(path):    
    fvec = open(path,encoding = "utf-8") #vec
    temp_vec = fvec.readline( )#.strip('\n').split(" ")
    words_num = temp_vec[0] #单词的数量
    vec_dim = eval(temp_vec[1]) #单词向量的维度

    dic_vec = {}
    #...按行读入向量
    while 1:
        line = fvec.readline().strip('\n')
        vec = [] #存向量
        if not line:
            break
        temp_vec = line.split(" ")    
        for i in range(1,len(temp_vec)):
            if(temp_vec[i]==''):
                continue
            try:
                temp_vec[i]=eval(temp_vec[i])
            except:
                temp_vec[i]=temp_vec[i]
            vec.append(temp_vec[i])
        dic_vec[temp_vec[0]] = vec


    return dic_vec

import numpy as np
def computeVectors_(ABrelationMatrix,BvecMatrix,vec_dim):
    A=[] # len(A)==len(ABrelationMatrix)==len(ABweightMatrix)
         # len(A[0])==len(BvecMatrix[0])
    
    lenA=len(ABrelationMatrix)
           
    for i in range(0,lenA): 
        ABrelation=ABrelationMatrix[i]

             
        sum_weight=len(ABrelation)
        tempVecA=np.zeros(vec_dim) #向量的维度

        for B in ABrelation:
            try:
                vecB=BvecMatrix[B] #当前词对应的词向量
            except:
                vecB=np.zeros(vec_dim) #如果wordvec中没有这个单词，附一个全0的词向量
            else:
                for j in range(0,vec_dim): #每个维度
                    tempVecA[j] += vecB[j]/sum_weight
            
        A.append(tempVecA)
       
    return A

def computeVectors(ABrelationMatrix,ABweightMatrix,BvecMatrix):
    A_vec=[] # len(A)==len(ABrelationMatrix)==len(ABweightMatrix)
         # len(A[0])==len(BvecMatrix[0])
    
    for A in ABrelationMatrix:
        B=ABrelationMatrix[A]
        
        Bweight=ABweightMatrix[A]
        sum_weight=sum(Bweight)
        tempVecA=np.zeros(para.vec_dim) #向量的维度
        k=0
        for b in B:
            try:
                vecb=BvecMatrix[b] #当前词对应的词向量
            except:
                vecb=np.zeros(para.vec_dim) #如果wordvec中没有这个单词，附一个全0的词向量
            else:
                for j in range(0,para.vec_dim): #每个维度
                    tempVecA[j] += vecb[j]*Bweight[k]/sum_weight
                    print(tempVecA[j])
            k=k+1
        A_vec.append(tempVecA)
    return A_vec
    

from gensim.models import KeyedVectors

def getSimWords_of_a_word(vecsPath,wordName):
    word_vectors=KeyedVectors.load_word2vec_format(vecsPath, binary=False)

    list1=word_vectors.most_similar(wordName,topn=12)
    print(list1)
    # return str(list1)+'\n'
    
def getSimValue_two_words(vecsPath,wordName1,wordname2):
    word_vectors=KeyedVectors.load_word2vec_format(vecsPath, binary=False)
    list1=word_vectors.similarity(wordName1,wordname2)
    print(list1)


#word2vec训练
import gensim
def train_word2vec_model(corups_path,model_path,vecdim,neighbor_num): #r"D:\word2vec\data\zoneDoc_place2vec.txt"
    with open(corups_path,'r',encoding="utf-8") as f:
        sentences = []
        for line in f:
            cols = line.strip().split(' ')
            sentences.append(cols)    
    model = gensim.models.Word2Vec(sentences, sg=1, vector_size=vecdim, alpha=0.025, window=neighbor_num+1, min_count=1, sample=1e-3, seed=1, workers=4, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1, hashfxn=hash)
    
    # save
    model.wv.save_word2vec_format(model_path+"\wordvec_place2vec.txt", binary=False) 
    model.save(model_path+"\wordvec_place2vec.model")
    
    return model
     
from sklearn.cluster import KMeans
import numpy as np
def zoneCluster(path1,path2):
    #读入zoneVec
    fzvec = open(path1,encoding = "utf-8")
    X = []
    fzvec.readline()
    while 1:
        temp = fzvec.readline() #每次读一行
        if not temp: #读完就跳出
            break
        temp = temp.split(' ')
        for i in range(1,len(temp)): #转型为数值类型
            temp[i] = eval(temp[i])
        del temp[0] #删掉zoneID
        X.append(temp)
    # 构造数据样本点集X，并计算K-means聚类
    X = np.array(X)
    
    from Bio.Cluster import kcluster
    clusterid, error, nfound = kcluster(X,nclusters=6, dist='u')


################
    # kmeans = KMeans(n_clusters=6, random_state=1).fit(X)
    
    # # 输出及聚类后的每个样本点的标签（即类别），预测新的样本点所属类别
    # kmeans_label = kmeans.labels_
    # print(kmeans_label)
    #print(kmeans.predict([[0, 0], [4, 4], [2, 1]]))
    #训练完毕...    
###################################    
    f=open(path2,"w",encoding = "utf-8")
    rowCount=len(clusterid)
    for i in range(0,rowCount):
        line=str(i)+' '+str(clusterid[i])
        
        f.write(line+'\n')
    f.close()
    print('save successful:'+path2)
    
    
    
    