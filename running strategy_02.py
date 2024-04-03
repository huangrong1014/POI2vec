# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:11:23 2023

@author: Yangxin
"""

import comfun
import random
import para
import numpy as np
from scipy import spatial 
import matplotlib.pyplot as plt


def formGraph(zone):
    K = 12
    dist_mat = spatial.distance_matrix(zone,zone)
    dist_mat = dist_mat.tolist()
    poiCount = len(zone)
    adjlists = []
    for i in range(0,poiCount):
        adjlists.append([])
    for i in range(0,poiCount): #对区域中的每一个POI找出最近的K个邻居
        dist = dist_mat[i]
        for m in range(0,K):
            minIndex = dist.index(min(dist))
            dist[minIndex] = 10000
            if(minIndex!=i):
                adjlists[i].append(minIndex)
    return adjlists

def intersection(nums1, nums2):
    set1 = set(nums1)
    set2 = set(nums2)
    return list(set1 & set2)
def remove_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1 - set2)
def ab_walk(_g, poiList):
    all_path = ''
    a=para.a
    b=para.b
    for  current_word in range(0,len(_g)):
        for i in range(0,walkCount):
            a_path = poiList[ current_word]
            current_neighbors = _g[current_word]  # 起始点的邻域
            pre_word = current_word
            current_word = current_neighbors[random.randint(0, len(current_neighbors) - 1)]
            a_path = a_path + ' ' + poiList[current_word]
            for j in range(1,int(walkLength)):
                current_neighbors = _g[current_word] #当前点的邻域
                pre_neighboors = _g[pre_word]        #上一点的邻域
                intersection_words = intersection(pre_neighboors, current_neighbors)
                difference_works = remove_intersection(current_neighbors, pre_neighboors)
                m=len(intersection_words)-1   #相交
                n=len(difference_works)       #B-A
                total_probability = a+b*m+n
                x=random.uniform(0.0,total_probability) #生成随机数x
                if(x<a):
                    t = current_word
                    current_word = pre_word
                    pre_word = t
                    a_path = a_path + ' ' + poiList[current_word]
                elif (x<b*m):
                    pre_neighboors =_g[pre_word]
                    current_neighbors=_g[current_word]
                    intersection_words=intersection(pre_neighboors,current_neighbors)
                    if intersection_words!=[]:
                        t = random.choice(intersection_words)
                        pre_word = current_word
                        current_word = t
                        a_path = a_path + ' ' + poiList[current_word]
                    else:
                        t = random.choice(current_neighbors)
                        pre_word = current_word
                        current_word = t
                        a_path = a_path + ' ' + poiList[current_word]
                else:
                    pre_neighboors = _g[pre_word]
                    current_neighbors = _g[current_word]
                    difference_works=remove_intersection(current_neighbors,pre_neighboors)
                    # t = (set(current_neighbors).difference(set(pre_neighboors)))  # b中有而a中没有的
                    if difference_works!=[]:
                        t = random.choice(current_neighbors)
                        pre_word = current_word
                        current_word = t
                        a_path = a_path + ' ' + poiList[current_word]
                    else:
                        continue
            all_path = all_path + a_path +'\n'
    return all_path

def random_walk(_g, poiList):
    all_path = ''
    p=para.p
    q=para.q
    for  word in range(0,len(_g)):
        for i in range(0,walkCount):
            current_word = word
            a_path = poiList[ current_word]
            current_neighbors = _g[current_word]  # 起始点的邻域
            pre_word = current_word
            current_word = current_neighbors[random.randint(0, len(current_neighbors) - 1)]
            a_path = a_path + ' ' + poiList[current_word]
            for j in range(1,int(walkLength)):
                x=random.uniform(0.0,float(p+q)) #生成随机数x
                if(x<p):
                    pre_neighboors =_g[pre_word]
                    current_neighbors=_g[current_word]
                    intersection_words=intersection(pre_neighboors,current_neighbors)
                    if intersection_words!=[]:
                        current_word = random.choice(intersection_words)
                        a_path = a_path + ' ' + poiList[current_word]
                    else:
                        current_word = random.choice(pre_neighboors)
                        a_path = a_path + ' ' + poiList[current_word]
                else:
                    pre_neighboors = _g[pre_word]
                    current_neighbors = _g[current_word]
                    difference_works=remove_intersection(current_neighbors,pre_neighboors)
                    # t = (set(current_neighbors).difference(set(pre_neighboors)))  # b中有而a中没有的
                    if difference_works!=[]:
                        t = random.choice(current_neighbors)
                        pre_word = current_word
                        current_word = t
                        a_path = a_path + ' ' + poiList[current_word]
                    else:
                        continue
            all_path = all_path + a_path +'\n'

            #     current_word = current_neighbors[ random.randint(0,len(current_neighbors)-1)]  #标记一下上一个点，生成随机数，确定p\q\1方向，上一点的邻域和当前点的邻域，选择下一个点
            #     a_path = poiList[current_word] + ' '  + a_path
            # current_word = center_word
            # for j in range(0,int(walkLength/2)):
            #     current_neighbors = _g[current_word]
            #     current_word = current_neighbors[ random.randint(0,len(current_neighbors)-1)]
            #     a_path = a_path + ' ' + poiList[current_word]
            # all_path = all_path + a_path +'\n'
    return all_path

def draw(zone,adjlists):
    poiCount = len(zone)
    plt.scatter(zone[:,0],zone[:,1],s=2,c='k')
    for i in range(0,poiCount):
        poi = zone[i]
        neighborCount = len(adjlists[i])
        neighbors = adjlists[i]
        for j in range(0,neighborCount):
            neighbor = zone[neighbors[j]]
            x = [poi[0],neighbor[0]]
            y = [poi[1],neighbor[1]]
            plt.plot(x,y)

###################################################
all_poi_path=r'D:\word2vec\spyder\data\group\allPOIs_2411.txt'       #####
zoneCount=2411                                 #####
corupsPath=r'D:\POI2vec\POI2vec\group\node2vec20230915\node2vecwalk_corups.txt' #####
walkCount = para.num_walks            #每个节点作为起始节点游走的次数
walkLength = para.walk_length
model_path = r'D:\POI2vec\POI2vec\group\node2vec20230915'
vecDim = 200
neighbor_num = walkLength
manualfunction_path = r'D:\POI2vec\POI2vec\group\node2vec20230915\allzone_manual.txt'
###################################################

#把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573', 'zondID': '429'}
f = open(all_poi_path,'r',encoding='utf-8') #打开all_POI
f.readline() #读出表头信息
zone_dic_wordname=[]
zone_dic_wordcordinate=[]
for i in range(0,zoneCount): #初始化
    zone_dic_wordname.append([])
    zone_dic_wordcordinate.append([])
while 1:
    temp = f.readline().strip('\n')
    if not temp:
        break
    temp = temp.split(',')
    zone_dic_wordname[eval(temp[4])].append(temp[1])
    zone_dic_wordcordinate[eval(temp[4])].append([eval(temp[2]),eval(temp[3])])
f.close()

f = open(corupsPath,'w',encoding='utf-8')
for i in range(0,zoneCount): #对每个区域
    print("正在生成第"+str(i)+"个区域的语料库")
    if len(zone_dic_wordcordinate[i]) < 3:
        continue
    zone = np.array(zone_dic_wordcordinate[i])
    G = formGraph(zone) #得到zones[i]的图信息，表示为每个POI的相邻POI
    # draw(zone,G)
    corups_of_a_zone = random_walk(G, zone_dic_wordname[i])
    f.write(corups_of_a_zone)
f.close()
print("随机游走生成语料库over")

#3、训练word2vec模型
model=comfun.train_word2vec_model(corupsPath,model_path,vecDim,neighbor_num)
print("word2vec model train over")

#4、计算zone_vec
wordvecMatrix=comfun.readVectors(model_path+r"\wordvec_place2vec.txt")
zoneVectors=comfun.computeVectors_(zone_dic_wordname,  wordvecMatrix,vecDim)
comfun.saveVectors(zoneVectors, model_path+r'\zoneVec.txt','zone')

for key in  model.wv.similar_by_word('住宅小区', topn=15):  # get other similar words
         print(key)

# zoneVec_path = model_path+'\zoneVec.txt'
# import vecClassification
# vecClassification.svmClassify(model_path, zoneVec_path, manualfunction_path)
