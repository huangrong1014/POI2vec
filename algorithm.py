# -*- coding: utf-8 -*-
'''
Implementation conditions of node2vec.
'''

import para
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import random
import gensim
from np import *
from gensim.models import Word2Vec
from math import radians, cos, sin, asin, sqrt

def read_allPOIsR(path):
    # 把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573'}
    f = open(path, encoding='utf-8')  # 打开POI文件
    f.readline()  # 读出表头信息

    poiname = []
    poicoordinate = []
    # zone_dic_wordcordinate = []
    id_zone=[]
    zone_dic_id= {}

    for i in range(0,para.zone_count): #初始化
        zone_dic_id[i]=[]
        # zone_dic_wordcordinate.append([])

    while 1:
        temp = f.readline().strip('\n')  # 去除首尾空格
        if not temp:
            break

        temp = temp.split(',')
        poiname.append(temp[1])
        poicoordinate.append([eval(temp[2]), eval(temp[3])])
        # zone_dic_wordcordinate[eval(temp[4])].append([eval(temp[2]), eval(temp[3])])
        id_zone.append(temp[4])
        zone_dic_id[eval(temp[4])].append(temp[0])    #zone-id

    f.close()
    return poiname,poicoordinate,id_zone,zone_dic_id
def read_allPOIsK(path):
    # 把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573'}
    f = open(path, encoding='utf-8')  # 打开POI文件
    f.readline()  # 读出表头信息

    poiname = []
    # poicoordinate = []
    zone_dic_wordcordinate = []
    # id_zone=[]
    zone_dic_id= {}

    for i in range(0,para.zone_count): #初始化
        zone_dic_id[i]=[]
        zone_dic_wordcordinate.append([])

    while 1:
        temp = f.readline().strip('\n')  # 去除首尾空格
        if not temp:
            break

        temp = temp.split(',')
        poiname.append(temp[1])
        # poicoordinate.append([eval(temp[2]), eval(temp[3])])
        zone_dic_wordcordinate[eval(temp[4])].append([eval(temp[2]), eval(temp[3])])
        # id_zone.append(temp[4])
        zone_dic_id[eval(temp[4])].append(temp[0])    #zone-id

    f.close()
    return poiname,zone_dic_wordcordinate,zone_dic_id


import csv
def generate_corups_node2vec( poicoordinate, id_zone, zone_dic_id,path):  # 获取每个poi点的k阶邻域内的邻居
    f = open(path, "w", encoding="utf-8")
    poicount = len(poicoordinate)
    print(poicount)

    for i in range(0,poicount):   #对每一个POI找出最近的1阶邻域内所有poi点
        mdist = 1
        ldist = 150000000

        i_zone=int(id_zone[i])     #点i的zoneID
        zone_len=len(zone_dic_id[i_zone])  #点i的zoneID的长度
        i_list=zone_dic_id[i_zone]    #i_list为zoneID为i的所有poi点
        k=0

        j=0
        while j<zone_len:   #遍历每个poi点
            nodes = []
            lon0,lat0=poicoordinate[i]
            t=int(i_list[j])
            lon1,lat1=poicoordinate[t]
            # 将十进制度数转化为弧度
            lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])

            # haversine公式
            dlon = lon1 - lon0
            dlat = lat1 - lat0
            a = sin(dlat / 2) ** 2 + cos(lat0) * cos(lat1) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            r = 6371  # 地球平均半径，单位为公里
            dist=c * r * 1000
            # 如第一个点的经纬度为（lon0, lat0），第二个点的经纬度为（lon1, lat1）
            if(0<dist<ldist): #最小距离ldist
                ldist=dist
            if(dist>=mdist):
               mdist=dist #最大距离mdist
            if (k*(para.R)< dist <= (k+1)*(para.R)):
                # temp = poiname[t]
                # temp1=poicoordinate[t]
                # poiname1[k].append(temp)
                # poicoordinate1[k].append(temp1)
                nodes.append(i)
                nodes.append(t)
                nodes.append(dist)
                writer = csv.writer(f)
                writer.writerow(nodes)
            j=j+1

    f.close()
    return

import csv
def generate_corups_neartpoi( poicoordinate,zone_dic_wordcordinate,id_zone, zone_dic_id,path):  # 获取每个poi点的k阶邻域内的邻居
    f = open(path, "w", encoding="utf-8")
    poicount = len(poicoordinate)      #poi点的总数量
    print(poicount)
    K = para.K
    for i in range(0,poicount):   #对每一个POI找出最近的1阶邻域内所有poi点
        i_zone=int(id_zone[i])     #点i的zoneID
        # zone_len=len(zone_dic_id[i_zone])  #点i的zoneID的长度

        if len(zone_dic_wordcordinate[i_zone]) < 3:
            continue
        zone = np.array(zone_dic_wordcordinate[i_zone])
        dist_mat = spatial.distance_matrix(zone, zone)
        dist_mat = dist_mat.tolist()
        poiCount = len(zone)          #zone的长度

        t=0
        list = zone_dic_id[i_zone]        #i_list为zoneID为i的所有poi点
        for m in range(0,poiCount):
            if i == list[m]:
                t=m
                break
        dist = dist_mat[t]
        for n in range(0, K):
            nodes = []
            x = min(dist)  # x表示点之间的距离
            minIndex = dist.index(min(dist))
            dist[minIndex] = 10000
            # adjlists[i].append(minIndex)
            if x > 0:
                node2 = minIndex
                nodes.append(list[m])
                nodes.append(list[node2])
                nodes.append(x)
                writer = csv.writer(f)
                writer.writerow(nodes)

    f.close()
    return


import csv
import math
def generate_corups_neartpoi2( zone_dic_wordcordinate, zone_dic_id,path):  # 获取每个poi点的k阶邻域内的邻居
    f = open(path, "w", encoding="utf-8")
    zone_count=len(zone_dic_id)        #功能区数目
    print(zone_count)
    K = para.K

    for i in range(0,zone_count):   #遍历每个功能区
        if len(zone_dic_wordcordinate[i]) < 3:
            continue
        zone = np.array(zone_dic_wordcordinate[i])
        dist_mat = spatial.distance_matrix(zone, zone)
        dist_mat = dist_mat.tolist()

        poiCount = len(zone_dic_wordcordinate[i])          #zone的长度
        list = zone_dic_id[i]         # list为zoneID为i的所有poi点的id
        for m in range(0,poiCount):     #遍历zone中每个POI
            dist = dist_mat[m]
            for n in range(0, K):
                nodes = []
                x = min(dist)
                x = x * 1000    # x表示点之间的距离
                minIndex = dist.index(min(dist))
                dist[minIndex] = 10000
                if x == 0:
                    continue
                elif x == 10000:
                    break
                else :
                    nodes.append(list[m])
                    nodes.append(list[minIndex])
                    nodes.append(x)
                    f.write(nodes[0]+','+nodes[1]+','+str(nodes[2]))
                    f.write('\n')

    f.close()
    return






def distance(a,b,poicoordiante):

    lon0,lat0=poicoordiante[a]
    lon1,lat1=poicoordiante[b]
    # 将十进制度数转化为弧度
    lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])

    # haversine公式
    dlon = lon1 - lon0
    dlat = lat1 - lat0
    a = sin(dlat / 2) ** 2 + cos(lat0) * cos(lat1) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    dist=c * r * 1000
    # 如第一个点的经纬度为（lon0, lat0），第二个点的经纬度为（lon1, lat1）
    return(dist)


# def generate_distance_node2vec(poiname, poicoordinate): #获取每个点到其余点的权值
#     poicount = len(poiname)
#
#     each_dist=[]
#     all_dist=[]
#     for i in range(0, poicount):  # 对每一个POI找出最近的n阶邻域内所有poi点
#         distance1=[]
#         all_dist1=0
#         for j in range(0, poicount):  # 遍历每个poi点
#             lon0, lat0 = poicoordinate[i]
#             lon1, lat1 = poicoordinate[j]
#             # 将十进制度数转化为弧度
#             lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])
#
#             # haversine公式
#             dlon = lon1 - lon0
#             dlat = lat1 - lat0
#             a = sin(dlat / 2) ** 2 + cos(lat0) * cos(lat1) * sin(dlon / 2) ** 2
#             c = 2 * asin(sqrt(a))
#             r = 6371  # 地球平均半径，单位为公里
#             dist = c * r * 1000
#             # 如第一个点的经纬度为（lon0, lat0），第二个点的经纬度为（lon1, lat1）
#
#             distance1.append(dist)
#             all_dist1+=dist
#
#         str0 = str(distance1)
#         each_dist.append(eval(str0))
#         all_dist.append(all_dist1)
#
#     return ( each_dist,all_dist)
#
#
# def generate_neighbor_node2vec(poicoordinate, poicoordinate2): #获取每个poi点的邻居权值
#     poicount = len(poicoordinate)
#     poi_1_dist= []
#     for i in range (0,poicount): #对象为每一个poi点
#         list1=poicoordinate2[i]
#         list2=list1[0]
#         distance0=[]
#         nodes=[]
#         for j in range(0, len(list2)):  # 遍历每个poi点的一阶范围内poi点
#             lon0, lat0 = poicoordinate[i]
#             lon1, lat1 = list2[j]
#             # 将十进制度数转化为弧度
#             lon0, lat0, lon1, lat1 = map(radians, [lon0, lat0, lon1, lat1])
#
#             # haversine公式
#             dlon = lon1 - lon0
#             dlat = lat1 - lat0
#             a = sin(dlat / 2) ** 2 + cos(lat0) * cos(lat1) * sin(dlon / 2) ** 2
#             c = 2 * asin(sqrt(a))
#             r = 6371  # 地球平均半径，单位为公里
#             dist = c * r * 1000
#             # 如第一个点的经纬度为（lon0, lat0），第二个点的经纬度为（lon1, lat1）
#
#             distance0.append(dist)
#
#         str0 = str(distance0)
#         poi_1_dist.append(eval(str0))
#     return ( poi_1_dist)


def preprocess_transition(nbr_dist,nodes,poicoordiante):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    length = len(nodes)

    alias_nodes = {}

    for node in range(0,length):
        unnormalized_probs =nbr_dist[node]  # 获取边的weight 读取每个邻点权重 node与其邻居节点之间的权重
        norm_const = sum(unnormalized_probs)  # 权重求和，作为公式中正则项常数的那个分母
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]  # 正则化权重weight 除以分母 ,计算从node到邻居的转移矩阵
        alias_nodes[node] = alias_setup(normalized_probs)


    alias_edges=[]
    p = para.p
    q = para.q
    r = para.R

    for m in range(0,len(nodes)):
        src=m                     #src为startnode
        src_nbr=nodes[m]          #m的邻居节点
        alias_edges1 = {}
        for n in src_nbr:
            cur,cur_src=n         #cur为当前值,cur_src为src到cur的距离
            cur_nbr = nodes[cur]  # cur_nbr为cur点的邻居节点
            alias_edges1[cur]=[]
            unnormalized_probs1 = []
            for l in cur_nbr:
                nbr,nbr_cur=l #nbr为cur的当前邻居节点，nbr_cur为nbr_cur的距离
                if(src==nbr):
                    unnormalized_probs1.append(1 / p)  # 加权后的转移概率
                elif(0<distance(m,nbr,poicoordiante)<r):
                    unnormalized_probs1.append(1)  # 加权后的转移概率
                else:
                    unnormalized_probs1.append(1/q)  # 加权后的转移概率
            norm_const1 = sum(unnormalized_probs1)
            normalized_probs1 = [float(u_prob1) / norm_const1 for u_prob1 in unnormalized_probs1]
            alias_edges1[cur].append(alias_setup(normalized_probs1))

        str_edge=str(alias_edges1)
        alias_edges.append(eval(str_edge))

    return(alias_nodes,alias_edges)




def alias_setup(probs):
    '''
	Compute utility lists for non-uniform sampling from discrete distributions.
	'''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def simulate_walks(poiname,poi_id, num_walks, walk_length,alias_nodes,alias_edges,nbr_id,path):  # num_walks每个节点作为开始节点的次数
    # walk_length每次游走生成的节点序列的长度
    '''
    Repeatedly simulate random walks from each node.
    '''
    f = open(path, "w", encoding="utf-8")
    walks = []
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        random.shuffle(poi_id)
        for node in poi_id:
            walk=node2vec_walk(alias_nodes, alias_edges, nbr_id, walk_length=walk_length, start_node=node)
            for z in walk:
                f.write(poiname[int(z)]+' ')
            f.write('\n')
    f.close()
    # node2vec_walk为一次随机游走，用于生成一次随机游走的序列
    return walks



def node2vec_walk(alias_nodes,alias_edges,nbr_id,walk_length, start_node):
    '''
    Simulate a random walk starting from start node.
    '''

    walk=[start_node]
    while len(walk) < walk_length:  # 序列长度为walk_length
        cur = walk[-1]
        cur_nbrs=nbr_id[int(cur)]
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(cur_nbrs[alias_draw(alias_nodes[int(cur)])])
                '''
                        alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]函数是以论文中转移概率公式P来选择下一个节点，返回值是下一个节点的index。这部分用到的函数alias_draw以及它调用的alias_setup函数是一种概率采样方法
                '''
            else:
                prev = walk[-2]
                a=alias_edges[int(prev)]
                edges=a[cur]
                walk.append(cur_nbrs[alias_draw(edges[0])])
        else:
            break

    return walk

def alias_draw(t):
    '''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
    J,q=t
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]



def train_word2vec_model(corups_path, model_path):  # r"D:\POI2vec\group\word2vec_place2vec.txt"
    with open(corups_path, 'r', encoding="utf-8") as f:
        sentences = []
        for line in f:
            cols = line.strip().split(' ')
            sentences.append(cols)
    model = gensim.models.Word2Vec(sentences, sg=1, vector_size=para.dimensions, alpha=0.025, window=para.window_size,
                                   min_count=1, sample=1e-3, seed=1, workers=8, min_alpha=0.0001, hs=0, negative=20,
                                   cbow_mean=1, hashfxn=hash)

    # save
    model.wv.save_word2vec_format(model_path + "\word2vec_place2vec_q_0.01.txt", binary=False)
    model.save(model_path + "\word2vec_place2vec_q_0.01.model")

    return model


# def learn_embeddings(walks):
#     '''
#     Learn embeddings by optimizing the Skipgram objective using SGD.
#     '''
#     walks = [list(map(str, walk)) for walk in walks]
#     model = Word2Vec(walks, vector_size=para.dimensions, window=para.window_size, min_count=0, sg=1,workers=8,epochs=1)
#     model.wv.save_word2vec_format(para.output)
#
#     return


def read_allPOIs_to_zonePOIs(path):
    # 把所有POI读入pois,形如{'FID': '0', 'TYPE2': '公司', 'X': '116.50991', 'Y': '40.041573', 'zondID': '429'}
    f = open(path, encoding='utf-8')  # 打开all_POI
    f.readline()  # 读出表头信息

    zone_dic_wordname = {}
    for i in range(0, para.zone_count):  # 初始化
        zone_dic_wordname[i] = []
    while 1:
        temp = f.readline().strip('\n')  # 去除首尾空格
        if not temp:
            break
        temp = temp.split(',')
        zone_dic_wordname[eval(temp[4])].append(temp[1])
    return zone_dic_wordname

def readzonePOIs(path):
    f = open(path, encoding='utf-8')  # 打开all_POI

    zone_dic_wordname = {}
    for i in range(0, 590):  # 初始化
        zone_dic_wordname[i] = []
    j=0
    while 1:
        temp = f.readline().strip('\n')  # 去除首尾空格
        if not temp:
            break
        temp = temp.split(' ')
        for m in range(0,len(temp)):
            zone_dic_wordname[j].append(str(temp[m]))
            if m==len(temp)-1:
                j=j+1
    return zone_dic_wordname

def readVectors(path):
    fvec = open(path,encoding = "utf-8") #vec
    temp_vec = fvec.readline()#.strip('\n').split(" ")
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



def computeVectors_(ABrelationMatrix, BvecMatrix):
    A = []  # len(A)==len(ABrelationMatrix)==len(ABweightMatrix)
    # len(A[0])==len(BvecMatrix[0])

    lenA = len(ABrelationMatrix)

    for i in range(0, lenA):
        ABrelation = ABrelationMatrix[i]

        sum_weight = len(ABrelation)
        tempVecA = np.zeros(para.vec_dim)  # 向量的维度

        for B in ABrelation:
            try:
                vecB = BvecMatrix[B]  # 当前词对应的词向量
            except:
                vecB = np.zeros(para.vec_dim)  # 如果wordvec中没有这个单词，附一个全0的词向量
            else:
                for j in range(0, para.vec_dim):  # 每个维度
                    tempVecA[j] += vecB[j] / sum_weight

        A.append(tempVecA)

    return A


def saveVectors(A, path, label=''):
    f = open(path, "w", encoding="utf-8")

    rowCount = len(A)
    colCount = len(A[0])

    headline = str(rowCount) + ' ' + str(colCount) + '\n'
    f.write(headline)

    for i in range(0, rowCount):
        line = label + str(i) + ' '
        for j in range(0, colCount):
            line += str(A[i][j])
            if j != (colCount - 1):
                line += ' '
        f.write(line + '\n')
    f.close()
    print('save successful:' + path)


def saveList(A, path, label=''):
    f = open(path, "w", encoding="utf-8")
    rowCount = len(A)
    for i in range(0, rowCount):
        line = label + str(i) + ' ' + str(A[i])
        f.write(line + '\n')
    f.close()
    print('save successful:' + path)




def readVectors2(path):
    fvec = open(path,encoding = "utf-8") #vec
    temp_vec = fvec.readline().strip('\n').split(" ")
    words_num = temp_vec[0] #单词的数量
    vec_dim = eval(temp_vec[1]) #单词向量的维度

    dic_vec = {}
    #...按行读入向量
    vec = []  # 存向量
    while 1:
        line = fvec.readline().strip('\n')

        if not line:
            break
        temp_vec = line.split(" ")
        vec.append(temp_vec[0])

    return words_num,vec

from gensim.models import KeyedVectors


def getSimWords_of_a_word(vecsPath, wordName):
    word_vectors = KeyedVectors.load_word2vec_format(vecsPath, binary=False)

    list1 = word_vectors.most_similar(wordName, topn=20)
    # print(list1)
    # print('\n')
    return str(list1)+'\n'


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
