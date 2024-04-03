# -*- coding: utf-8 -*-
"""
Document notes
"""
vec_dim=200
zone_count=2411

node2vec_path=r'D:\POI2vec\group'
poi_path=r'D:c\POI2vec\group\allPOIs_2411.txt'
R_edgelist=r'D:\POI2vec\group\R_edgelist.csv'
K_edgelist=r'D:\POI2vec\group\K_edgelist.txt'
words_walk=r'D:\POI2vec\group\words_walk.txt'
poi_graph=r'D:\POI2vec\group\allPOIs_2411_graph.txt'
# #zonevec聚类
# zone_sum=2411
# zone_path=r'D:\POI2vec\group\allPOI_2411.txt'

K=12               #K-main
R=25              #R-main

num_walks=20      #walk_counts 每个节点作为开始节点的次数
walk_length=12     # walk_length每次游走生成的节点序列的长度

p=5            #通过调节p、q改变广度、深度搜索
q=1




a=100
b=100


dimensions=200

window_size=100



# poi_id包括所有poi的id poiname包含所有的poi名称  poicoordinate包括所有的poi坐标
# nodes包含所有poi点到邻居节点的ID及权值 nbr_dist\nbr_id为邻居权值和节点
# alias_nodes为startnode到每个邻居的概率矩阵  alias_edges是初始点为多个点时到达邻居节点的概率矩阵