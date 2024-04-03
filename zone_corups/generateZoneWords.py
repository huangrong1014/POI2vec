# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:43:01 2023

@author: Yangxin
"""
import random

current_path=r'D:\POI2vec\POI2vec\zone_corups'
zone_topiclist=current_path+'\zone_topiclist.txt'
clear_topiclist=current_path+'\clear_topiclist.txt'
generatedZone=current_path+'\zone_random_corups.txt'

f = open(clear_topiclist,encoding='utf-8') 
topiclist={}
for i in range(0,9):
    topiclist[i]=[]
while 1:
    temp = f.readline().strip('\n')
    if not temp:
        break    
    temp = temp.split(',')
    
    a_topic={}
    for i in range(0,10):
        a_topic[temp[i+1]]=temp[i+11]
    topiclist[eval(temp[0])].append(a_topic)
f.close()
print('读取topic列表成功。。。')

f = open(zone_topiclist,encoding='utf-8') 
f.readline()
all_zones=''
while 1:
    temp = f.readline().strip('\n')
    temp=temp.strip(',')
    if not temp:
        break    
    temp = temp.split(',')
    POICount=eval(temp[1])
    zone_all_poi=''
    for k in range(0,POICount):
        
        topic_cls=[]
        for i in range(2,len(temp)):
            topic_cls.append(temp[i]) #每个功能区内包含的主题类型


        x=random.random()
        if x<0.3:
            rand_topiccls=random.randint(0,8) #选第几类主题呢？
            selectable_topics=topiclist[rand_topiccls]
            rand_topic=random.randint(0, len(selectable_topics)-1)#选这类主题中的第几个主题呢？
            the_topic=selectable_topics[rand_topic]
            rand_poi=random.uniform(0,1)
            prob=0
            # print(str(k)+':')
            for key,value in the_topic.items():
                prob=prob+eval(value)
                # print(str(rand_poi)+','+str(prob))
                if(rand_poi<=prob):
                    print(k)
                    zone_all_poi=zone_all_poi+' '+key
                    break
        else:
            rand_topiccls = random.randint(0, len(topic_cls) - 1)  # 选第几类主题呢？
            selectable_topics = topiclist[eval(topic_cls[rand_topiccls])]
            rand_topic = random.randint(0, len(selectable_topics) - 1)  # 选这类主题中的第几个主题呢？
            the_topic = selectable_topics[rand_topic]
            rand_poi = random.uniform(0, 1)
            prob = 0
            # print(str(k)+':')
            for key, value in the_topic.items():
                prob = prob + eval(value)
                # print(str(rand_poi)+','+str(prob))
                if (rand_poi <= prob):
                    print(k)
                    zone_all_poi = zone_all_poi + ' ' + key
                    break
    zone_all_poi=zone_all_poi.strip(' ')
    all_zones=all_zones+zone_all_poi+'\n'
    
f.close()
f=open(generatedZone,"w",encoding = "utf-8")
f.write(all_zones)
f.close()
        
        
        