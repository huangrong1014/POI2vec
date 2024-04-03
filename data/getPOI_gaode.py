# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:27:35 2021

@author: dell
"""
import urllib3
import json
import re

# acquire top-left-coordinate, bottom-right-coordinate of city
def get_city_scope(key, cityname):
    parameters = 'key={}&keywords={}&subdistrict={}&output=JSON&extensions=all'.format(key, cityname, 0)
    url = 'https://restapi.amap.com/v3/config/district?' + parameters
    http = urllib3.PoolManager()
    response = http.request('GET',url)
    jsonData = json.loads(response.data.decode('utf8'))
    if jsonData['status'] == '1':
        district = jsonData['districts'][0]['polyline']
        district_list = re.split(';|\|',district)
        xlist, ylist = [], []
        for d in district_list:
            xlist.append(float(d.split(',')[0]))
            ylist.append(float(d.split(',')[1]))
        xmax = max(xlist)
        xmin = min(xlist)
        ymax = max(ylist)
        ymin = min(ylist)
        print(xmax)
        print(xmin)
        print(ymax)
        print(ymin)
        return [xmin, xmax, ymin, ymax]
    else:
        print ('fail to acquire: {}'.format(jsonData['info']))
        return None


# get_city_scope(key, '110105')
# 116.645273   116.462,39.947561  116.468888,39.945073
# 116.350882   116.461904,39.951098  116.468695,39.948511
# 40.107933
# 39.809768
    
import numpy as np
import time
import pandas
# import geopandas
# search POI based on polygon (key, cityname, 宽度分成多少份, 高度分成多少份, POI types(list), output path)
def get_Amap_POI(key, cityname, width, height, filename):
    print ('getting scope...')
    #scope = get_city_scope(key, cityname)
    scope=[116.350882,116.645273,39.809768,40.107933]
    # 生成 width * height 个网格，每个网格左上角和右下角的坐标
    xlist = np.arange(scope[0], scope[1], (scope[1]-scope[0])/width)
    ylist = np.arange(scope[2], scope[3], (scope[3]-scope[2])/height)
    xlist = xlist.tolist()
    ylist = ylist.tolist()
    xlist.append(scope[1])
    ylist.append(scope[3])
    polygons = []
    for y in range(len(ylist)-1):
        for x in range(len(xlist)-1):
            polygons.append(str(xlist[x]) + ',' + str(ylist[y+1]) + '|' + str(xlist[x+1]) + ',' + str(ylist[y]))
    del xlist, ylist, scope
    
    # poi_dict = {'id':[],'name':[],'type':[],'address':[],'lon':[],'lat':[]}
    print ('getting data...')
    break_flag = False     # prevent failing to request API, break loop and save data
    
    f = open(filename,"w",encoding = "utf-8")
    f.write('id,name,type,lon,lat\n')
    t='汽车服务|汽车销售|汽车维修|摩托车服务|餐饮服务|购物服务|生活服务|体育休闲服务|医疗保健服务|住宿服务|风景名胜|商务住宅|政府机构及社会团体|科教文化服务|交通设施服务|金融保险服务|公司企业|道路附属设施|地名地址信息|公共设施|事件活动'
    
    for polygon in polygons:
        
        count, page_index = 1,1
        while count != 0:
            parameters = "key={}&polygon={}&types={}&output=JSON&page_size=25&page={}".format(key,polygon,t,page_index)
            url = "https://restapi.amap.com/v3/place/polygon?" + parameters
            http = urllib3.PoolManager()
            try: response = http.request('GET', url, timeout = 10.0)
            except:
                print ('request timeout...saving data...')
                print (' polygon {}'.format(polygons.index(polygon)))
                break_flag = True
                break
                
            jsonData = json.loads(response.data.decode('utf8'))
            if jsonData['status'] == '1':
                print ("getting data from page {} in polygon {}...".format(page_index, polygons.index(polygon)))
                count = int(jsonData['count'])
                page_index += 1
                for poi in jsonData['pois']:                    
                    
                    lon, lat = gcj2wgs(float(poi['location'].split(',')[0]),float(poi['location'].split(',')[1]))
                    
                    f.write(str(poi['id'])+','+str(poi['name'])+','+str(poi['type'])+','+str(lon)+','+str(lat)+'\n')
                    
                   # time.sleep(0.1)
            else: 
                print ("fail to request api in page {} in polygon {}: {}".format(page_index, polygons.index(polygon), jsonData['info']))
                break_flag = True
                break
        if break_flag == True: break
    
    print ("finished getting poi.")
    f.close()
   

import math
def gcj2wgs(lon,lat):
 
     a = 6378245.0 # 克拉索夫斯基椭球参数长半轴a
     ee = 0.00669342162296594323 #克拉索夫斯基椭球参数第一偏心率平方
     PI = 3.14159265358979324 # 圆周率
     # 以下为转换公式
     x = lon - 105.0
     y = lat - 35.0
     # 经度
     dLon = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x));
     dLon += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0;
     dLon += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0;
     dLon += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0;
     #纬度
     dLat = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x));
     dLat += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0;
     dLat += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0;
     dLat += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0;
     radLat = lat / 180.0 * PI
     magic = math.sin(radLat)
     magic = 1 - ee * magic * magic
     sqrtMagic = math.sqrt(magic)
     dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI);
     dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI);
     wgsLon = lon - dLon
     wgsLat = lat - dLat
     return wgsLon,wgsLat
 
key='6b3e360b3cfb2c89939bba4e3bbc770e'
cityname='110105'
width=100
height=120
filename=r'G:\word2vec\data\poi_cy_all9771.txt'    
get_Amap_POI(key,cityname,width,height,filename)