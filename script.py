import os.path
import sys
import math
import threading
import json
import copy
from multiprocessing.pool import Pool
from itertools import repeat
from functools import partial
from osgeo import gdal, ogr, osr
import pickle
import numpy as np
from tqdm import tqdm
import logging
import warnings
LOGGER = logging.getLogger(__name__)
warnings.filterwarnings('error')
import time
from main import readInfo, buildPage, getPageByGeom, getBoundingBox



def filter_warp(dataset, layer, index):
    f = layer.GetFeature(index)
    difficult = f.GetField('difficult')
    ids = f.GetField('UUID')

    sql = f"SELECT * FROM {layername} WHERE UUID = \'{ids}\'"
    lyr = dataset.ExecuteSQL(sql)
    n = lyr.GetFeatureCount()
    if n > 1 and difficult == 0:
        f.SetField('difficult', 2)
    layer.SetFeature(f)

def feature2Json(layer, feature):
    featureInfo = {}
    geom = feature.GetGeometryRef()
    count = geom.Boundary().GetPointCount()
    featureDefn = layer.GetLayerDefn()
    pts = []
    for i in range(count - 1):
        x = geom.Boundary().GetX(i)
        y = geom.Boundary().GetY(i)
        pts.append((x, y))
    fieldCounts = feature.GetFieldCount()
    for i in range(fieldCounts):
        fieldDefn = featureDefn.GetFieldDefn(i)
        key = fieldDefn.GetNameRef()
        dataType = fieldDefn.GetType()
        if dataType == ogr.OFTInteger:
            featureInfo[key] = feature.GetFieldAsInteger(i)

        elif dataType == ogr.OFTInteger64:
            featureInfo[key] = feature.GetFieldAsInteger64(i)

        elif dataType == ogr.OFTReal:
            featureInfo[key] = feature.GetFieldAsDouble(i)

        elif dataType == ogr.OFTStringList:
            featureInfo[key] = feature.GetFieldAsStringList(i)

        elif dataType == ogr.OFTIntegerList:
            featureInfo[key] = feature.GetFieldAsIntegerList(i)

        elif dataType == ogr.OFTInteger64List:
            featureInfo[key] = feature.GetFieldAsInteger64List(i)

        elif dataType == ogr.OFTRealList:
            featureInfo[key] = feature.GetFieldAsDoubleList(i)

        elif dataType == ogr.OFTString:
            featureInfo[key] = feature.GetFieldAsString(i)

        elif dataType == ogr.OFTDate:
            date = feature.GetField(i)
            featureInfo[key] = date



    featureInfo['geometry'] = copy.deepcopy(pts)
    return featureInfo

if __name__ == '__main__':
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    shpname = "E:/workspace/python_workspace/fangpu2danti/data/1102/exp/result.shp"
    #dataset = ogr.Open("E:/workspace/python_workspace/fangpu2danti/data/1030/result.shp", 1)
    dataset = ogr.Open(shpname, 1)
    layer = dataset.GetLayer()

    #pageField = ogr.FieldDefn('pagecode', ogr.OFTString)
    #layer.CreateField(pageField, 1)
    layer_b = readInfo(shpname,'log.txt', 'UUID')
    layer_b_page_info,_ = buildPage(layer_b, 'result', None, 500, 50)
    layername = layer.GetName()
    num = layer.GetFeatureCount()

    optimized_num = 0
    for i in range(num):
        f = layer.GetFeature(i)
        ids = f.GetField('UUID')
        difficult = f.GetField('difficult')
        if ids == None or ids == 'Null':
            continue
        fj = feature2Json(layer,f)
        page = getPageByGeom(fj, layer_b_page_info)
        page_ids = page[0]
        pagecode = page[1]
        if pagecode == -1:
            page_ids = layer_b
        num_page = len(page_ids)
        num_repeat_uuid = 0
        for j in range(num_page):
            feaureb = page_ids[j]
            uuid_ids_b = feaureb['UUID']
            if uuid_ids_b == ids:
                num_repeat_uuid = num_repeat_uuid + 1
            if num_repeat_uuid > 1:
                break

        if num_repeat_uuid > 1 and difficult == 0:
            f.SetField('difficult', 2)
            optimized_num += 1
        elif num_repeat_uuid == 1 and difficult == 2:
            f.SetField('difficult', 0)
            optimized_num += 1

        print(f"processed: {i} / {num}, optimized: {optimized_num}") if i % 100 == 0 else None
        layer.SetFeature(f)



    dataset.Destroy()




