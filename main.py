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
from shapely.geometry import Polygon
import argparse
import time
from utils import matchAndFuse, pkl2GDB, pkl2GeoJSON
LOGGER = logging.getLogger(__name__)
warnings.filterwarnings('error')

from utils import writeLog, readInfo, pkl2Shp, readGDBInfo, GetAliasFromGDBTable

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def readConfig(configfile):
    with open(configfile, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
"""
def writeLog(logfilepath, error, encoding="GBK"):
    with open(logfilepath, 'a',encoding=encoding) as f:
        f.write(error)
"""
def buildPage(features, layername, filename, pagesize=1000, overlap=0):
    xmin, ymin,xmax,ymax = 999999999,999999999,0,0
    pages = {
        'layername': layername,
        'pages': []
    }
    for feature in features:
        geom = feature['geometry']
        x0,y0,x1,y1 = getBoundingBox(geom)
        if xmin > x0:
            xmin = x0
        if ymin > y0:
            ymin = y0
        if xmax < x1:
            xmax = x1
        if ymax < y1:
            ymax = y1

    box_x, box_y = xmin-overlap, ymin-overlap
    stride = pagesize
    code = 0

    while (box_x < xmax):
        box_y = ymin-overlap
        while (box_y < ymax):
            code = code + 1
            xa0, ya0, xa1, ya1 = box_x - overlap, box_y - overlap, box_x + pagesize + overlap, box_y + pagesize + overlap
            pages['pages'].append({
                "code": code,
                "bbox": [xa0, ya0, xa1, ya1],
                "featureid":[]
            })
            box_y += stride
        box_x += stride

    for i, item in enumerate(features):
        geom = item['geometry']
        x, y = geom[0]
        for page in pages['pages']:
            box = page['bbox']
            if x > box[0] and x <box[2] and y > box[1] and y <box[3]:
                page['featureid'].append(copy.deepcopy(item))
    results = {
        'layername': layername,
        'pages': []
    }
    for page in pages['pages']:
        if len(page['featureid']) > 0:
            results['pages'].append(page)

    num = 0
    for result in results['pages']:
        n = len(result['featureid'])
        num += n

    if filename is not None:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    return results, num

def loadPageMangeFile(filename):
    with open(filename, 'rb') as f:
        pagefile = pickle.load(f)
    return pagefile

def getPageByGeom(feature, pageInfo):
    geom = feature['geometry']
    x0, y0 = geom[0]
    result = None
    pagecode = -1
    for page in pageInfo['pages']:
        bbox = page['bbox']
        if x0 > bbox[0] and x0 < bbox[2] and y0 > bbox[1] and y0 <bbox[3]:
            result = page['featureid']
            pagecode = page['code']
            return (result, pagecode)

    return (None, -1)


"""
def readInfo(filename, logfile, id, num_tread=8, encoding='GBK', callback=writeLog, showlog=None):
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    dataset = ogr.Open(filename, 0)
    mLayer = dataset.GetLayerByIndex(0)

    #spatialRef = osr.SpatialReference()
    #spatialRef.ImportFromEPSG(2436)
    mLayer.ResetReading()
    pt = mLayer.GetNextFeature()
    featureDefn = mLayer.GetLayerDefn()
    info = []
    ind = 0
    while (pt is not None):
        geom = pt.GetGeometryRef()
        if geom is None:
            showlog(f'{filename}: read failed for {ind}')
            continue
        #values = [pt.GetFieldAsString(field) for field in fields]
        count = geom.Boundary().GetPointCount()



        pts = []
        for i in range(count - 1):
            x = geom.Boundary().GetX(i)
            y = geom.Boundary().GetY(i)
            pts.append((x, y))
        ind += 1
        featureInfo = {}
        fieldCounts = pt.GetFieldCount()

        for i in range(fieldCounts):
            fieldDefn = featureDefn.GetFieldDefn(i)
            key = fieldDefn.GetNameRef()
            dataType = fieldDefn.GetType()
            if dataType == ogr.OFTInteger:
                featureInfo[key] = pt.GetFieldAsInteger(i)

            elif dataType == ogr.OFTInteger64:
                featureInfo[key] = pt.GetFieldAsInteger64(i)

            elif dataType == ogr.OFTReal:
                featureInfo[key] = pt.GetFieldAsDouble(i)

            elif dataType == ogr.OFTStringList:
                featureInfo[key] = pt.GetFieldAsStringList(i)

            elif dataType == ogr.OFTIntegerList:
                featureInfo[key] = pt.GetFieldAsIntegerList(i)

            elif dataType == ogr.OFTInteger64List:
                featureInfo[key] = pt.GetFieldAsInteger64List(i)

            elif dataType == ogr.OFTRealList:
                featureInfo[key] = pt.GetFieldAsDoubleList(i)

            elif dataType == ogr.OFTString:
                featureInfo[key] = pt.GetFieldAsString(i)

            elif dataType == ogr.OFTDate:
                date = pt.GetField(i)
                featureInfo[key] = date


        if len(pts) == 0 and callback is not None:
            callback(logfile, f'{filename}: read failed for {id} - {featureInfo[id]}\n',encoding)
            if showlog is not None:
                showlog(f'{filename}: read failed for {id} - {featureInfo[id]}\n')
        else:
            featureInfo['geometry'] = copy.deepcopy(pts)
            info.append(featureInfo)
        #if ind > 100:
        #   break
        pt = mLayer.GetNextFeature()
    return info
"""
def getBoundingBox(geometry):
    if isinstance(geometry, list):
        geometry_np = np.array(geometry)
        xmin = np.min(geometry_np[:, 0])
        ymin = np.min(geometry_np[:, 1])
        xmax = np.max(geometry_np[:, 0])
        ymax = np.max(geometry_np[:, 1])
    else:
        xmin = np.min(geometry[:, 0])
        ymin = np.min(geometry[:, 1])
        xmax = np.max(geometry[:, 0])
        ymax = np.max(geometry[:, 1])

    return [xmin, ymin, xmax, ymax]



def box_iou(box_a, box_b):
    #box_a:[xmin, ymin, xmax, ymax]
    #box_b:[xmin, ymin, xmax, ymax]
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    interArea = max(0, xb-xa+1) * max(0, yb-ya+1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = interArea / (box_a_area + box_b_area + 1)
    return iou
"""
def getIntersectRatio(polygon_a, polygon_b):
    linering_a = ogr.Geometry(ogr.wkbLinearRing)
    for p in polygon_a:
        x, y = p[0], p[1]
        linering_a.AddPoint(x, y)
    polygon_a = ogr.Geometry(ogr.wkbPolygon)
    polygon_a.AddGeometryDirectly(linering_a)
    polygon_a.CloseRings()

    linering_b = ogr.Geometry(ogr.wkbLinearRing)
    for p in polygon_b:
        x, y = p[0], p[1]
        linering_b.AddPoint(x, y)
    polygon_b = ogr.Geometry(ogr.wkbPolygon)
    polygon_b.AddGeometryDirectly(linering_b)
    polygon_b.CloseRings()

    intersection_geom = polygon_b.Intersection(polygon_a)
    intersection_area = intersection_geom.GetArea()
    a_area = polygon_a.GetArea()
    b_area = polygon_b.GetArea()
    if  a_area > b_area:
        return intersection_area / b_area,a_area + 1e-5/b_area + 1e-5,  True  #one2multi or one2one
    else:
        return intersection_area / a_area, a_area + 1e-5/b_area + 1e-5, False #multi2one or multi2multi or one2one
"""
def getIntersectRatio(polygon_a, polygon_b, iou_threh=0.01):
    #2 in 1 ==> 2
    #1 in 2 ==> 1
    #1 insect 2 ==>0
    #1 away 2 ==>-1
    poly1 = Polygon(polygon_a)
    poly2 = Polygon(polygon_b)
    overlap = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = overlap / union if union > 0 else 0
    if iou < iou_threh:
        return iou, 0, -1
    if poly1.area > poly2.area:
        area_ratio = poly2.area / poly1.area
        if math.fabs(area_ratio-iou) < 0.05:
            return iou, area_ratio, 2
        else:
            return iou, area_ratio, 0
    else:
        area_ratio = poly1.area / poly2.area
        if math.fabs(area_ratio - iou) < 0.05:
            return iou, area_ratio, 1
        else:
            return iou, area_ratio, 0

def find_use_fangpu_warp(args):
    single_info, fangpu_info = args
    fangpu_info_bbox = fangpu_info['geometry']
    fangpu_info_bbox_geom = getBoundingBox(fangpu_info_bbox) if len(fangpu_info_bbox) > 0 else [0, 0, 0, 0]
    single_info_bbox = single_info['geometry']
    single_info_bbox_geom = getBoundingBox(single_info_bbox) if len(fangpu_info_bbox) > 0 else [0, 0, 0, 0]
    iou = box_iou(single_info_bbox_geom, fangpu_info_bbox_geom)
    if iou > 0:
        return fangpu_info_bbox
    else:
        return None

def main(configfile):
    config = readConfig(configfile)

    if os.path.exists(config['log']):
        os.remove(config['log'])

    name = config['savepath'].split('/')[-1].split('.')[0]
    dir = config['savepath'].split(name)[0]

    if os.path.exists(config['savepath']):
        files = GetFileFromThisRootDir(dir, ext=None)
        for file in files:
            if name in file:
                os.remove(file)

    single_info = readInfo(config['dantijianzhu'], config['log'], config['dantijianzhu_id'], config['codec'])
    fangpu = readInfo(config['fangpujianzhu'], config['log'], config["fangpujianzhu_id"], config['codec'])
    if config['processByPage']:
        if os.path.exists('./pagefile.pkl'):
            pageInfo = loadPageMangeFile('./pagefile.pkl')
        else:
            pageInfo,_ = buildPage(fangpu, name, './pagefile.pkl', 800, 100)
    else:
        pageInfo = None

    ogr.RegisterAll()
    if config['codec'] == 'GBK':
        gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    if config['codec'] == 'UTF-8':
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")

    # spatialRef = osr.SpatialReference()
    # spatialRef.ImportFromEPSG(4490)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #driver = ogr.GetDriverByName("GeoJSON")
    #driver = ogr.GetDriverByName("FileGDB")
    dataset = driver.CreateDataSource(config['savepath'])
    Layer = dataset.CreateLayer(name, None, ogr.wkbPolygon)
    assert Layer is not None

    for key in single_info[0].keys():
        value = single_info[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            field = ogr.FieldDefn(key, ogr.OFTInteger)
        elif isinstance(value, float):
            field = ogr.FieldDefn(key, ogr.OFTReal)
            field.SetPrecision(3)
        elif isinstance(value, str):
            field = ogr.FieldDefn(key, ogr.OFTString)
            field.SetWidth(len(key))
        Layer.CreateField(field)

    for key in fangpu[0].keys():
        value = fangpu[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            field = ogr.FieldDefn(key, ogr.OFTInteger)
            #field.SetWidth(32)
            #field.SetDefault(-1)
        elif isinstance(value, float):
            field = ogr.FieldDefn(key, ogr.OFTReal)
            field.SetPrecision(3)
            #field.SetWidth(32)
            #field.SetDefault(0.0)
        elif isinstance(value, str):
            field = ogr.FieldDefn(key, ogr.OFTString)
            #field.SetWidth(len(key)-2)
            #field.SetDefault('Null')

        Layer.CreateField(field)

    dicfficultField = ogr.FieldDefn('difficult', ogr.OFTInteger)  # 创建难度字段: -1: 无对应 0: 1对1 1：1对多 2：多对一 3，多对多
    #dicfficultField.SetDefault(-1)
    Layer.CreateField(dicfficultField, 1)

    #LOGGER.info(('\n' + '%10s' * 5) % ('Epoch', 'gpu_mem', 'mask', 'edge', 'img_size'))
    total_single_info = len(single_info)
    #pbar = tqdm(enumerate(single_info), total=len(single_info))
    #geomname = None
    unresolved = 0
    for i,(sif) in enumerate(single_info):
        #pbar.set_description(('%10s' * 2) % (
        #    f'{i}/{len(single_info) - 1}', unresolved))
        #feature.SetField('difficult', -1)

        print(f'processing : {i} / {total_single_info}, unresolved: {unresolved}') if i % 100 == 0 else None
        leyer_defn = Layer.GetLayerDefn()
        feature = ogr.Feature(leyer_defn)
        linering = ogr.Geometry(ogr.wkbLinearRing)
        for p in single_info[i]['geometry']:
            x, y = p[0], p[1]
            linering.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometryDirectly(linering)
        polygon.CloseRings()
        feature.SetGeometry(polygon)
        feature.SetField(config['fangpujianzhu_id'], 'Null')
        feature.SetField('difficult', -1)
        single_info_bbox = getBoundingBox(single_info[i]['geometry']) if len(single_info[i]['geometry']) > 0 else [0, 0,
                                                                                                                   0, 0]
        use_Fangpu = []
        fangpu_page_code = getPageByGeom(single_info[i], pageInfo) if pageInfo is not None else (fangpu, -1)
        fangpu_page = fangpu_page_code[0]
        pagecode = fangpu_page_code[1]
        if pagecode == -1:
            fangpu_page = fangpu
        for j in range(len(fangpu_page)):
            fangpu_info_bbox = getBoundingBox(fangpu_page[j]['geometry']) if len(fangpu_page[j]['geometry']) > 0 else [0, 0, 0, 0]
            iou = box_iou(single_info_bbox, fangpu_info_bbox)
            if iou > 0:
                use_Fangpu.append(fangpu_page[j])


        if len(use_Fangpu) > 1:
            valid_fangpu = [] #single2multi
            #feature.SetField("difficult", 1)
            for item in use_Fangpu:
                r, r0, flag = getIntersectRatio(single_info[i]['geometry'], item['geometry'])
                if r > config['iou_threshold'] or flag == 1 or flag == 2:
                    valid_fangpu.append(item)

            #print(f'selected num: {len(valid_fangpu)}')
            if len(valid_fangpu) == 0:
                unresolved = unresolved + 1



            if len(valid_fangpu) == 1:
                r, r0, flag = getIntersectRatio(single_info[i]['geometry'], valid_fangpu[0]['geometry'])
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                if r > config['iou_threshold'] or flag == 2: #1-1
                    feature.SetField("difficult", 0)
                    for key in valid_fangpu[0].keys():
                        feature.SetField(key, valid_fangpu[0][key]) if key != 'geometry' else None
                elif flag == 1 and r0 < 0.95: #multi - 1
                    feature.SetField('difficult', 2)
                    for key in valid_fangpu[0].keys():
                        if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS', "ROOM_NUM_1", "ROOM_NO_1", 'BUILDINGUN_1', 'BUILDINGUS_1']:
                            vv = round(r0 * valid_fangpu[0][key])
                            feature.SetField(key, vv)
                        else:
                            feature.SetField(key, valid_fangpu[0][key]) if key != 'geometry' else None

            if len(valid_fangpu) > 1:
                feature.SetField("difficult", 1)
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                for key in valid_fangpu[0].keys():
                    if isinstance(valid_fangpu[0][key], int) or isinstance(valid_fangpu[0][key], float):
                        feature.SetField(key, 0) if key != 'geometry' else None
                    if isinstance(valid_fangpu[0][key], str):
                        feature.SetField(key, '') if key != 'geometry' else None
                for fangpu_item in valid_fangpu:
                    for key in fangpu_item.keys():
                        if key == 'geometry':
                            continue
                        if isinstance(fangpu_item[key], int):
                            value = feature.GetFieldAsInteger(key)
                            value = value + fangpu_item[key]
                        if isinstance(fangpu_item[key], float):
                            value = feature.GetFieldAsDouble(key)
                            value = value + fangpu_item[key]
                        if isinstance(fangpu_item[key], str):
                            value = feature.GetFieldAsString(key)
                            values = value.split(',')
                            valid_values = []
                            for i, v in enumerate(values):
                                if v !='':
                                    valid_values.append(v)
                            if key != config['fangpujianzhu_id']:
                                if fangpu_item[key] not in valid_values:
                                    valid_values.append(fangpu_item[key])
                            else:
                                valid_values.append(fangpu_item[key])
                            value = ','.join(valid_values)
                        feature.SetField(key, value)
            if len(valid_fangpu) < 1:
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                feature.SetField("difficult", -1)
            if Layer.CreateFeature(feature) != 0:
                print("failed to create feature to layer")
        if len(use_Fangpu) == 1:
            r, r0, flag = getIntersectRatio(single_info[i]['geometry'], use_Fangpu[0]['geometry']) #single2single or multi2single
            if r > config['iou_threshold'] or flag == 2:
                valid_fangpu = use_Fangpu[0]
                feature.SetField("difficult", 0)
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                for key in valid_fangpu.keys():
                    feature.SetField(key, valid_fangpu[key]) if key != 'geometry' else None
                if Layer.CreateFeature(feature) != 0:
                    print("failed to create feature to layer")
            elif flag == 1:  # multi - 1
                feature.SetField('difficult', 2)
                for key in  use_Fangpu[0].keys():
                    if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS']:
                        feature.SetField(key, r0 * use_Fangpu[0][key])

                    else:
                        feature.SetField(key, use_Fangpu[0][key]) if key != 'geometry' else None

            else:
                feature.SetField("difficult", -1)
                feat_id = config['dantijianzhu_id']
                writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                if Layer.CreateFeature(feature) != 0:
                    print("failed to create feature to layer")
                valid_fangpu = None

        if len(use_Fangpu) < 1:
            feature.SetField("difficult", -1)
            feat_id = config['dantijianzhu_id']
            writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
            for key in single_info[i].keys():
                feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
            if Layer.CreateFeature(feature) != 0:
                print("failed to create feature to layer")

    print("begin to set difficult type")
    #num_features = Layer.GetFeatureCount()
    dataset.Destroy()
    print('done')

def process_warp(index, layer, single_info, fangpu_info_page, config):
    leyer_defn = layer.GetLayerDefn()
    feature = ogr.Feature(leyer_defn)
    linering = ogr.Geometry(ogr.wkbLinearRing)
    for p in single_info[index]['geometry']:
        x, y = p[0], p[1]
        linering.AddPoint(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometryDirectly(linering)
    polygon.CloseRings()
    feature.SetGeometry(polygon)
    feature.SetField(config['fangpujianzhu_id'], 'Null')
    feature.SetField('difficult', -1)
    single_info_bbox = getBoundingBox(single_info[index]['geometry']) if len(single_info[index]['geometry']) > 0 else [0, 0,
                                                                                                               0, 0]
    use_Fangpu = []
    fangpu_page, pagecode = getPageByGeom(single_info[index], fangpu_info_page) if fangpu_info_page is not None else fangpu_info_page, -1
    if pagecode == -1:
        fangpu_page = fangpu_info_page
    for j in range(len(fangpu_page)):
        fangpu_info_bbox = getBoundingBox(fangpu_page[j]['geometry']) if len(fangpu_page[j]['geometry']) > 0 else [0, 0,
                                                                                                                   0, 0]
        iou = box_iou(single_info_bbox, fangpu_info_bbox)
        if iou > 0:
            use_Fangpu.append(fangpu_page[j])

    if len(use_Fangpu) > 1:
        valid_fangpu = []  # single2multi
        # feature.SetField("difficult", 1)
        for item in use_Fangpu:
            r, r0, flag = getIntersectRatio(single_info[index]['geometry'], item['geometry'])
            if r > config['iou_threshold'] or flag == 1 or flag == 2:
                valid_fangpu.append(item)

        # print(f'selected num: {len(valid_fangpu)}')
        #if len(valid_fangpu) == 0:
        #    unresolved = unresolved + 1

        #print(f'processed : {index} / {total_single_info}') if index % 200 == 0 else None

        if len(valid_fangpu) == 1:
            r, r0, flag = getIntersectRatio(single_info[index]['geometry'], valid_fangpu[0]['geometry'])
            for key in single_info[index].keys():
                feature.SetField(key, single_info[index][key]) if key != 'geometry' else None
            if r > config['iou_threshold'] or flag == 2:  # 1-1
                feature.SetField("difficult", 0)
                for key in valid_fangpu[0].keys():
                    feature.SetField(key, valid_fangpu[0][key]) if key != 'geometry' else None
            elif flag == 1 and r0 < 0.95:  # multi - 1
                feature.SetField('difficult', 2)
                for key in valid_fangpu[0].keys():
                    if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS', "ROOM_NUM_1", "ROOM_NO_1",
                               'BUILDINGUN_1', 'BUILDINGUS_1']:
                        vv = round(r0 * valid_fangpu[0][key])
                        feature.SetField(key, vv)
                    else:
                        feature.SetField(key, valid_fangpu[0][key]) if key != 'geometry' else None

        if len(valid_fangpu) > 1:
            feature.SetField("difficult", 1)
            for key in single_info[index].keys():
                feature.SetField(key, single_info[index][key]) if key != 'geometry' else None
            for key in valid_fangpu[0].keys():
                if isinstance(valid_fangpu[0][key], int) or isinstance(valid_fangpu[0][key], float):
                    feature.SetField(key, 0) if key != 'geometry' else None
                if isinstance(valid_fangpu[0][key], str):
                    feature.SetField(key, '') if key != 'geometry' else None
            for fangpu_item in valid_fangpu:
                for key in fangpu_item.keys():
                    if key == 'geometry':
                        continue
                    if isinstance(fangpu_item[key], int):
                        value = feature.GetFieldAsInteger(key)
                        value = value + fangpu_item[key]
                    if isinstance(fangpu_item[key], float):
                        value = feature.GetFieldAsDouble(key)
                        value = value + fangpu_item[key]
                    if isinstance(fangpu_item[key], str):
                        value = feature.GetFieldAsString(key)
                        values = value.split(',')
                        valid_values = []
                        for i, v in enumerate(values):
                            if v != '':
                                valid_values.append(v)
                        if key != config['fangpujianzhu_id']:
                            if fangpu_item[key] not in valid_values:
                                valid_values.append(fangpu_item[key])
                        else:
                            valid_values.append(fangpu_item[key])
                        value = ','.join(valid_values)
                    feature.SetField(key, value)
        if len(valid_fangpu) < 1:
            for key in single_info[i].keys():
                feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
            feature.SetField("difficult", -1)
        if layer.CreateFeature(feature) != 0:
            print("failed to create feature to layer")
    if len(use_Fangpu) == 1:
        r, r0, flag = getIntersectRatio(single_info[i]['geometry'],
                                        use_Fangpu[0]['geometry'])  # single2single or multi2single
        if r > config['iou_threshold'] or flag == 2:
            valid_fangpu = use_Fangpu[0]
            feature.SetField("difficult", 0)
            for key in single_info[i].keys():
                feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
            for key in valid_fangpu.keys():
                feature.SetField(key, valid_fangpu[key]) if key != 'geometry' else None
            if layer.CreateFeature(feature) != 0:
                print("failed to create feature to layer")
        elif flag == 1:  # multi - 1
            feature.SetField('difficult', 2)
            for key in use_Fangpu[0].keys():
                if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS']:
                    feature.SetField(key, r0 * use_Fangpu[0][key])
                else:
                    feature.SetField(key, use_Fangpu[0][key]) if key != 'geometry' else None

        else:
            feature.SetField("difficult", -1)
            feat_id = config['dantijianzhu_id']
            writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
            for key in single_info[i].keys():
                feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
            if layer.CreateFeature(feature) != 0:
                print("failed to create feature to layer")
            valid_fangpu = None

    if len(use_Fangpu) < 1:
        # feature.SetField("difficult", -1)
        feat_id = config['dantijianzhu_id']
        writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
        for key in single_info[i].keys():
            feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
        if layer.CreateFeature(feature) != 0:
            print("failed to create feature to layer")

def warp(args):
    index, layer, single_info, fangpu_info_page, config = args
    process_warp(index, layer, index, layer, single_info, fangpu_info_page, config)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gdbfile', type=str, default='E:\\workspace\\python_workspace\\fangpu2danti\\data\\1124\\东城单体与房普融合处理使用数据1124.gdb', help='initial weights path')
    parser.add_argument('--dantilayer', type=str, default='东城单体2023年第二期196930幢', help='model.yaml path')
    parser.add_argument('--fangpulayer', type=str,default='东城房普总295361幢',help='model.yaml path')
    parser.add_argument('--savepath', type=str,default='E:\\workspace\\python_workspace\\fangpu2danti\\data\\1124\\dongchengfusion.pkl',help='model.yaml path')
    parser.add_argument('--thresh', type=float, default=0.9, help='dataset.yaml path')
    parser.add_argument('--codec', type=str, default='GBK', help='dataset.yaml path')
    parser.add_argument('--logfile', type=str, default='./log_dongcheng.txt', help='dataset.yaml path')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt(True)
    #danti_path, fangpu_path, savpath, thresh, codec, logfile = opt.danti, opt.fangpu, opt.savepath, opt.thresh, opt.codec, opt.logfile
    gdbfile, dantilayer, fangpulayer, savepath,thresh, codec, logfile = opt.gdbfile,opt.dantilayer, opt.fangpulayer, opt.savepath, opt.thresh, opt.codec, opt.logfile

    if os.path.exists(savepath):
        os.remove(savepath)



    if os.path.exists(logfile):
        os.remove(logfile)

    #single_info = readInfo(danti_path, None, "id", codec)
    #fangpu = readInfo(fangpu_path, None, 'uuid', codec)
    single_info = readGDBInfo(gdbfile,dantilayer, None, "ID", codec, callback=None,showlog=None, unstrict=True,unstrict_flag='IS_GEOM_CHG_DT')
    fangpu = readGDBInfo(gdbfile,fangpulayer, None, 'UUID', codec, callback=None,showlog=None, unstrict=True,unstrict_flag='IS_GEOM_CHG_FP')
    print(f'{str(len(single_info))} danti feature and {str(len(fangpu))} fangpu feature can be read')
    results = matchAndFuse(single_info,fangpu,thresh, False)

    single_alias_name = GetAliasFromGDBTable(gdbfile, dantilayer)
    fangpu_alias_name = GetAliasFromGDBTable(gdbfile, fangpulayer)
    results_alias_name = {}


    for key in results[0].keys():
        if key in fangpu_alias_name.keys():
            results_alias_name[key] = fangpu_alias_name[key]
        elif key in single_alias_name.keys():
            results_alias_name[key] = single_alias_name[key]
        else:
            results_alias_name[key] = key

    with open(savepath, 'wb') as f:  #save fusion results
        pickle.dump(results, f)

    if os.path.exists(savepath.replace('.pkl', '_aliasname.pkl')):
        os.remove(savepath.replace('.pkl', '_aliasname.pkl'))

    with open(savepath.replace('.pkl', '_aliasname.pkl'), 'wb') as f: #save aliasname file
        pickle.dump(results_alias_name, f)

    time.sleep(1)

    #pkl2Shp(savepath, savepath.replace('.pkl', '.shp'))
    pkl2GeoJSON(savepath, savepath.replace('.pkl', '.geojson'))
    print('done')

