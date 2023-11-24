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
import uuid
from tqdm import tqdm
import logging
import warnings
from shapely.geometry import Polygon



def writeLog(logfilepath, error, encoding="GBK"):
    with open(logfilepath, 'a',encoding=encoding) as f:
        f.write(error)

def readGDBInfo(filename,layername, logfile, id, encoding='GBK', callback=writeLog, showlog=None, unstrict=False, unstrict_flag='IS_GEOM_CHG'):
    assert encoding in ['GBK', 'UTF8']
    if encoding == 'GBK':
        gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    elif encoding == 'UTF8':
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")

    #driver = ogr.GetDriverByName('OpenFileGDB')
    datasource = gdal.OpenEx(filename, gdal.OF_VECTOR)

    n_layer = datasource.GetLayerCount()
    queried_layer = datasource.GetLayerByName(layername)
    #for i in range(n_layer):
    #    layer = datasource.GetLayerByIndex(i)
    #    if(layer.GetName() == layername):
    #        queried_layer = datasource.GetLayerByIndex(i)

    if queried_layer is None:
        print(f"{layername} is not exist in gdb file")
        return None

    featureDefn = queried_layer.GetLayerDefn()

    n = queried_layer.GetFeatureCount()
    info = []
    for ind in range(n):
        f = queried_layer.GetFeature(ind)
        if f is None:
            print(f'{str(ind)} of layers cannot be read')
            continue
        geom = f.GetGeometryRef()
        if geom is None:
            print(f'{filename}: read failed for {ind}')
            continue

        geomType = geom.GetGeometryType()
        assert geomType == ogr.wkbPolygon or ogr.wkbMultiPolygon, f"geometry type of {geomType} is error, which is {id} of {f.GetField('UUID')}"
        if geomType == ogr.wkbMultiPolygon:
            geom = geom.GetGeometryRef(0)
        count = geom.Boundary().GetPointCount()
        pts = []
        featureInfo = {}
        if count == 0 and unstrict:
            geom = geom.GetGeometryRef(0)
            if (not geom is None):
                count = geom.GetPointCount()
            else:
                continue
            for i in range(count - 1):
                x = geom.GetX(i)
                y = geom.GetY(i)
                pts.append((x, y))
            featureInfo[unstrict_flag] = 'Y'
        elif count > 0 and unstrict:
            for i in range(count - 1):
                x = geom.Boundary().GetX(i)
                y = geom.Boundary().GetY(i)
                pts.append((x, y))
            featureInfo[unstrict_flag] = 'N'
        elif count > 0 and (not unstrict):
            for i in range(count - 1):
                x = geom.Boundary().GetX(i)
                y = geom.Boundary().GetY(i)
                pts.append((x, y))


        fieldCounts = f.GetFieldCount()

        for i in range(fieldCounts):
            fieldDefn = featureDefn.GetFieldDefn(i)
            key = fieldDefn.GetNameRef()
            dataType = fieldDefn.GetType()
            if dataType == ogr.OFTInteger:
                featureInfo[key] = f.GetFieldAsInteger(i)

            elif dataType == ogr.OFTInteger64:
                featureInfo[key] = f.GetFieldAsInteger64(i)

            elif dataType == ogr.OFTReal:
                featureInfo[key] = f.GetFieldAsDouble(i)

            elif dataType == ogr.OFTStringList:
                featureInfo[key] = f.GetFieldAsStringList(i)

            elif dataType == ogr.OFTIntegerList:
                featureInfo[key] = f.GetFieldAsIntegerList(i)

            elif dataType == ogr.OFTInteger64List:
                featureInfo[key] = f.GetFieldAsInteger64List(i)

            elif dataType == ogr.OFTRealList:
                featureInfo[key] = f.GetFieldAsDoubleList(i)

            elif dataType == ogr.OFTString:
                featureInfo[key] = f.GetFieldAsString(i)

            elif dataType == ogr.OFTDate:
                date = f.GetField(i)
                featureInfo[key] = date

        if len(pts) == 0 and callback is not None and logfile is not None:
            callback(logfile, f'{filename}: read failed for {id} - {featureInfo[id]}\n', encoding)
            if showlog is not None:
                showlog(f'{filename}: read failed for {id} - {featureInfo[id]}\n')
        elif len(pts) == 0 and logfile is None:
            print(f'{filename}: read failed for {id} - {featureInfo[id]}')
        else:
            featureInfo['geometry'] = copy.deepcopy(pts)
            info.append(featureInfo)

    return info




def readShpInfo(filename, logfile, id, encoding='GBK', callback=writeLog, showlog=None):
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    dataset = ogr.Open(filename, 0)
    layer = dataset.GetLayerByIndex(0)

    # spatialRef = osr.SpatialReference()
    # spatialRef.ImportFromEPSG(2436)
    #layer.ResetReading()
    featureDefn = layer.GetLayerDefn()

    n = layer.GetFeatureCount()
    info = []
    for ind in range(n):
        f = layer.GetFeature(ind)
        geom = f.GetGeometryRef()
        if geom is None:
            print(f'{filename}: read failed for {ind}')
            continue

        geomType = geom.GetGeometryType()
        assert geomType == ogr.wkbPolygon, f"geometry type of {geomType} is error, which is {id} of {f.GetField('UUID')}"
        count = geom.Boundary().GetPointCount()
        pts = []
        for i in range(count - 1):
            x = geom.Boundary().GetX(i)
            y = geom.Boundary().GetY(i)
            pts.append((x, y))

        featureInfo = {}
        fieldCounts = f.GetFieldCount()

        for i in range(fieldCounts):
            fieldDefn = featureDefn.GetFieldDefn(i)
            key = fieldDefn.GetNameRef()
            dataType = fieldDefn.GetType()
            if dataType == ogr.OFTInteger:
                featureInfo[key] = f.GetFieldAsInteger(i)

            elif dataType == ogr.OFTInteger64:
                featureInfo[key] = f.GetFieldAsInteger64(i)

            elif dataType == ogr.OFTReal:
                featureInfo[key] = f.GetFieldAsDouble(i)

            elif dataType == ogr.OFTStringList:
                featureInfo[key] = f.GetFieldAsStringList(i)

            elif dataType == ogr.OFTIntegerList:
                featureInfo[key] = f.GetFieldAsIntegerList(i)

            elif dataType == ogr.OFTInteger64List:
                featureInfo[key] = f.GetFieldAsInteger64List(i)

            elif dataType == ogr.OFTRealList:
                featureInfo[key] = f.GetFieldAsDoubleList(i)

            elif dataType == ogr.OFTString:
                featureInfo[key] = f.GetFieldAsString(i)

            elif dataType == ogr.OFTDate:
                date = f.GetField(i)
                featureInfo[key] = date

        if len(pts) == 0 and callback is not None and logfile is not None:
            callback(logfile, f'{filename}: read failed for {id} - {featureInfo[id]}\n',encoding)
            if showlog is not None:
                showlog(f'{filename}: read failed for {id} - {featureInfo[id]}\n')
        elif len(pts) == 0 and logfile is None:
            print(f'{filename}: read failed for {id} - {featureInfo[id]}')
        else:
            featureInfo['geometry'] = copy.deepcopy(pts)
            info.append(featureInfo)

    return info


def readInfo(filename, logfile, id, encoding='GBK', callback=writeLog, showlog=None, filetype='ESRI Shapefile'):
    if filetype == 'ESRI Shapefile':
        return readShpInfo(filename, logfile, id, encoding, callback, showlog)
    elif filetype == 'OpenFileGDB':
        return readGDBInfo(filename, logfile, id, encoding, callback, showlog)

def buildPage(features, layername, filename, env=None, pagesize=1000, overlap=0):
    pages = {
        'name': layername,
        'pages':[]
    }
    #env = layer.GetExtent()
    if env is not None:
        xmin, ymin, xmax, ymax = env
    else:
        xmin, ymin, xmax, ymax = 999999999, 999999999, 0, 0

        for feature in features:
            geom = feature['geometry']
            x0, y0, x1, y1 = getBoundingBox(geom)
            if xmin > x0:
                xmin = x0
            if ymin > y0:
                ymin = y0
            if xmax < x1:
                xmax = x1
            if ymax < y1:
                ymax = y1
    box_x, box_y = xmin - overlap, ymin - overlap
    stride = pagesize
    code = 0

    while (box_x < xmax):
        box_y = ymin - overlap
        while (box_y < ymax):
            code = code + 1
            xa0, ya0, xa1, ya1 = box_x - overlap, box_y - overlap, box_x + pagesize + overlap, box_y + pagesize + overlap
            pages['pages'].append({
                "code": code,
                "bbox": [xa0, ya0, xa1, ya1],
                "featureid": []
            })
            box_y += stride
        box_x += stride

    for i, item in enumerate(features):
        geom = item['geometry']
        x, y = geom[0]
        for page in pages['pages']:
            box = page['bbox']
            if x > box[0] and x < box[2] and y > box[1] and y < box[3]:
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

def getEnvOfShp(shpname):
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    dataset = ogr.Open(shpname, 1)
    layer = dataset.GetLayer()
    env = layer.GetExtent()

    xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]
    dataset.Destroy()
    return [xmin, ymin, xmax, ymax]

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

def getIntersectRatio(polygon_a, polygon_b, iou_threh=0.01):
    #2 in 1 ==> 2
    #1 in 2 ==> 1
    #1 insect 2 ==>0
    #1 away 2 ==>-1
    def adjust(iou, poly1, poly2, t=0.5, t1=0.1):
        if poly1.area > poly2.area:
            ar = poly2.area / poly1.area
            if math.fabs(iou / ar) > 0.9 and ar < t and iou > t1:
                return 1,1,True
            else:
                return iou,ar,False
        else:
            ar = poly1.area / poly2.area
            if math.fabs(iou / ar) < 0.9 and ar < t and iou > t1:
                return 1,1,True
            else:
                return iou,ar,False

    poly1 = Polygon(polygon_a)
    poly2 = Polygon(polygon_b)
    overlap = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = overlap / union if union > 0 else 0
    iou, ar, is_adjust = adjust(iou, poly1, poly2)

    if iou < iou_threh:
        return iou, 0, -1
    if poly1.area > poly2.area:
        area_ratio = poly2.area / poly1.area if not is_adjust else ar
        if math.fabs(area_ratio-iou) < 0.05:
            return iou, area_ratio, 2
        else:
            return iou, area_ratio, 0
    else:
        area_ratio = poly1.area / poly2.area if not is_adjust else ar
        if math.fabs(area_ratio - iou) < 0.05:
            return iou, area_ratio, 1
        else:
            return iou, area_ratio, 0

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

def pkl2GDB(pklname, gdbname, tablename):
    with open(pklname, 'rb') as f:
        featuresOfPkl = pickle.load(f)
    n = len(featuresOfPkl)
    #os.chdir(gdbname)
    driver = ogr.GetDriverByName("FileGDB")
    #datasource = gdal.OpenEx(gdbname, gdal.OF_VECTOR)
    datasource = driver.Open(gdbname, 0)
    layer = datasource.CreateLayer(tablename, None, ogr.wkbPolygon)
    #layer = dataset.CreateLayer('layer', None, ogr.wkbPolygon)
    fields = []
    for key in featuresOfPkl[0].keys():
        value = featuresOfPkl[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            field = ogr.FieldDefn(key, ogr.OFTInteger)
        elif isinstance(value, float):
            field = ogr.FieldDefn(key, ogr.OFTReal)
            field.SetPrecision(3)
        elif isinstance(value, str):
            field = ogr.FieldDefn(key, ogr.OFTString)

        layer.CreateField(field)
        fields.append(key)

    for i in range(n):
        f = featuresOfPkl[i]
        if 'geometry' in f.keys():
            geom = f['geometry']
        else:
            continue
        leyer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(leyer_defn)
        linering = ogr.Geometry(ogr.wkbLinearRing)
        for p in geom:
            x, y = p[0], p[1]
            linering.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometryDirectly(linering)
        polygon.CloseRings()
        feature.SetGeometry(polygon)

        for key in fields:
            if key in f.keys():
                feature.SetField(key, f[key])

        if layer.CreateFeature(feature) != 0:
            print("failed to create feature to layer")

    datasource.Destroy()

def pkl2GeoJSON(pklname, jsonname):
    with open(pklname, 'rb') as f:
        featuresOfPkl = pickle.load(f)
    n = len(featuresOfPkl)
    driver = ogr.GetDriverByName("GeoJSON")
    dataset = driver.CreateDataSource(jsonname)
    layer = dataset.CreateLayer('layer', None, ogr.wkbPolygon)

    fields = []
    for key in featuresOfPkl[0].keys():
        value = featuresOfPkl[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            field = ogr.FieldDefn(key, ogr.OFTInteger)
        elif isinstance(value, float):
            field = ogr.FieldDefn(key, ogr.OFTReal)
            field.SetPrecision(3)
        elif isinstance(value, str):
            field = ogr.FieldDefn(key, ogr.OFTString)

        layer.CreateField(field)
        fields.append(key)

    for i in range(n):
        f = featuresOfPkl[i]
        if 'geometry' in f.keys():
            geom = f['geometry']
        else:
            continue
        leyer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(leyer_defn)
        linering = ogr.Geometry(ogr.wkbLinearRing)
        for p in geom:
            x, y = p[0], p[1]
            linering.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometryDirectly(linering)
        polygon.CloseRings()
        feature.SetGeometry(polygon)

        for key in fields:
            if key in f.keys():
                feature.SetField(key, f[key])


        if layer.CreateFeature(feature) != 0:
            print("failed to create feature to layer")


    dataset.Destroy()


def pkl2Shp(pklname, shpname):
    with open(pklname, 'rb') as f:
        featuresOfPkl = pickle.load(f)
    n = len(featuresOfPkl)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataset = driver.CreateDataSource(shpname)
    layer = dataset.CreateLayer('layer', None, ogr.wkbPolygon)

    fields = []
    for key in featuresOfPkl[0].keys():
        value = featuresOfPkl[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            field = ogr.FieldDefn(key, ogr.OFTInteger)
        elif isinstance(value, float):
            field = ogr.FieldDefn(key, ogr.OFTReal)
            field.SetPrecision(3)
        elif isinstance(value, str):
            field = ogr.FieldDefn(key, ogr.OFTString)

        layer.CreateField(field)
        fields.append(key)

    for i in range(n):
        f = featuresOfPkl[i]
        if 'geometry' in f.keys():
            geom = f['geometry']
        else:
            continue
        leyer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(leyer_defn)
        linering = ogr.Geometry(ogr.wkbLinearRing)
        for p in geom:
            x, y = p[0], p[1]
            linering.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometryDirectly(linering)
        polygon.CloseRings()
        feature.SetGeometry(polygon)

        for key in fields:
            if key in f.keys():
                feature.SetField(key, f[key])


        if layer.CreateFeature(feature) != 0:
            print("failed to create feature to layer")


    dataset.Destroy()




def matchAndFuse(single_info, fangpu_info, thresh=0.9, makeuuid=False):
    if makeuuid:
        for i in range(len(fangpu_info)):
            fangpu_info[i]['uuid'] = ''.join(str(uuid.uuid4()).split('-'))

    fangpu_pages_info,_ = buildPage(fangpu_info,'fangpu',None,None,800, 100)
    results = []
    num = len(single_info)
    for i in range(num):
        print(f'processing : {i} / {num}') if i % 100 == 0 else None
        merge_feature = {}
        #for key in single_info[i].keys():
        #    value = single_info[i][key]
        #    merge_feature[key] = value
        merge_feature['difficult'] = -1
        single_info_bbox = getBoundingBox(single_info[i]['geometry']) if len(single_info[i]['geometry']) > 0 else [0, 0, 0, 0]

        use_Fangpu = []
        fangpu_page_code = getPageByGeom(single_info[i], fangpu_pages_info) if fangpu_pages_info is not None else (fangpu_info, -1)
        fangpu_page = fangpu_page_code[0]
        pagecode = fangpu_page_code[1]
        if pagecode == -1:
            fangpu_page = fangpu_info
        for j in range(len(fangpu_page)):
            fangpu_info_bbox = getBoundingBox(fangpu_page[j]['geometry']) if len(fangpu_page[j]['geometry']) > 0 else [
                0, 0, 0, 0]
            iou = box_iou(single_info_bbox, fangpu_info_bbox)
            if iou > 0:
                use_Fangpu.append(fangpu_page[j])

        if len(use_Fangpu) > 1:
            valid_fangpu = []
            #feature.SetField("difficult", 1)
            for item in use_Fangpu:
                r, r0, flag = getIntersectRatio(single_info[i]['geometry'], item['geometry'])
                if r > thresh or flag == 1 or flag == 2:
                    valid_fangpu.append(item)

            #print(f'selected num: {len(valid_fangpu)}')
            #if len(valid_fangpu) == 0:
            #    unresolved = unresolved + 1


            if len(valid_fangpu) == 1:
                r, r0, flag = getIntersectRatio(single_info[i]['geometry'], valid_fangpu[0]['geometry'])
                for key in single_info[i].keys():
                    merge_feature[key] = single_info[i][key]
                if r > thresh or flag == 2: #1-1
                    merge_feature['difficult'] = 0
                    for key in valid_fangpu[0].keys():
                        if key != 'geometry':
                            merge_feature[key] = valid_fangpu[0][key]
                elif flag == 1 and r0 < 0.95: #multi - 1
                    merge_feature['difficult'] = 2
                    #feature.SetField('difficult', 2)
                    for key in valid_fangpu[0].keys():
                        if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS','ROOM_NUM_1']:
                            single_area = Polygon(single_info[i]['geometry']).area
                            fangpu_area = Polygon(valid_fangpu[0]['geometry']).area
                            r0 = single_area / fangpu_area if single_area < fangpu_area else fangpu_area / single_area
                            vv = round(r0 * valid_fangpu[0][key])
                            merge_feature[key] = vv
                        else:
                            if key != 'geometry':
                                merge_feature[key] = valid_fangpu[0][key]
                results.append(merge_feature)
            if len(valid_fangpu) > 1:
                merge_feature['difficult'] = 1 #1-multi
                for key in single_info[i].keys():
                    merge_feature[key] = single_info[i][key]
                for key in valid_fangpu[0].keys():
                    if isinstance(valid_fangpu[0][key], int) or isinstance(valid_fangpu[0][key], float):
                        merge_feature[key] = 0
                    if isinstance(valid_fangpu[0][key], str):
                        merge_feature[key] = ''

                for fangpu_item in valid_fangpu:
                    for key in fangpu_item.keys():
                        if key == 'geometry':
                            continue
                        if isinstance(fangpu_item[key], int):
                            value = merge_feature[key]
                            value = value + fangpu_item[key]
                        if isinstance(fangpu_item[key], float):
                            value = merge_feature[key]
                            value = value + fangpu_item[key]
                        if isinstance(fangpu_item[key], str):
                            value = merge_feature[key]
                            values = value.split(',')
                            valid_values = []
                            for i, v in enumerate(values):
                                    valid_values.append(v)

                            if fangpu_item[key] not in valid_values:
                                valid_values.append(fangpu_item[key])

                            value = (','.join(valid_values)).strip(',')

                        merge_feature[key] = value
                results.append(merge_feature)
            if len(valid_fangpu) < 1:
                for key in single_info[i].keys():
                    merge_feature[key] = single_info[i][key]
                for key in fangpu_info[0].keys():
                    if isinstance(fangpu_info[0][key], int) or isinstance(fangpu_info[0][key], float):
                        if key != 'geometry':
                            merge_feature[key] = 0
                    if isinstance(fangpu_info[0][key], str):
                        if key != 'geometry':
                            merge_feature[key] = ''

                merge_feature["difficult"] = -1
                results.append(merge_feature)

        if len(use_Fangpu) == 1:
            r, r0, flag = getIntersectRatio(single_info[i]['geometry'], use_Fangpu[0]['geometry']) #single2single or multi2single
            if r > thresh or flag == 2:
                valid_fangpu = use_Fangpu[0]
                merge_feature['difficult'] = 0
                for key in single_info[i].keys():
                    merge_feature[key] = single_info[i][key]
                for key in valid_fangpu.keys():
                    if key != 'geometry':
                        merge_feature[key] = valid_fangpu[key]
                results.append(merge_feature)
            elif flag == 1:  # multi - 1
                merge_feature['difficult'] = 2
                for key in  use_Fangpu[0].keys():
                    if key in ['ROOM_NUM', 'ROOM_NO', 'BUILDINGUN', 'BUILDINGUS']:
                        single_area = Polygon(single_info[i]['geometry']).area
                        fangpu_area = Polygon(use_Fangpu[0]['geometry']).area
                        r0 = single_area / fangpu_area if single_area < fangpu_area else fangpu_area / single_area
                        merge_feature[key]= r0 * use_Fangpu[0][key]
                    else:
                        if key != 'geometry':
                            merge_feature[key]= use_Fangpu[0][key]
                results.append(merge_feature)

            else:
                merge_feature['difficult'] = -1
                #feat_id = config['dantijianzhu_id']
                #writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
                for key in single_info[i].keys():
                    merge_feature[key] = single_info[i][key]
                for key in fangpu_info[0].keys():
                    if isinstance(fangpu_info[0][key], int) or isinstance(fangpu_info[0][key], float):
                        if key != 'geometry':
                            merge_feature[key] = 0
                    if isinstance(fangpu_info[0][key], str):
                        if key != 'geometry':
                            merge_feature[key] = ''
                results.append(merge_feature)
                valid_fangpu = None

        if len(use_Fangpu) < 1:
            merge_feature['difficult'] = -1
            for key in single_info[i].keys():
                merge_feature[key] = single_info[i][key]
            for key in fangpu_info[0].keys():
                if isinstance(fangpu_info[0][key], int) or isinstance(fangpu_info[0][key], float):
                    if key != 'geometry':
                        merge_feature[key] = 0
                if isinstance(fangpu_info[0][key], str):
                    if key != 'geometry':
                        merge_feature[key] = ''

            results.append(merge_feature)
        #merge_feature['geometry'] = single_info[i]['geometry']

    return results

def fuse(single_info, fangpu_info, match_field, match_type):
    single_info_fields = single_info[0].keys()
    assert match_field in single_info_fields
    assert match_type in single_info_fields

    num_single_info = len(single_info)
    num_fangpu_info = len(fangpu_info)

    print(f'Number of danti is {num_single_info}, Number of Fangpu is {num_fangpu_info}')
    fangpu_pages_info,_ = buildPage(fangpu_info,'fangpu',None,None,800, 100)

    for i in range(num_single_info):
        print(f'processing : {i} / {num_single_info}') if i % 100 == 0 else None
        current_single = single_info[i]
        current_match_uuid = current_single[match_field]
        current_match_uuids = current_match_uuid.split(',')
        current_type = current_single[match_type]
        fangpu_page_code = getPageByGeom(single_info[i], fangpu_pages_info) if fangpu_pages_info is not None else (
        fangpu_info, -1)
        fangpu_page = fangpu_page_code[0]
        pagecode = fangpu_page_code[1]

        if pagecode == -1:
            fangpu_page = fangpu_info
        num_fangpu_info = len(fangpu_page)

        if current_type == 2 or current_type == '2':
            assert len(current_match_uuids) == 1
            for j in range(num_fangpu_info):
                if current_match_uuids[0] == fangpu_page[j]['uuid']:
                    single_info[i]['JZGM'] = fangpu_page[j]['JZGM']
                    single_info[i]['UPFLOORS'] = fangpu_page[j]['UPFLOORS']
                    break
        elif current_type == 1 or current_type == '1':
            assert len(current_match_uuids) > 1
            single_info[i]['JZGM'] = ''
            single_info[i]['UPFLOORS'] =''

            for j in range(num_fangpu_info):
                if fangpu_page[j]['uuid'] in current_match_uuids:
                    single_info[i]['JZGM'] = single_info[i]['JZGM'] +',' + fangpu_page[j]['JZGM']
                    single_info[i]['UPFLOORS'] = single_info[i]['UPFLOORS'] + ',' + fangpu_page[j]['UPFLOORS']
            single_info[i]['JZGM'] = single_info[i]['JZGM'].strip(',')
            single_info[i]['UPFLOORS'] = single_info[i]['UPFLOORS'].strip(',')

        elif current_type == 0 or current_type == '0':
            assert len(current_match_uuids) == 1
            for j in range(num_fangpu_info):
                if current_match_uuids[0] == fangpu_page[j]['uuid']:
                    single_info[i]['JZGM'] = fangpu_page[j]['JZGM']
                    single_info[i]['UPFLOORS'] = fangpu_page[j]['UPFLOORS']
                    break

        elif current_type == -1 or current_type == '-1':
            single_info[i]['JZGM'] = ''
            single_info[i]['UPFLOORS'] = ''

    return single_info

def GetAliasFromGDBTable(filename, layername, encoding='GBK'):
    assert encoding in ['GBK', 'UTF8']
    if encoding == 'GBK':
        gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    elif encoding == 'UTF8':
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")

    # driver = ogr.GetDriverByName('OpenFileGDB')
    datasource = gdal.OpenEx(filename, gdal.OF_VECTOR)

    n_layer = datasource.GetLayerCount()
    queried_layer = None
    for i in range(n_layer):
        layer = datasource.GetLayerByIndex(i)
        if (layer.GetName() == layername):
            queried_layer = datasource.GetLayerByIndex(i)

    if queried_layer is None:
        print(f"{layername} is not exist in gdb file")
        return None

    featureDefn = queried_layer.GetLayerDefn()

    n = featureDefn.GetFieldCount()
    fields = {}
    for i in range(n):
        field = featureDefn.GetFieldDefn(i)
        name = field.GetNameRef()
        alias = field.GetAlternativeName()
        fields[name]=alias  # name-> 名字  alias->别名
    return fields

def SetAliasFromGDBTable(filename, layername, names, encoding='GBK'):
    assert encoding in ['GBK', 'UTF8']
    if encoding == 'GBK':
        gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    elif encoding == 'UTF8':
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")

    # driver = ogr.GetDriverByName('OpenFileGDB')
    datasource = gdal.OpenEx(filename, gdal.OF_VECTOR)

    n_layer = datasource.GetLayerCount()
    queried_layer = None
    for i in range(n_layer):
        layer = datasource.GetLayerByIndex(i)
        if (layer.GetName() == layername):
            queried_layer = datasource.GetLayerByIndex(i)

    if queried_layer is None:
        print(f"{layername} is not exist in gdb file")
        return False

    featureDefn = queried_layer.GetLayerDefn()
    n = featureDefn.GetFieldCount()

    for i in range(n):
        field = featureDefn.GetFieldDefn(i)
        key = field.GetNameRef()
        field.SetAlternativeName(names[key])

    return True
