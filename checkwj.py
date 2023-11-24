import os.path
import random
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
from utils import writeLog, readInfo, buildPage, getEnvOfShp, getPageByGeom, getIntersectRatio, getBoundingBox,box_iou

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--danti', type=str, default='E:\\workspace\\python_workspace\\fangpu2danti\\data\\1109\\xichengdanti.shp', help='initial weights path')
    parser.add_argument('--wj', type=str, default='E:\\workspace\\python_workspace\\fangpu2danti\\data\\1109\\xichengwj.shp', help='model.yaml path')
    parser.add_argument('--savepath', type=str,default='E:\\workspace\\python_workspace\\fangpu2danti\\data\\1109\\results.shp',help='model.yaml path')
    parser.add_argument('--thresh', type=float, default=0.8, help='dataset.yaml path')
    parser.add_argument('--codec', type=str, default='GBK', help='dataset.yaml path')
    parser.add_argument('--multiprocesses', type=int, default=8, help='number of process')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def process(args):
    danti_feature, wj_pages_info,thresh, total_single_info, result_template, all_wj = args
    wj_pages_code = getPageByGeom(danti_feature, wj_pages_info) if wj_pages_info is not None else (all_wj, -1)
    wj_page = wj_pages_code[0]
    pagecode = wj_pages_code[1]
    if pagecode == -1:
        wj_page = all_wj
    use_wj = []
    current_item = copy.deepcopy(result_template)

    for key in danti_feature.keys():
        current_item[key] = danti_feature[key]
    danti_bounding_box = getBoundingBox(danti_feature['geometry']) if len(danti_feature['geometry']) > 0 else [0, 0, 0,0]
    for j, wj_feature in enumerate(wj_page):
        wj_bounding_box = getBoundingBox(wj_feature['geometry']) if len(wj_feature['geometry']) > 0 else [0, 0, 0, 0]
        iou = box_iou(danti_bounding_box, wj_bounding_box)
        if iou > 0:
            use_wj.append(wj_page[j])

        if len(use_wj) > 1:
            valid_wj = []  # single2multi
            # feature.SetField("difficult", 1)
            for item in use_wj:
                r, r0, flag = getIntersectRatio(danti_feature['geometry'], item['geometry'])
                if r > thresh or flag == 1 or flag == 2:
                    valid_wj.append(item)

            if len(valid_wj) == 1:
                r, r0, flag = getIntersectRatio(danti_feature['geometry'], valid_wj[0]['geometry'])
                if r > thresh or flag == 2:
                    current_item['iswj'] = 0
                    for key in valid_wj[0].keys():
                        if key != 'geometry':
                            current_item[key] = valid_wj[0][key]
                elif flag == 1 and r0 < 0.95:  # multi - 1
                    current_item['iswj'] = 2
            elif len(valid_wj) > 1:
                current_item['iswj'] = 1
                for wj in valid_wj:
                    for key in wj.keys():
                        if key == 'geometry':
                            continue
                        if isinstance(wj[key], int):
                            value = wj[key]
                            value = value + wj[key]
                        if isinstance(wj[key], float):
                            value = wj[key]
                            value = value + wj[key]
                        if isinstance(wj[key], str):
                            value = wj[key]
                            values = value.split(',')
                            valid_values = []
                            for i, v in enumerate(values):
                                if v != '':
                                    valid_values.append(v)
                            if key != 'UUID':
                                if wj[key] not in valid_values:
                                    valid_values.append(wj[key])
                            else:
                                valid_values.append(wj[key])
                            value = ','.join(valid_values)
                        current_item[key] = value
            elif len(valid_wj) < 1:
                current_item['iswj'] = -1
        elif len(use_wj) == 1:
            r, r0, flag = getIntersectRatio(danti_feature['geometry'],
                                            use_wj[0]['geometry'])  # single2single or multi2single
            if r > thresh or flag == 2:
                valid_wj = use_wj[0]
                current_item["iswj"] = 0

                for key in valid_wj.keys():
                    if key != 'geometry':
                        current_item[key] = valid_wj[key]
            elif flag == 1:  # multi - 1
                current_item["iswj"] = 2
            else:
                current_item["iswj"] = -1
                valid_wj = None
        else:
            current_item["iswj"] = -1



    return current_item



def main(opt):
    danti_path, wj_path, save_path, thresh, codec, multiprocesses = opt.danti, opt.wj,opt.savepath, opt.thresh, opt.codec, opt.multiprocesses
    ogr.RegisterAll()
    if codec == 'GBK':
        gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")

    if codec == 'UTF-8':
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")

    danti_info = readInfo(danti_path, None, 'xcid', codec)
    wj_info = readInfo(wj_path, None, 'xcwjid', codec)
    wj_pages_info,num_page = buildPage(wj_info, 'wj', None, getEnvOfShp(wj_path))
    results = []
    total_single_info = len(danti_info)

    result_template ={}

    for key in danti_info[0].keys():
        value = danti_info[0][key]
        if key == 'geometry':
            continue
        if isinstance(value, int):
            result_template[key] = -1
        elif isinstance(value, float):
            result_template[key] = -1
        elif isinstance(value, str):
            result_template[key] = ''

    for key in wj_info[0].keys():
        value = wj_info[0][key]
        if (key == 'geometry') or (key in result_template.keys()):
            continue
        if isinstance(value, int):
            result_template[key] = -1
        elif isinstance(value, float):
            result_template[key] = -1
        elif isinstance(value, str):
            result_template[key] = ''



    if multiprocesses < 1:
        for i, danti_feature in enumerate(danti_info):
            print(f'processing : {i} / {total_single_info}') if i % 100 == 0 else None
            wj_pages_code = getPageByGeom(danti_info[i], wj_pages_info) if wj_pages_info is not None else (wj_pages_info, -1)
            wj_page = wj_pages_code[0]
            pagecode = wj_pages_code[1]
            if pagecode == -1:
                wj_page = wj_pages_info
            use_wj = []

            current_item = copy.deepcopy(result_template)
            for key in danti_feature.keys():
                current_item[key] = danti_feature[key]
            danti_bounding_box = getBoundingBox(danti_feature['geometry']) if len(danti_feature['geometry']) > 0 else [0, 0, 0, 0]
            for j, wj_feature in enumerate(wj_page):
                wj_bounding_box = getBoundingBox(wj_feature['geometry']) if len(wj_feature['geometry']) > 0 else [0, 0, 0, 0]
                iou = box_iou(danti_bounding_box, wj_bounding_box)
                if iou > 0:
                    use_wj.append(wj_page[j])

            if len(use_wj) > 1:
                valid_wj = []  # single2multi
                # feature.SetField("difficult", 1)
                for item in use_wj:
                    r, r0, flag = getIntersectRatio(danti_feature['geometry'], item['geometry'])
                    if r > thresh or flag == 1 or flag == 2:
                        valid_wj.append(item)

                if len(valid_wj) == 1:
                    r, r0, flag = getIntersectRatio(danti_feature['geometry'], valid_wj[0]['geometry'])
                    if r > thresh or flag == 2:
                        current_item['iswj'] = 0
                        for key in valid_wj[0].keys():
                            if key != 'geometry':
                                current_item[key] = valid_wj[0][key]
                    elif flag == 1 and r0 < 0.95:  # multi - 1
                        current_item['iswj'] = 2
                elif len(valid_wj) > 1:
                    current_item['iswj'] = 1
                    for wj in valid_wj:
                        for key in wj.keys():
                            if key == 'geometry':
                                continue
                            if isinstance(wj[key], int):
                                value = wj[key]
                                value = value + wj[key]
                            if isinstance(wj[key], float):
                                value = wj[key]
                                value = value + wj[key]
                            if isinstance(wj[key], str):
                                value = wj[key]
                                values = value.split(',')
                                valid_values = []
                                for i, v in enumerate(values):
                                    if v != '':
                                        valid_values.append(v)
                                if key != 'UUID':
                                    if wj[key] not in valid_values:
                                        valid_values.append(wj[key])
                                else:
                                    valid_values.append(wj[key])
                                value = ','.join(valid_values)
                            current_item[key] = value
                elif len(valid_wj) < 1:
                        current_item['iswj'] = -1

                results.append(current_item)
            elif len(use_wj) == 1:
                r, r0, flag = getIntersectRatio(danti_feature[i]['geometry'], use_wj[0]['geometry'])  # single2single or multi2single
                if r > thresh or flag == 2:
                    valid_wj = use_wj[0]
                    current_item["iswj"] = 0

                    for key in valid_wj.keys():
                        if key != 'geometry':
                            current_item[key] = valid_fangpu[key]
                    results.append(current_item)
                elif flag == 1:  # multi - 1
                    current_item["iswj"] = 2
                    results.append(current_item)
                else:
                    current_item["iswj"] = -1
                    results.append(current_item)
                    valid_fangpu = None
            else:
                current_item["iswj"] = -1
                results.append(current_item)
            with open('./results.pkl', 'wb') as f:
                pickle.dump(results, f)
    else:
        pool = Pool(multiprocesses)
        #results = pool.imap(process, zip(danti_info, repeat(wj_pages_info), repeat(thresh), repeat(total_single_info),repeat(result_template)))

        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataset = driver.CreateDataSource(save_path)
        layer = dataset.CreateLayer('layer', None, ogr.wkbPolygon)

        fields = []
        for key in result_template.keys():
            value = result_template[key]
            if key == 'geometry':
                continue
            if isinstance(value, int):
                field = ogr.FieldDefn(key, ogr.OFTInteger)
            elif isinstance(value, float):
                field = ogr.FieldDefn(key, ogr.OFTReal)
            elif isinstance(value, str):
                field = ogr.FieldDefn(key, ogr.OFTString)

            layer.CreateField(field)
            fields.append(key)
        field = ogr.FieldDefn('iswj', ogr.OFTInteger)
        layer.CreateField(field)
        fields.append('iswj')

        for f in pool.imap(process, zip(danti_info, repeat(wj_pages_info), repeat(thresh), repeat(total_single_info),repeat(result_template), repeat(wj_info))):
            print(f"{f['xcid']} is processing") if random.random() > 0.95 else None
            geom = f['geometry']
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
                if key == 'geometry':
                    continue
                feature.SetField(key, f[key])

            if layer.CreateFeature(feature) != 0:
                print("failed to create feature to layer")

        dataset.Destroy()









if __name__ == '__main__':
    opt = parse_opt(True)
    main(opt)
    print('done')


