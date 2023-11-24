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

pagesize = 200
overlap = 20

if __name__ == "__main__":
    ogr.RegisterAll()
    gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    shpname = "E:/workspace/python_workspace/fangpu2danti/data/1101/exp/result.shp"
    # dataset = ogr.Open("E:/workspace/python_workspace/fangpu2danti/data/1030/result.shp", 1)
    dataset = ogr.Open(shpname, 1)
    layer = dataset.GetLayer()

    pageField = ogr.FieldDefn('pagecode', ogr.OFTString)
    layer.CreateField(pageField, 1)

    layername = layer.GetName()
    env = layer.GetExtent()

    xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]
    box_x, box_y = env[0], env[2]
    stride = pagesize
    pagecodes = []
    pages = []
    code = 0
    env = layer.GetExtent()

    xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]
    box_x, box_y = env[0], env[2]
    stride = pagesize
    pagecodes = []
    pages = []
    code = 0
    while (box_x < xmax):
        box_y = ymin - overlap
        while (box_y < ymax):
            code = code + 1
            xa0, ya0, xa1, ya1 = box_x - overlap, box_y - overlap, box_x + pagesize + overlap, box_y + pagesize + overlap
            pagecodes.append(code)
            pages.append([xa0, ya0, xa1, ya1])
            box_y += stride
        box_x += stride
    num = layer.GetFeatureCount()
    for i in range(num):
        f = layer.GetFeature(i)
        geom = f.GetGeometryRef()
        x, y = geom.Boundary().GetX(0), geom.Boundary().GetY(0)
        for j in range(len(pagecodes)):
            x_min, y_min, x_max, y_max = pages[j]
            if x > x_min and x < x_max and y > y_min and y < y_max:
                f.SetField('pagecode', str(pagecodes[j]))
                layer.SetFeature(f)
                break
    dataset.Destroy()
