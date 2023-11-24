import os.path
import sys
import math
import threading
import json
import copy
from multiprocessing.pool import Pool
from osgeo import gdal, ogr, osr
import pickle
import numpy as np
from tqdm import tqdm
import logging
import warnings
from shapely.geometry import Polygon
import uuid

savapath = "E:\\workspace\\python_workspace\\fangpu2danti\\data\\1118\\dongchengfangpu.shp"
field = 'uuid'

if __name__ == '__main__':
    print("begin to set uuid to layer")
    ogr.RegisterAll()

    dataset = ogr.Open(savapath,1)
    layer = dataset.GetLayer()
    num = layer.GetFeatureCount()

    uuidField = ogr.FieldDefn(field, ogr.OFTString)
    layer.CreateField(uuidField, 1)

    for i in range(num):
        f = layer.GetFeature(i)
        id = ''.join(str(uuid.uuid4()).split('-'))
        f.SetField(field, id)
        layer.SetFeature(f)

    dataset.Destroy()
