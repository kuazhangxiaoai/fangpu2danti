from sys import argv, exit
import os.path
import sys
import math
import threading
import json
import copy
from multiprocessing import Pool
from functools import partial
from osgeo import gdal, ogr, osr
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox
from FangPu2Danti import Ui_Form
from main import getBoundingBox, readInfo, GetFileFromThisRootDir, box_iou, getIntersectRatio, writeLog

class MainWindow(QWidget, Ui_Form):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.OpenExplorePushButton1.clicked.connect(self.setDantiDataPath)
        self.OpenExplorePushButton2.clicked.connect(self.setFangpuDataPath)
        self.OpenExplorePushButton3.clicked.connect(self.setOutputDir)
        self.pushButton.clicked.connect(self.process)
        self.GBKRadioButton.setChecked(True)
        self.progressBar.setValue(0)

    def getLayerFields(self, filename):
        ogr.RegisterAll()
        if self.GBKRadioButton.isChecked():
            gdal.SetConfigOption("GDAL_FILENAME_IS_GBK", "YES")
            gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
        elif self.UTF8RadioButton.isChecked():
            gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
            gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")

        dataset = ogr.Open(filename, 0)
        mLayer = dataset.GetLayerByIndex(0)

        fields = []
        mLayer.ResetReading()
        pt = mLayer.GetNextFeature()
        featureDefn = mLayer.GetLayerDefn()
        while (pt is not None):
            fieldCounts = pt.GetFieldCount()
            for i in range(fieldCounts):
                fieldDefn = featureDefn.GetFieldDefn(i)
                key = fieldDefn.GetNameRef()
                fields.append(key)
            break
        return fields

    def setDantiDataPath(self):
        self.dantipath =QFileDialog.getOpenFileName()[0]
        self.DantiPathLineEdit.setText(self.dantipath)
        self.allDantiFields = self.getLayerFields(self.dantipath)
        for field in self.allDantiFields:
            self.DantiUniqueCodeCombox.addItem(field)


    def setFangpuDataPath(self):
        self.fangpupath = QFileDialog.getOpenFileName()[0]
        self.FangpuLineEdit.setText(self.fangpupath)
        self.allFangpuFields = self.getLayerFields(self.fangpupath)
        for field in self.allFangpuFields:
            self.FangpuUniqueCodeCombox.addItem(field)

    def setOutputDir(self):
        self.outputdir = QFileDialog.getExistingDirectory()
        self.outputDirLineEdit.setText(self.outputdir)


    def process(self):
        self.outputname = self.outputNameLineEdit.text()
        self.dantiUniqueField = self.DantiUniqueCodeCombox.currentText()
        self.fangpuUniqueField = self.FangpuUniqueCodeCombox.currentText()

        if self.dantipath == "":
            QMessageBox.information(self, "error", "无单体数据")
            return

        if self.fangpupath == "":
            QMessageBox.information(self, "error", "无房普数据")
            return

        if self.outputname == "":
            QMessageBox.information(self, "error", "未设置输出文件名")
            return

        if self.outputdir == "":
            QMessageBox.information(self, "error", "未设置输出目录")
            return

        if self.fangpuUniqueField is None:
            QMessageBox.information(self, "error", "未设置房普数据唯一编码")
            return

        if self.dantiUniqueField is None:
            QMessageBox.information(self, "error", "未设置单体数据唯一编码")
            return


        if self.GBKRadioButton.isChecked():
            self.codec = 'GBK'
        elif self.UTF8RadioButton.isChecked():
            self.codec = 'UTF-8'



        config = {
            "dantijianzhu": self.dantipath,
            "fangpujianzhu": self.fangpupath,
            "savedir": self.outputdir,
            "savename":self.outputname,
            "log": './log.txt',
            "codec": self.codec,
            "dantijianzhu_id": self.dantiUniqueField,
            "fangpujianzhu_id": self.fangpuUniqueField
        }

        if os.path.exists(config['log']):
            os.remove(config['log'])

        savepath = os.path.join(self.outputdir, self.outputname)
        if not self.outputname.endswith('.shp'):
            self.outputname + '.shp'



        if os.path.exists(savepath):
            QMessageBox.information(self, 'error', "文件已存在")
            return

        name = self.outputname
        dir = self.outputdir

        single_info = readInfo(config['dantijianzhu'], config['log'], config['dantijianzhu_id'], config['codec'], callback=writeLog, showlog=self.writelog)
        fangpu = readInfo(config['fangpujianzhu'], config['log'], config["fangpujianzhu_id"], config['codec'], callback=writeLog, showlog=self.writelog)

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
        dataset = driver.CreateDataSource(dir)
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
            elif isinstance(value, str):
                field = ogr.FieldDefn(key, ogr.OFTString)

            Layer.CreateField(field, 1)

        for key in fangpu[0].keys():
            value = fangpu[0][key]
            if key == 'geometry':
                continue
            if isinstance(value, int):
                field = ogr.FieldDefn(key, ogr.OFTInteger)
            elif isinstance(value, float):
                field = ogr.FieldDefn(key, ogr.OFTReal)
            elif isinstance(value, str):
                field = ogr.FieldDefn(key, ogr.OFTString)

            Layer.CreateField(field, 1)

        for i in range(len(single_info)):
            self.progressBar.setValue(int(i * 100 / len(single_info)))
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
            single_info_bbox = getBoundingBox(single_info[i]['geometry']) if len(single_info[i]['geometry']) > 0 else [
                0, 0,
                0, 0]
            use_Fangpu = []
            for j in range(len(fangpu)):
                fangpu_info_bbox = getBoundingBox(fangpu[j]['geometry']) if len(fangpu[j]['geometry']) > 0 else [0, 0,
                                                                                                                 0, 0]
                iou = box_iou(single_info_bbox, fangpu_info_bbox)
                if iou > 0:
                    use_Fangpu.append(fangpu[j])
            if len(use_Fangpu) > 1:
                valid_fangpu = []
                for item in use_Fangpu:
                    r, flag = getIntersectRatio(single_info[i]['geometry'], item['geometry'])
                    if r > 0.9:
                        valid_fangpu.append(item)

                if len(valid_fangpu) == 1:
                    for key in single_info[i].keys():
                        feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                    for key in valid_fangpu[0].keys():
                        feature.SetField(key, valid_fangpu[0][key]) if key != 'geometry' else None
                if len(valid_fangpu) > 1:
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
                                value = value + ',' + fangpu_item[key]
                            feature.SetField(key, value)
                if len(valid_fangpu) < 1:
                    for key in single_info[i].keys():
                        feature.SetField(key, single_info[i][key]) if key != 'geometry' else None

                if Layer.CreateFeature(feature) != 0:
                    self.writelog("failed to create feature to layer\n")
            if len(use_Fangpu) == 1:
                r, flag = getIntersectRatio(single_info[i]['geometry'], use_Fangpu[0]['geometry'])
                if r > 0.9:
                    valid_fangpu = use_Fangpu[0]
                    for key in single_info[i].keys():
                        feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                    for key in valid_fangpu.keys():
                        feature.SetField(key, valid_fangpu[key]) if key != 'geometry' else None
                    if Layer.CreateFeature(feature) != 0:
                        self.writelog("failed to create feature to layer\n")
                else:
                    feat_id = config['dantijianzhu_id']
                    writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
                    for key in single_info[i].keys():
                        feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                    if Layer.CreateFeature(feature) != 0:
                        self.writelog("failed to create feature to layer\n")
                    valid_fangpu = None

            if len(use_Fangpu) < 1:
                feat_id = config['dantijianzhu_id']
                writeLog(config['log'], f'Cannot find good fang pu feature for {single_info[i][feat_id]} \n')
                for key in single_info[i].keys():
                    feature.SetField(key, single_info[i][key]) if key != 'geometry' else None
                if Layer.CreateFeature(feature) != 0:
                    self.writelog("failed to create feature to layer\n")

        dataset.Destroy()
        QMessageBox.information(self, "完成", 'Done')
        return

    def writelog(self, text):
        self.LogTextEdit.insertPlainText(text)


if __name__ == '__main__':
    app = QApplication(argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
