from utils import readInfo, fuse, pkl2Shp
import pickle
if __name__ == '__main__':
    savepath = "E:\\workspace\\python_workspace\\fangpu2danti\\data\\1121impl\\fusion_dongcheng_1121.pkl"
    fusion = readInfo("E:\\workspace\\python_workspace\\fangpu2danti\\data\\1118\\fusion.shp", None, 'id')
    fangpu = readInfo("E:\\workspace\\python_workspace\\fangpu2danti\\data\\1118\\dongchengfangpu.shp", None, 'uuid')
    res = fuse(fusion, fangpu, 'uuid', 'difficult')
    with open(savepath, 'wb') as f:
        pickle.dump(res, f)
    pkl2Shp(savepath, savepath.replace('.pkl', '.shp'))