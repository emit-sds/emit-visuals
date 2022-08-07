


from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess



def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='aggregated abundance file')
    parser.add_argument('output_file', type=str, metavar='output file to write')
    parser.add_argument('--mask_file', type=str, default=None, metavar='l2a mask file')
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    args = parser.parse_args()


    source_ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    trans = source_ds.GetGeoTransform()

    cover_ds = gdal.Warp('',args.landcover_file,format='MEM',resampleAlg='mode',
                         outputBounds=[trans[0],trans[3]+trans[5]*source_ds.RasterYSize, trans[0] + trans[1]*source_ds.RasterXSize, trans[3]], xRes=trans[1], yRes=trans[5])

    cover = cover_ds.ReadAsArray()
    
    #scale_ds = gdal.Translate('',args.input_file,format='MEM',scaleParams=[[0.0,1.0,1,255]],noData=0,outputType=gdal.GDT_Byte,bandList=[2,4,1])
    scale_ds = gdal.Translate('',args.input_file,format='MEM',scaleParams=[[0.0,1.0,1,255]],noData=0,outputType=gdal.GDT_Byte,bandList=[1,2,3])

    if args.mask_file is not None:
        cloud = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
        cirrus = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(2).ReadAsArray()
        


    scale = scale_ds.ReadAsArray()
    scale_sum = np.sum(scale,axis=0)
    scale[:,np.logical_and(cover==50, scale_sum != 0)] = np.array([180.,180.,180.])[:,np.newaxis]
    scale[:,np.logical_and(cover==80, scale_sum != 0)] = np.array([100.,100.,100.])[:,np.newaxis]
    scale[:,np.logical_and(cloud==1, scale_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
    scale[:,np.logical_and(cirrus==1, scale_sum != 0)] = np.array([200., 200., 200.])[:,np.newaxis]

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(args.output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte)
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(scale.shape[0]):
        outDataset.GetRasterBand(_b+1).WriteArray(scale[_b,...])
    del outDataset

    #cmd_str = f'gdal2tiles.py {args.output_file} {args.output_file}_tiled -z 7-13 --srcnodata 0 --processes=40'
    #subprocess.call(cmd_str,shell=True)

if __name__ == "__main__":
    main()
