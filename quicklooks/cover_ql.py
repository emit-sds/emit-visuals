


from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import os



def envi_header(inputpath):
    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'



def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='aggregated abundance file')
    parser.add_argument('output_file', type=str, metavar='output file to write')
    parser.add_argument('--mask_file', type=str, default=None, metavar='l2a mask file')
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    args = parser.parse_args()


    print('read input ds')
    source_ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    trans = source_ds.GetGeoTransform()
    print(trans)

    print('warp cover')
    cover_ds = gdal.Warp('',args.landcover_file,format='MEM',resampleAlg='mode',
                         outputBounds=[trans[0],trans[3]+trans[5]*source_ds.RasterYSize, trans[0] + trans[1]*source_ds.RasterXSize, trans[3]], xRes=trans[1], yRes=trans[5])

    print('read cover')
    cover = cover_ds.ReadAsArray()
    
    print('gdal translate')
    #scale_ds = gdal.Translate('',args.input_file,format='MEM',scaleParams=[[0.0,1.0,1,255]],noData=0,outputType=gdal.GDT_Byte,bandList=[2,4,1])
    scale_ds = gdal.Translate('',args.input_file,format='MEM',scaleParams=[[0.0,1.0,1,255]],noData=0,outputType=gdal.GDT_Byte,bandList=[1,2,3])

    print('read mask')
    if args.mask_file is not None:
        #cloud = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
        #cirrus = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(2).ReadAsArray()
        print(args.mask_file)
        cloud = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,0].copy()
        cirrus = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,1].copy()
        cloud_buffer = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,4].copy()
        
        


    print('read translate')
    scale = scale_ds.ReadAsArray()
    scale[:,cloud == -9999] = 0
    scale_sum = np.sum(scale,axis=0)
    print('mask')
    scale[:,np.logical_and(cover==50, scale_sum != 0)] = np.array([180.,180.,180.])[:,np.newaxis]
    scale[:,np.logical_and(cover==80, scale_sum != 0)] = np.array([100.,100.,100.])[:,np.newaxis]
    scale[:,np.logical_and(cloud_buffer==1, scale_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
    scale[:,np.logical_and(cirrus==1, scale_sum != 0)] = np.array([200., 200., 200.])[:,np.newaxis]
    scale[:,np.logical_and(cloud==1, scale_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]

    print('write')
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
