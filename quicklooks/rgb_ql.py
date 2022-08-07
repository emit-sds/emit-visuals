from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import cv2 as cv
import os


def get_wl_inds(wl, match_wl=[641, 552, 462]):
    if np.all(wl < 10):
        wl *= 1000
    return_inds = []
    return_inds.append(np.argmin(np.abs(match_wl[0] - wl)))
    return_inds.append(np.argmin(np.abs(match_wl[1] - wl)))
    return_inds.append(np.argmin(np.abs(match_wl[2] - wl)))

    return np.array(return_inds)+1

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
    args = parser.parse_args()

    source_ds = gdal.Open(args.input_file)
    if envi.open(envi_header(args.input_file)).open_memmap(interleave='bip').shape[2] > 3:
        wl = np.array([float(x) for x in envi.open(args.input_file + '.hdr').metadata['wavelength']])
        rgb_inds = get_wl_inds(wl)

    rgb = envi.open(envi_header(args.input_file)).open_memmap(interleave='bip')[...,rgb_inds].copy()
    rgb_m = np.ma.masked_equal(rgb,np.all(rgb < -1, axis=-1))
    rgb_m = (rgb_m - rgb_m.min())*255/(rgb_m.max()-rgb_m.min())

    rgb_m -= np.min(rgb_m ,1 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    rgb_m /= np.max(rgb_m ,99,axis=(0 ,1))[np.newaxis ,np.newaxis ,:] * 255
    rgb_m[rgb_m > 255] = 255
    rgb_m = np.ma.filled(rgb_m,0).astype('uint8')

    #scale_ds = gdal.Translate('',args.input_file,format='MEM',scaleParams=[[0.0,1.0,1,255]],noData=0,outputType=gdal.GDT_Byte,bandList=[2,4,1])

    #scale = scale_ds.ReadAsArray()
    #scale_sum = np.sum(scale,axis=0)
    #scale[:,np.logical_and(cover==50, scale_sum != 0)] = np.array([128.,128.,128.])[:,np.newaxis]
    #scale[:,np.logical_and(cover==80, scale_sum != 0)] = np.array([1.,1.,1.])[:,np.newaxis]

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(args.output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte)
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(rgb_m.shape[0]):
        outDataset.GetRasterBand(_b+1).WriteArray(rgb_m[_b,...])
    del outDataset


if __name__ == "__main__":
    main()
