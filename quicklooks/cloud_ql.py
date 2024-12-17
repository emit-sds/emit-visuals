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
    for twl in match_wl:
        return_inds.append(np.argmin(np.abs(twl - wl)))

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
    #parser.add_argument('--wl', type=float, nargs=3, default=[1250, 1380, 1890], metavar='l2a mask file')
    parser.add_argument('--wl', type=float, nargs=3, default=[1380, 1420, 1890], metavar='l2a mask file')
    args = parser.parse_args()

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':') 
    os.environ['PATH'] = path

    source_ds = gdal.Open(args.input_file)
    wl = np.array([float(x) for x in envi.open(envi_header(args.input_file)).metadata['wavelength']])
    cloud_inds = get_wl_inds(wl, args.wl)


    rgb = envi.open(envi_header(args.input_file)).open_memmap(interleave='bip')[...,cloud_inds].copy().astype(np.float32)
    mask = np.any(rgb == -9999, axis=-1)
    #rgb -= 0.05
    #rgb[rgb < 0.05] = 0
    rgb[rgb < 0] = 0
    rgb /= np.percentile(rgb[np.logical_not(mask),:],99,axis=0)[np.newaxis,np.newaxis,:]
    rgb[rgb > 1] = 1
    rgb *=255
    rgb = rgb.astype(int)
    rgb[rgb == 0] = 1
    rgb[mask,:] = 0

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(args.output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(rgb.shape[-1]):
        outDataset.GetRasterBand(_b+1).WriteArray(rgb[...,_b])
    del outDataset


if __name__ == "__main__":
    main()

