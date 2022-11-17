from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import cv2 as cv
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
    args = parser.parse_args()

    source_ds = gdal.Open(args.input_file)

    clouds = envi.open(envi_header(args.input_file)).open_memmap(interleave='bip')

    rgb = np.zeros((clouds.shape[0],clouds.shape[1],3))
    rgb[np.all(rgb < -1,axis=-1), :] = np.nan

    # Black Buffer
    rgb[clouds[:,:,4] == 1,:] = 1

    # Grey Cirrus
    rgb[clouds[:,:,1] == 1,:] = np.array([128,128,128])[np.newaxis,:]

    # Blue Water
    rgb[clouds[:,:,2] == 1,:] = np.array([1,1,255])[np.newaxis,:]

    # Red Aerosol
    rgb[clouds[:,:,5] > 0.4,:] = np.array([255,1,1])[np.newaxis,:]

    # White Cloud
    rgb[clouds[:,:,0] == 1,:] = np.array([255,255,255])[np.newaxis,:]

    rgb[np.isnan(rgb)] = 0
    rgb = rgb.astype('uint8')


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

