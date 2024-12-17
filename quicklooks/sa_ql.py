



from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch

import cv2 as cv

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

def get_wl_inds(wl, match_wl=[641, 552, 462]):
    if np.all(wl < 10):
        wl *= 1000
    return_inds = []
    return_inds.append(np.argmin(np.abs(match_wl[0] - wl)))
    return_inds.append(np.argmin(np.abs(match_wl[1] - wl)))
    return_inds.append(np.argmin(np.abs(match_wl[2] - wl)))

    return np.array(return_inds)+1



def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('sa_file', type=str)
    parser.add_argument('outbase', type=str)
    parser.add_argument('--mask_file', type=str, default=None, metavar='l2a mask file')
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    args = parser.parse_args()


    source_ds = gdal.Open(args.sa_file)
    sa_ds = envi.open(envi_header(args.sa_file))
    sa = envi.open(envi_header(args.sa_file)).open_memmap(interleave='bip')
    band_names = sa_ds.metadata['band names']

    colorlist={\
    'calcite' :         [1, 255, 1],  # chartreuse
    'dolomite' :        [100, 200, 100],  # classic green
    'chlorite' :        [178, 255, 102],  # very light green - pale green
    'vermiculite' :     [200, 150, 0],  # brown
    'illite+muscovite': [100, 150, 200],  #lighter blue / green
    'montmorillonite' : [1, 255, 255],  #light blue 
    'kaolinite' :       [1, 1, 255],  #darker blue - true blue
    'gypsum' :          [175,  1,255],  # vibrant magenta - sulfate...keep away
    'goethite':         [255, 255,  1], # yellow/orange
    'hematite':         [255, 1, 1]     # red
    }

    colorlist_io = colorlist.copy()
    colorlist_no_io = colorlist.copy()
    for key in list(colorlist_io.keys()).copy():
        if key in ['goethite','hematite']:
            del colorlist_no_io[key]
        else:
            del colorlist_io[key]
    iron_oxide_idx = np.array([x.lower() in ['goethite','hematite'] for x in band_names])
    

    total_mask = sa[:,:,0] == -9999
    if args.mask_file is not None:
        print(args.mask_file)
        cloud = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,0].copy()
        cirrus = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,1].copy()
        cloud_buffer = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,4].copy()
        

    sa_sum = np.sum(sa,axis=-1)


    maxband = np.argmax(sa[...,iron_oxide_idx], axis=-1)
    un_vals = np.unique(maxband)
    output_io = np.zeros((sa.shape[0], sa.shape[1],3))
    for _v, val in enumerate(un_vals):
        print(f'{_v}/{len(un_vals)-1})')
        subset = np.logical_and(maxband == val, sa[...,iron_oxide_idx][...,val] > 0, np.logical_not(total_mask))
        minname = np.array(band_names)[iron_oxide_idx][val].lower()
        
        if np.sum(subset) > 0:

            sa_subset = sa[...,iron_oxide_idx][subset, val]
            #sa_subset -=  np.min(sa_subset)
            #sa_subset /=  np.percentile(sa_subset,95)
            #sa_subset -=  np.min(sa_subset)
            #sa_subset /=  np.max(sa_subset)
            #sa_subset = cv.equalizeHist(sa_subset.astype('uint8'))
            if np.all(sa_subset == np.min(sa_subset)):
                sa_subset[:] = 1
            else:
                sa_subset -= np.min(sa_subset)
                sa_subset /= np.max(sa_subset)
                sa_subset = (255*sa_subset).astype('uint8')
                sa_subset = cv.equalizeHist(sa_subset).astype(np.float32)/255.

            output_io[subset,:] = np.squeeze(sa_subset)[:,np.newaxis] * np.squeeze(np.array(colorlist[minname]))[np.newaxis,:]
    
    maxband = np.argmax(sa[...,np.logical_not(iron_oxide_idx)], axis=-1)
    un_vals = np.unique(maxband)
    print(un_vals)
    output_noio = np.zeros((sa.shape[0], sa.shape[1],3))
    for _v, val in enumerate(un_vals):
        print(f'{np.array(band_names)[np.logical_not(iron_oxide_idx)][val]} - {_v}/{len(un_vals)-1})')
        subset = np.logical_and(maxband == val, sa[...,np.logical_not(iron_oxide_idx)][..., val] > 0, np.logical_not(total_mask))
        minname = np.array(band_names)[np.logical_not(iron_oxide_idx)][val].lower()
        
        if np.sum(subset) > 1:

            sa_subset = sa[...,np.logical_not(iron_oxide_idx)][subset, val]
            #sa_subset -=  np.min(sa_subset)
            #sa_subset /=  np.percentile(sa_subset,95)
            #sa_subset = cv.equalizeHist((sa_subset*255).astype('uint8'))

            #print(np.sum(sa_subset >= 1))
            if np.all(sa_subset == np.min(sa_subset)):
                sa_subset[:] = 1
            else:
                sa_subset -= np.min(sa_subset)
                sa_subset /= np.max(sa_subset)
                sa_subset = (255*sa_subset).astype('uint8')
                sa_subset = cv.equalizeHist(sa_subset).astype(np.float32)/255.

            output_noio[subset,:] = np.squeeze(sa_subset)[:,np.newaxis] * np.squeeze(np.array(colorlist[minname]))[np.newaxis,:]
    





    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(args.outbase + '_io.tif',source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_io.shape[-1]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_io[...,_b])
    del outDataset

    outDataset = driver.Create(args.outbase + '_noio.tif',source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte, options=['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_noio.shape[-1]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_noio[...,_b])
    del outDataset






if __name__ == "__main__":
    main()

