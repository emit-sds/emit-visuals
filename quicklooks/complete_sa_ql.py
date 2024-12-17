


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
from collections import OrderedDict






def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_img.shape[0]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_img[_b,...])
        outDataset.GetRasterBand(_b+1).SetNoDataValue(0)
    del outDataset




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
    parser.add_argument('--lc', action='store_true')
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    parser.add_argument('--scaling_type',type=str,default='fixed-linear',choices=['fixed-linear','percentile'])
    parser.add_argument('--scaling_bounds',type=float,default=[0,0.5], nargs=2)
    args = parser.parse_args()


    print('read input ds')
    source_ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    trans = source_ds.GetGeoTransform()
    print(trans)

    if args.lc:
        print('warp cover')
        cover_ds = gdal.Warp('',args.landcover_file,format='MEM',resampleAlg='mode',
                             outputBounds=[trans[0],trans[3]+trans[5]*source_ds.RasterYSize, trans[0] + trans[1]*source_ds.RasterXSize, trans[3]], xRes=trans[1], yRes=trans[5])

        print('read cover')
        cover = cover_ds.ReadAsArray()
    
    print('read sa')
    sa = source_ds.ReadAsArray() 
    if not args.lc:
        cover = np.zeros((sa.shape[1],sa.shape[2]))-1

    if args.mask_file is not None:
        print('read mask')
        #cloud = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
        #cirrus = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(2).ReadAsArray()
        cloud = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,0].copy()
        cirrus = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,1].copy()
        cloud_buffer = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,4].copy()
 
    
        sa[:,cloud == -9999] = -9999

    # neutral grey for all others.


    for band in range(1,295):
        print(f'{band}')

        output_img = np.zeros((3, sa.shape[1], sa.shape[2]))
        if band < 96:
            subset = sa[1,...] == band
        else:
            subset = sa[3,...] == band

        if args.lc:
            subset[cover==50] = False
            subset[cover==80] = False

        if args.mask_file is not None:
            subset[cirrus==1] = False
            subset[cloud==1] = False
            subset[cloud_buffer==1] = False

        if np.sum(subset) < 10:
            continue
        
        output_img[:, subset] = np.array([254., 254., 254.])[:,np.newaxis]
        if args.scaling_type == 'fixed-linear':

            if band < 96:
                scale_dat = ((sa[0, subset] - args.scaling_bounds[0]) / (args.scaling_bounds[1]-args.scaling_bounds[0]))
            else:
                scale_dat = ((sa[2, subset] - args.scaling_bounds[0]) / (args.scaling_bounds[1]-args.scaling_bounds[0]))
        else:
            if band < 96:
                scale_dat = ((sa[0, subset] - np.percentile(sa[band,subset], args.scaling_bounds[0])) / np.percentile(sa[0,subset],args.scaling_bounds[1]))[np.newaxis,:]
            else:
                scale_dat = ((sa[2, subset] - np.percentile(sa[band,subset], args.scaling_bounds[0])) / np.percentile(sa[2,subset],args.scaling_bounds[1]))[np.newaxis,:]

        scale_dat[scale_dat < 0] = 0
        scale_dat[scale_dat > 1] = 1
        output_img[:, subset] *= scale_dat[np.newaxis,:] # colorize
        
        if args.lc:
            output_img[:,np.logical_and(cover==50, subset)] = np.array([180.,180.,180.])[:,np.newaxis]
            output_img[:,np.logical_and(cover==80, subset)] = np.array([100.,100.,100.])[:,np.newaxis]
        if args.mask_file is not None:
            output_img[:,np.logical_and(cirrus==1, subset)] = np.array([200., 200., 200.])[:,np.newaxis]
            output_img[:,np.logical_and(cloud==1, subset)]  = np.array([254., 254., 254.])[:,np.newaxis]

        output_img = np.round(output_img)
        output_img[:, subset] += 1 # account for 1-based offset for all-0 nodata
        output_img = output_img.astype(np.uint8)
 
        write_output_file(source_ds, output_img, f'{args.output_file}_c{band}.tif')

        #for band in range(len(band_names)):
        #    cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="gdal2tiles.py {args.output_file}_{band_names[band]} {args.output_file}_{band_names[band]}_tiled -z 7-13 --srcnodata 0 --processes=40"'
        #    subprocess.call(cmd_str,shell=True)





if __name__ == "__main__":
    main()

