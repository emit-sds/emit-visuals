


from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import os
import cv2 as cv



def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,creationOptions = ['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_img.shape[0]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_img[_b,...])
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
    parser.add_argument('--independent', action='store_true')
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
    
    print('read sa')
    sa = source_ds.ReadAsArray() 

    print('read mask')
    if args.mask_file is not None:
        #cloud = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
        #cirrus = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(2).ReadAsArray()
        cloud = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,0].copy()
        cirrus = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,1].copy()
        cloud_buffer = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,4].copy()
 
    
    sa[:,cloud == -9999] = 0
    band_names = ['calcite',  'chlorite' ,'dolomite', 'goethite' , 'gypsum' , 'hematite', 'illite+muscovite', 'kaolinite' , 'montmorillonite', 'vermiculite']
    #band_names = \
    #[\
    #'kaolinite',
    #'calcite',
    #'dolomite',
    #'hematite',
    #'goethite',
    #'gypsum',
    #'chlorite',
    #'illite+muscovite',
    #'montmorillonite']


    colorlist={\
    'kaolinite':       [255, 225, 25], #yellow
    'calcite':         [1, 130, 200], #blue
    'dolomite':        [245, 130, 48], #orange
    'hematite':        [145, 30, 180], #purple
    'goethite':        [240, 50, 230], #magenta
    'gypsum':          [250, 190, 212], #pink
    'chlorite':        [170, 110, 40], #brown
    'illite':          [128, 128, 1], #olive
    'montmorillonite': [128, 128, 128] #grey
    }

    colorlist={\
    'calcite' :         [40, 130, 10], # green
    'chlorite' :        [235, 20,  1], #pink
    'dolomite' :        [195,25, 235], #purple-ish
    'gypsum' :          [240,  1,255], #magenta (
    'illite+muscovite': [220,110,  1], #dark orange (
    'kaolinite' :       [30, 95, 250], #darker blue (
    'montmorillonite' : [100,190,145], #light blue-green (
    'vermiculite' :     [155,220, 45],  # chartreuse (
    'goethite':         [255, 1,  1], 
    'hematite':         [255, 255, 25]
    }

    if args.independent:

        for band in range(len(band_names)):
            print(f'{band}/{len(band_names)-1})')

            output_img = np.zeros((3, sa.shape[1], sa.shape[2]))
            subset = sa[band,...] > 0

            minval = np.percentile(sa[band, subset],0)
            maxval = np.percentile(sa[band, subset],85)
            output_img[:, subset] = np.array(colorlist[band_names[band]])[:,np.newaxis]
            output_img[:, subset] *= ((sa[band, subset] - minval) / maxval)[np.newaxis,:]
            
            output_img[:,np.logical_and(cover==50, subset)] = np.array([180.,180.,180.])[:,np.newaxis]
            output_img[:,np.logical_and(cover==80, subset)] = np.array([100.,100.,100.])[:,np.newaxis]
            output_img[:,np.logical_and(cirrus==1, subset)] = np.array([200., 200., 200.])[:,np.newaxis]
            output_img[:,np.logical_and(cloud==1, subset)]  = np.array([255., 255., 255.])[:,np.newaxis]

            output_img = np.round(output_img)
            output_img[output_img > 255] = 255
            output_img[output_img < 0] = 0
            output_img = output_img.astype(np.uint8)
 
            write_output_file(source_ds, output_img, f'{args.output_file}_{band_names[band]}')

        #for band in range(len(band_names)):
        #    cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="gdal2tiles.py {args.output_file}_{band_names[band]} {args.output_file}_{band_names[band]}_tiled -z 7-13 --srcnodata 0 --processes=40"'
        #    subprocess.call(cmd_str,shell=True)


    else:
    
        maxband = np.argmax(sa, axis=0)
        un_vals = np.unique(maxband)

        sa_sum = np.sum(sa,axis=0)
        total_mask = np.zeros((sa.shape[1],sa.shape[2])).astype(bool)
        total_mask[cover==50] = True
        total_mask[cover==80] = True
        total_mask[cirrus==1] = True
        total_mask[cloud==1] = True
        total_mask[cloud_buffer==1] = True
        
        output_img = np.zeros((3,sa.shape[1], sa.shape[2]))
        leg_handles = []
        for _v, val in enumerate(un_vals):
            print(f'{_v}/{len(un_vals)-1})')
            subset = np.logical_and(maxband == val, sa[val,...] > 0, np.logical_not(total_mask))
            
            if np.sum(subset) > 0:

                sa_subset = sa[val, subset]
                sa_subset = ((sa_subset - np.min(sa_subset)) / np.percentile(sa_subset,99))
                #sa_subset = (255*(sa_subset - 0) / np.max(sa_subset)).astype('uint8')
                #sa_subset = cv.equalizeHist(sa_subset)/255.
                output_img[:,subset] = np.squeeze(sa_subset) * np.array(colorlist[band_names[val]])[:,np.newaxis]

                #output_img[:, subset] = np.array(colorlist[band_names[val]])[:,np.newaxis]

                #minval = np.min(sa[val, sa[val,...] > 0])
                #maxval = np.max(sa[val, sa[val,...] > 0])
                #output_img[:, subset] *= ((sa[val, subset] - minval) / maxval)[np.newaxis,:]
        
        del sa

        #output_img[:,np.logical_and(cover==50, sa_sum != 0)] = np.array([180.,180.,180.])[:,np.newaxis]
        #output_img[:,np.logical_and(cover==80, sa_sum != 0)] = np.array([100.,100.,100.])[:,np.newaxis]
        #output_img[:,np.logical_and(cirrus==1, sa_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
        #output_img[:,np.logical_and(cloud==1, sa_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
        #output_img[:,np.logical_and(cloud_buffer==1, sa_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
        #output_img[:,total_mask] = np.array([255.,255.,255.])[:,np.newaxis]

        output_img = np.round(output_img)
        output_img[output_img > 255] = 255
        output_img[output_img < 0] = 0
        output_img = output_img.astype(np.uint8)
 
        write_output_file(source_ds, output_img, args.output_file)
        #cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="gdal2tiles.py {args.output_file} {args.output_file}_tiled -z 7-13 --srcnodata 0 --processes=40"'
        #subprocess.call(cmd_str,shell=True)



if __name__ == "__main__":
    main()
