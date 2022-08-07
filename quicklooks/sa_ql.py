


from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess



def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte)
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_img.shape[2]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_img[...,_b])
    del outDataset




def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='aggregated abundance file')
    parser.add_argument('output_file', type=str, metavar='output file to write')
    parser.add_argument('--mask_file', type=str, default=None, metavar='l2a mask file')
    parser.add_argument('--independent', action='store_true')
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    args = parser.parse_args()


    source_ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    trans = source_ds.GetGeoTransform()

    cover_ds = gdal.Warp('',args.landcover_file,format='MEM',resampleAlg='mode',
                         outputBounds=[trans[0],trans[3]+trans[5]*source_ds.RasterYSize, trans[0] + trans[1]*source_ds.RasterXSize, trans[3]], xRes=trans[1], yRes=trans[5])

    cover = cover_ds.ReadAsArray()
    
    sa = source_ds.ReadAsArray() 

    if args.mask_file is not None:
        cloud = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(1).ReadAsArray()
        cirrus = gdal.Open(args.mask_file, gdal.GA_ReadOnly).GetRasterBand(2).ReadAsArray()
 
    
    #band_names = ['calcite',  'chlorite' ,'dolomite', 'goethite' , 'gypsum' , 'hematite', 'illite+muscovite', 'kaolinite' , 'montmorillonite', 'vermiculite']
    band_names = \
    [\
    'kaolinite',
    'calcite',
    'dolomite',
    'hematite',
    'goethite',
    'gypsum',
    'chlorite',
    'illite+muscovite',
    'montmorillonite']


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

            output_img = np.zeros((sa.shape[1], sa.shape[2], 3))
            subset = sa[band,...] > 0

            minval = np.percentile(sa[band, subset],0)
            maxval = np.percentile(sa[band, subset],85)
            output_img[subset,:] = np.array(colorlist[band_names[band]])[np.newaxis,:]
            output_img[subset,:] *= ((sa[band, subset] - minval) / maxval)[:,np.newaxis]
            
            output_img[np.logical_and(cover==50, subset),:] = np.array([0.,0.,0.])[np.newaxis,:]
            output_img[np.logical_and(cover==80, subset),:] = np.array([0.,0.,0.])[np.newaxis,:]
            output_img[:,np.logical_and(cover==50, subset)] = np.array([180.,180.,180.])[:,np.newaxis]
            output_img[:,np.logical_and(cover==80, subset)] = np.array([100.,100.,100.])[:,np.newaxis]
            output_img[:,np.logical_and(cloud==1, subset)]  = np.array([255., 255., 255.])[:,np.newaxis]
            output_img[:,np.logical_and(cirrus==1, subset)] = np.array([200., 200., 200.])[:,np.newaxis]

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
        
        output_img = np.zeros((sa.shape[1], sa.shape[2], 3))
        leg_handles = []
        for _v, val in enumerate(un_vals):
            print(f'{_v}/{len(un_vals)-1})')
            subset = np.logical_and(maxband == val, sa[val,...] > 0)
            output_img[subset,:] = np.array(colorlist[band_names[val]])[np.newaxis,:]

            minval = np.percentile(sa[val, sa[val,...] > 0],0)
            maxval = np.percentile(sa[val, sa[val,...] > 0],85)
            output_img[subset,:] *= ((sa[val, subset] - minval) / maxval)[:,np.newaxis]
        
        sa_sum = np.sum(sa,axis=0)
        del sa

        output_img[:,np.logical_and(cover==50, sa_sum != 0)] = np.array([180.,180.,180.])[:,np.newaxis]
        output_img[:,np.logical_and(cover==80, sa_sum != 0)] = np.array([100.,100.,100.])[:,np.newaxis]
        output_img[:,np.logical_and(cloud==1, sa_sum != 0)] = np.array([255., 255., 255.])[:,np.newaxis]
        output_img[:,np.logical_and(cirrus==1, sa_sum != 0)] = np.array([200., 200., 200.])[:,np.newaxis]

        output_img = np.round(output_img)
        output_img[output_img > 255] = 255
        output_img[output_img < 0] = 0
        output_img = output_img.astype(np.uint8)
 
        write_output_file(source_ds, output_img, args.output_file)
        #cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="gdal2tiles.py {args.output_file} {args.output_file}_tiled -z 7-13 --srcnodata 0 --processes=40"'
        #subprocess.call(cmd_str,shell=True)



if __name__ == "__main__":
    main()
