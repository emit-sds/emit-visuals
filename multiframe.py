


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
    parser.add_argument('rfl_file', type=str)
    parser.add_argument('sa_file', type=str)
    parser.add_argument('--sma_file', type=str, default=None)
    parser.add_argument('--outbase', type=str, default='figs')
    parser.add_argument('--mask_file', type=str, default=None, metavar='l2a mask file')
    parser.add_argument('--coord_circ', nargs=2, type=int, default=None)
    parser.add_argument('--landcover_file', type=str, default='/beegfs/store/brodrick/emit/landcover/complete_landcover.vrt', 
                        metavar='landcover classification file')
    args = parser.parse_args()

    rfl_ds = envi.open(envi_header(args.rfl_file))
    if envi.open(envi_header(args.rfl_file)).open_memmap(interleave='bip').shape[2] > 3:
        wl = np.array([float(x) for x in envi.open(args.rfl_file + '.hdr').metadata['wavelength']])
        rgb_inds = get_wl_inds(wl)
        wl_at_rgb = wl[rgb_inds]
    else:
        rgb_inds = np.array([0,1,2])
    rgb = envi.open(envi_header(args.rfl_file)).open_memmap(interleave='bip')[...,rgb_inds].copy()
    mask = np.any(rgb < 0, axis=(1,2))
    rgb[mask,...] = np.nan
    #rgb[np.any(rgb) > 1] = 0.999
    #rgb[np.any(rgb) < 0] = 0
    #rgb -= np.nanmin(rgb ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    #rgb /= np.nanmax(rgb ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    #rgb *= 255
    #rgb[rgb > 255] = 255
    #rgb[np.isnan(rgb)] = 0
    #rgb = rgb.astype('uint8')
    #for _b in range(rgb.shape[-1]):
    #    rgb[...,_b] = cv.equalizeHist(rgb[...,_b])

    rgb -= np.nanpercentile(rgb ,2, axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    rgb /= np.nanpercentile(rgb ,99.5,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]


    sa_ds = envi.open(envi_header(args.sa_file))
    sa = envi.open(envi_header(args.sa_file)).open_memmap(interleave='bip')
    band_names = sa_ds.metadata['band names']

    if args.sma_file is not None:
        sma = envi.open(envi_header(args.sma_file)).open_memmap(interleave='bip')[...,[0,1,2]]

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
    colorlist={\
    'calcite' :         [150, 255, 1],  # chartreuse
    'dolomite' :        [60, 200, 50],  # classic green
    'chlorite' :        [115, 157, 98],  # very light green - pale green
    'vermiculite' :     [189, 120, 66],  # brown
    'illite+muscovite': [7, 250, 163],  #lighter blue / green
    'montmorillonite' : [5, 173, 250],  #light blue 
    'kaolinite' :       [1, 10, 255],  #darker blue - true blue
    'gypsum' :          [150,  1,255],  # vibrant magenta - sulfate...keep away
    'goethite':         [255, 215,  1], # yellow/orange
    'hematite':         [255, 1, 1]     # red
    }
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
    iron_oxide_idx = np.array([x in ['goethite','hematite'] for x in band_names])
    


    print('read mask')
    total_mask = np.zeros((sa.shape[0],sa.shape[1]))
    if args.mask_file is not None:
        print(args.mask_file)
        cloud = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,0].copy()
        cirrus = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,1].copy()
        cloud_buffer = envi.open(envi_header(args.mask_file), gdal.GA_ReadOnly).open_memmap(interleave='bip')[:,:,4].copy()
        


    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    sa_sum = np.sum(sa,axis=-1)

    maxband = np.argmax(sa, axis=-1)
    un_vals = np.unique(maxband)
    output_all = np.zeros((sa.shape[0], sa.shape[1],3))
    leg_handles = []
    for _v, val in enumerate(un_vals):
        print(f'{_v}/{len(un_vals)-1})')
        subset = np.logical_and(maxband == val, sa[..., val] > 0, np.logical_not(total_mask))
        minname = band_names[val]
        
        if np.sum(subset) > 1:

            sa_subset = sa[subset, val]
            #sa_subset -=  np.min(sa_subset)
            #sa_subset /=  np.max(sa_subset)
            #sa_subset = (255*(sa_subset - 0) / np.max(sa_subset)).astype('uint8')
            #sa_subset = cv.equalizeHist(sa_subset)/255.
            if np.all(sa_subset == np.min(sa_subset)):
                sa_subset[:] = 1
            else:
                sa_subset -= np.min(sa_subset)
                sa_subset /= np.max(sa_subset)
                sa_subset = (255*sa_subset).astype('uint8')
                sa_subset = cv.equalizeHist(sa_subset).astype(np.float32)/255.

            output_all[subset,:] = np.squeeze(sa_subset)[:,np.newaxis] * np.squeeze(np.array(colorlist[minname]))[np.newaxis,:]
            leg_handles.append(Patch(facecolor=np.array(colorlist[minname])/255., edgecolor='black', label=minname.capitalize().replace('e+m','e & M')))
    

    maxband = np.argmax(sa[...,iron_oxide_idx], axis=-1)
    un_vals = np.unique(maxband)
    output_io = np.zeros((sa.shape[0], sa.shape[1],3))
    leg_handles_io = []
    for _v, val in enumerate(un_vals):
        print(f'{_v}/{len(un_vals)-1})')
        subset = np.logical_and(maxband == val, sa[...,iron_oxide_idx][...,val] > 0, np.logical_not(total_mask))
        minname = np.array(band_names)[iron_oxide_idx][val]
        
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
            leg_handles_io.append(Patch(facecolor=np.array(colorlist[minname])/255., edgecolor='black', label=minname.capitalize().replace('e+m','e & M')))
    
    maxband = np.argmax(sa[...,np.logical_not(iron_oxide_idx)], axis=-1)
    un_vals = np.unique(maxband)
    print(un_vals)
    output_noio = np.zeros((sa.shape[0], sa.shape[1],3))
    leg_handles_noio = []
    for _v, val in enumerate(un_vals):
        print(f'{np.array(band_names)[np.logical_not(iron_oxide_idx)][val]} - {_v}/{len(un_vals)-1})')
        subset = np.logical_and(maxband == val, sa[...,np.logical_not(iron_oxide_idx)][..., val] > 0, np.logical_not(total_mask))
        minname = np.array(band_names)[np.logical_not(iron_oxide_idx)][val]
        
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
            leg_handles_noio.append(Patch(facecolor=np.array(colorlist[minname])/255., edgecolor='black', label=minname.capitalize().replace('e+m','e & M')))
    




    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(ncols=3, nrows=2, wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[0,0])
    im = plt.imshow(rgb)
    if args.coord_circ is not None:
        plt.scatter(args.coord_circ[1],args.coord_circ[0],edgecolors='grey',facecolors='none', marker='o')
    plt.axis('off')
    plt.title(f'RGB [{round(wl_at_rgb[0])} nm, {round(wl_at_rgb[1])} nm, {round(wl_at_rgb[2])} nm]')

    ax = fig.add_subplot(gs[0,1])
    im = plt.imshow(output_io/255.)
    if args.coord_circ is not None:
        plt.scatter(args.coord_circ[1],args.coord_circ[0],edgecolors='grey',facecolors='none', marker='o')
    plt.axis('off')
    plt.title('Dominant Mineral Abundances - Iron Oxides')
    #plt.legend(handles=leg_handles_io, fontsize=8, loc='lower right')

    ax = fig.add_subplot(gs[0,2])
    im = plt.imshow(output_noio/255.)
    if args.coord_circ is not None:
        plt.scatter(args.coord_circ[1],args.coord_circ[0],edgecolors='grey',facecolors='none', marker='o')
    plt.axis('off')
    plt.title(f'Dominant Mineral Abundances - 2$\mu$m')
    #plt.legend(handles=leg_handles_noio, fontsize=8, loc='lower right')


    plt.savefig(os.path.join(args.outbase, os.path.splitext(os.path.basename(args.rfl_file))[0] + '_3panel.png'), bbox_inches='tight', dpi=200)



    if args.sma_file is None:
        exit()



    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[0,0])
    im = plt.imshow(rgb)
    plt.axis('off')
    plt.title(f'RGB [{round(wl_at_rgb[0])} nm, {round(wl_at_rgb[1])} nm, {round(wl_at_rgb[2])} nm]')

    ax = fig.add_subplot(gs[0,1])
    im = plt.imshow(output_io/255.)
    plt.axis('off')
    plt.title('Mineral Abundances - Iron Oxides')
    plt.legend(handles=leg_handles_io, fontsize=8, loc='lower right')

    ax = fig.add_subplot(gs[1,1])
    im = plt.imshow(output_noio/255.)
    plt.axis('off')
    plt.title(f'Mineral Abundances - 2 $\mu$m')
    plt.legend(handles=leg_handles_noio, fontsize=8, loc='lower right')


    ax = fig.add_subplot(gs[1,0])
    im = plt.imshow(sma, vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Fractional Cover')
    plt.legend(handles=[Patch(facecolor=np.array([0, 255, 0])/255., edgecolor='black', label='PV'), Patch(facecolor=np.array([255, 0, 0])/255., edgecolor='black', label='NPV'), Patch(facecolor=np.array([0, 0, 255])/255., edgecolor='black', label='Soil')], fontsize=8, loc='lower right')


    plt.savefig(os.path.join(args.outbase, os.path.splitext(os.path.basename(args.rfl_file))[0] + '_4panel.png'), bbox_inches='tight', dpi=200)

    exit()


    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.1, hspace=0.1)
    ax = fig.add_subplot(gs[0,0])
    im = plt.imshow(rgb)
    plt.axis('off')
    plt.title(f'RGB [{round(wl_at_rgb[0])} nm, {round(wl_at_rgb[1])} nm, {round(wl_at_rgb[2])} nm]')


    ax = fig.add_subplot(gs[0,1])
    plt.imshow(output_all/255.)
    plt.axis('off')
    plt.legend(handles=leg_handles, fontsize=8, loc='lower right')
    plt.title(f'Surface Minerology')
    plt.savefig(os.path.join(args.outbase, os.path.splitext(os.path.basename(args.rfl_file))[0] + '_1panel.png'), bbox_inches='tight', dpi=200)







if __name__ == "__main__":
    main()
