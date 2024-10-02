#! /usr/bin/env python
#
#  Copyright 2020 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Philip G Brodrick, philip.brodrick@jpl.nasa.gov


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms
import matplotlib
import numpy as np
import skimage.transform
import argparse
from scipy.signal import savgol_filter
from scipy import interpolate
from osgeo import gdal
from scipy.ndimage import gaussian_filter

import spectral.io.envi as envi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os

matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.handleheight'] = 1
#
plt.switch_backend("Agg")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 18


def trim_image(in_img):
    img = in_img.copy()
    img[np.all(img == 0, axis=-1), :] = np.nan
    good_x = np.logical_not(np.all(np.isnan(img), axis=(0, 2)))
    img = img[:, good_x, :]
    good_y = np.logical_not(np.all(np.isnan(img), axis=(1, 2)))
    img = img[good_y, ...]
    img[np.all(np.isnan(img), axis=-1), :] = 0
    img[np.isfinite(img) == False] = 0
    return img


def find_closest(array, value):
    return np.argmin(np.abs(np.array(array) - value))


def scale(left, right, top, bottom, water_edges):
    # Scale right-left-tope-bottom together
    scale_array = []
    for im in [left, right, top, bottom]:
        tmp_im = im.copy()
        if len(tmp_im.shape) > 2:
            tmp_im = np.reshape((tmp_im.shape[0] * tmp_im.shape[1], tmp_im.shape[2]))
        for we in water_edges:
            tmp_im[:, we[0]:we[1]] = np.nan
        scale_array.append(tmp_im)
    scale_array = np.vstack(scale_array)
    offset = np.nanpercentile(scale_array, 2)
    scale = np.nanpercentile(scale_array - offset, 99)

    bottom = np.minimum(np.maximum((bottom - offset) / scale, np.zeros(bottom.shape) + 0.01),
                        np.ones(bottom.shape) - 0.01)
    top = np.minimum(np.maximum((top - offset) / scale, np.zeros(top.shape) + 0.01), np.ones(top.shape) - 0.01)
    left = np.minimum(np.maximum((left - offset) / scale, np.zeros(left.shape) + 0.01), np.ones(left.shape) - 0.01)
    right = np.minimum(np.maximum((right - offset) / scale, np.zeros(right.shape) + 0.01), np.ones(right.shape) - 0.01)

    return left, right, top, bottom


def main():
    parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
    parser.add_argument('data_file')
    parser.add_argument('methane_file',type=str)
    parser.add_argument('mineral_file', default=None, type=str)
    parser.add_argument('-wavelength_file', default=None, type=str)
    parser.add_argument('-rows', nargs=2, default=[-1, -1], type=int)
    parser.add_argument('-cols', nargs=2, default=[-1, -1], type=int)
    parser.add_argument('-ul_lr', nargs=4, default=[-1, -1, -1, -1], type=float)
    parser.add_argument('-smooth_spectra', default=0, type=int)
    parser.add_argument('-remove_water_features', default=0, type=int)
    parser.add_argument('-cube_type', default='radiance', choices=['radiance', 'reflectance'], type=str)
    parser.add_argument('-populate_defaults', default=1, type=int)
    parser.add_argument('-force_cube', default=0, type=int)
    parser.add_argument('-output_directory', default='figs', type=str)
    parser.add_argument('-dpi', default=600, type=int)
    parser.add_argument('-front_wavelengths', default=[647, 547, 457], type=int, nargs=3)
    parser.add_argument('-back_wavelengths', default=[647, 547, 457], type=int, nargs=3)
    args = parser.parse_args()

    if args.populate_defaults == 1:
        if args.cube_type == 'radiance':
            args.remove_water_features = 0
            args.smooth_spectra = 0
        elif args.cube_type == 'reflectance':
            args.remove_water_features = 1
            args.smooth_spectra = 1

    # use_rows = [13500, 17000]
    ds = envi.open(args.data_file + '.hdr')
    data_cube = ds.open_memmap(interleave='bip')[:1242,:,:].transpose((1,0,2))

    emit_ds = envi.open(args.mineral_file + '.hdr')
    minerals = emit_ds.open_memmap(interleave='bip')[:1242,:,:].copy().transpose((1,0,2))
    band_names = emit_ds.metadata['band names']


    wavelengths = None
    if 'wavelength' in ds.metadata.keys():
        wavelengths = np.array([float(x) for x in ds.metadata['wavelength']])

    if args.wavelength_file is not None:
        wavelengths = np.genfromtxt(args.wavelength_file)

    if wavelengths is None:
        raise AttributeError(
            'Data cube has no property "wavelengths" and wavelengths file is not specified.  Please provide one of the two')

    if args.rows[0] == -1:
        args.rows[0] = 0
    if args.rows[1] == -1:
        args.rows[1] = data_cube.shape[0]

    if args.cols[0] == -1:
        args.cols[0] = 0
    if args.cols[1] == -1:
        args.cols[1] = data_cube.shape[1]

    if args.ul_lr[0] != -1:
        trans = gdal.Open(args.data_file).GetGeoTransform()
        args.cols[0] = int(round((args.ul_lr[0] - trans[0]) / trans[1]))
        args.cols[1] = int(round((args.ul_lr[2] - args.ul_lr[0]) / trans[1]))

        args.rows[0] = int(round((trans[3] - args.ul_lr[1]) / trans[5]))
        args.rows[1] = int(round((args.ul_lr[1] - args.ul_lr[3]) / trans[5]))

    data_cube = data_cube[args.rows[0]:args.rows[1], args.cols[0]:args.cols[1], :]
    minerals = minerals[args.rows[0]:args.rows[1], args.cols[0]:args.cols[1], :]
    if args.force_cube == 1 and data_cube.shape[0] != data_cube.shape[1]:
        minsize = min(data_cube.shape[0], data_cube.shape[1])
        data_cube = data_cube[data_cube.shape[0] - minsize:, data_cube.shape[1] - minsize:, :]
        minerals = minerals[minerals.shape[0] - minsize:, minerals.shape[1] - minsize:, :]


    ch4 = envi.open(args.methane_file + '.hdr').open_memmap(interleave='bip').copy().squeeze()
    mask = gdal.Open(f'/beegfs/scratch/brodrick/methane/public_mmgis_masked/{os.path.basename(args.methane_file).split("_")[0]}_ch4_mask.tif').ReadAsArray()
    ch4[mask != 1] = np.nan
    ch4 = ch4[:1242,:].T

    V=ch4.copy()
    V[np.isnan(ch4)]=0
    VV=gaussian_filter(V,sigma=1)

    W=0*ch4.copy()+1
    W[np.isnan(ch4)]=0
    WW=gaussian_filter(W,sigma=1)

    ch4_smooth=VV/WW

    mask = np.isfinite(ch4)
    ch4_smooth = ch4_smooth / 1500
    ch4_overlay = matplotlib.colormaps['plasma'](ch4_smooth)
    print(np.max(ch4_overlay))
    #rgb = ch4_overlay[:,:,:3]
    



    #if args.force_cube == 1 and data_cube.shape[0] != data_cube.shape[-1]:
    #    new_wl = np.linspace(wavelengths[0], wavelengths[-1], num=data_cube.shape[0])
    #    orig_shape = data_cube.shape
    #    data_cube = np.reshape(data_cube, (data_cube.shape[0] * data_cube.shape[1], data_cube.shape[2]))

    #    finterp = interpolate.interp1d(wavelengths, data_cube)
    #    new_data_cube = finterp(new_wl)
    #    data_cube = new_data_cube.reshape((orig_shape[0], orig_shape[1], len(new_wl)))
    #    wavelengths = new_wl

    minerals[...,band_names.index('Hematite')] *= 5
    minerals[...,band_names.index('Goethite')] *= 5
    maxband = np.argmax(minerals, axis=-1)

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


    un_vals = np.unique(maxband)
    print(un_vals)
    band_names = np.array(band_names)
    mineral_img = np.zeros((minerals.shape[0], minerals.shape[1], 3), dtype=np.uint8)
    leg_handles = []
    for _v, val in enumerate(un_vals):
        print(f'{_v}/{len(un_vals) - 1})')
        leg_handles.append(Patch(facecolor=np.array(colorlist[band_names[val].lower()]) / 255., edgecolor='black', label=band_names[val].capitalize()))

        subset = np.logical_and(maxband == val, minerals[...,val] > 0)
        if np.sum(subset) == 0:
            continue
        minval = np.percentile(minerals[subset, val], 5)
        maxval = np.percentile(minerals[subset, val], 85)

        print(band_names[val], minval, maxval, np.sum(subset))
        loc_int = ((minerals[subset, val] - minval) / maxval)
        loc_int[loc_int > 1] = 1
        loc_int[loc_int < 0] = 0
        if np.sum(subset) < 30000:
            loc_int = 1

        loc_int = np.power(loc_int, 0.8)
        for _c in range(3):
            mineral_img[subset, _c] = colorlist[band_names[val].lower()][_c] * loc_int
            #mineral_img[subset, _c] = colorlist[band_names[val].lower()][_c]

    del minerals
    #tp = np.zeros((mineral_img.shape[0], int(mineral_img.shape[1]*1.3), mineral_img.shape[2]))
    #tp[:,:mineral_img.shape[1],:] = mineral_img
    #plt.imshow(tp.astype(float)/255.)
    #plt.axis('off')
    #plt.legend(handles=leg_handles, loc='lower right', fontsize=6)
    #plt.savefig('tmp.png',dpi=400,bbox_inches='tight')
    #quit()

    cm_list = plt.get_cmap('jet')(np.linspace(0, 1, 256))[:, :3].tolist()
    cm = LinearSegmentedColormap.from_list('cm_list', cm_list, N=len(cm_list))


    ###### Right
    right = data_cube[:, -1, :].copy().astype(float)
    right[right < 0] = 0

    ###### Left
    left = data_cube[:, 0, :].copy().astype(float)
    left = left[:, ::-1]
    left[left < 0] = 0

    # Top
    top = data_cube[0, ...].copy().astype(float)
    top = top[:, ::-1]
    top[top < 0] = 0

    # Bottom
    bottom = data_cube[-1, ...].copy().astype(float)
    bottom[bottom < 0] = 0

    water_edges = [[find_closest(wavelengths, 1268), find_closest(wavelengths, 1473)],
                   [find_closest(wavelengths, 1794), find_closest(wavelengths, 1944)],
                   ]

    left, right, top, bottom = scale(left, right, top, bottom, water_edges)

    if args.cube_type == 'radiance':
        right[right != 0] = np.log(right[right != 0])
        left[left != 0] = np.log(left[left != 0])
        top[top != 0] = np.log(top[top != 0])
        bottom[bottom != 0] = np.log(bottom[bottom != 0])

    left, right, top, bottom = scale(left, right, top, bottom, water_edges)

    if args.force_cube == 1 and right.shape[0] != right.shape[1]:
        new_wl = np.linspace(wavelengths[0], wavelengths[-1], num=data_cube.shape[0])

        finterp = interpolate.interp1d(wavelengths, bottom)
        bottom = finterp(new_wl)

        finterp = interpolate.interp1d(wavelengths, top)
        top = finterp(new_wl)

        finterp = interpolate.interp1d(wavelengths, right)
        right = finterp(new_wl)

        finterp = interpolate.interp1d(wavelengths, left)
        left = finterp(new_wl)
        cm = LinearSegmentedColormap.from_list('cm_list', cm_list, N=len(new_wl))

    bottom = cm(bottom)[..., :3]
    top = cm(top)[..., :3]
    left = cm(left)[..., :3]
    right = cm(right)[..., :3]

    # Transposes as necessary
    top = top.transpose([1, 0, 2])
    bottom = bottom.transpose([1, 0, 2])

    # Back
    #back_bands = np.array([find_closest(wavelengths, x) for x in args.back_wavelengths])
    #back = data_cube[:, :, back_bands].copy().astype(float)
    #back[back == -9999] = np.nan
    #back -= np.nanpercentile(back, 2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    #back /= np.nanpercentile(back, 98, axis=(0, 1))[np.newaxis, np.newaxis, :]
    #back = back[:, ::-1, :]
    back = mineral_img[:,::-1,:].astype(float) / 255.
    #back = mineral_img.astype(float) / 255.

    # Front
    front_bands = np.array([find_closest(wavelengths, x) for x in args.front_wavelengths])
    front = data_cube[:, :, front_bands].copy().astype(float)
    front[front == -9999] = np.nan
    front -= np.nanpercentile(front, 2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    front /= np.nanpercentile(front, 98, axis=(0, 1))[np.newaxis, np.newaxis, :]
    back = front.copy()[:,::-1,:] 

    #front[...] = 0

    #alpha = 0.8
    #alpha_0 = alpha + 1 * (1-alpha)
    #front[...] = (alpha * 1. + front * (1-alpha))/alpha_0

    front[...] = np.sum(front,axis=-1)[:,:,np.newaxis]/4.

    front[mask,:] = ch4_overlay[mask,:3]

    # Set up output image
    output_image = np.zeros((top.shape[0] + front.shape[0] + bottom.shape[0],
                             front.shape[1] + back.shape[1] + left.shape[1] + right.shape[1], 3))
    output_image[...] = np.nan

    # populate output image
    output_image[top.shape[0]:-bottom.shape[0], :left.shape[1], :] = left
    output_image[top.shape[0]:-bottom.shape[0], left.shape[1]:left.shape[1] + front.shape[1], :] = front
    output_image[top.shape[0]:-bottom.shape[0],
    left.shape[1] + front.shape[1]:left.shape[1] + front.shape[1] + right.shape[1]:, :] = right
    output_image[top.shape[0]:-bottom.shape[0], -back.shape[1]:, :] = back
    output_image[:top.shape[0], left.shape[1]:left.shape[1] + front.shape[1], :] = top
    output_image[-bottom.shape[0]:, left.shape[1]:left.shape[1] + front.shape[1], :] = bottom

    output_image[np.isnan(output_image)] = 255
    plt.imshow(output_image)
    plt.axis('off')
    #plt.legend(handles=leg_handles, loc='lower right', fontsize=6)

    if os.path.isdir(args.output_directory) is False:
        os.mkdir(args.output_directory)
    plt.savefig('{}/buildable_data_cube_{}.png'.format(args.output_directory, os.path.basename(args.data_file)),
                dpi=args.dpi, bbox_inches='tight')


if __name__ == '__main__':
    main()
