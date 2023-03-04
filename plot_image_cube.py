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
import cv2 as cv


import spectral.io.envi as envi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os

plt.switch_backend("Agg")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 18

def trim_image(in_img):
    img = in_img.copy()
    img[np.all(img == 0,axis=-1),:] = np.nan
    good_x = np.logical_not(np.all(np.isnan(img),axis=(0,2)))
    img = img[:,good_x,:]
    good_y = np.logical_not(np.all(np.isnan(img),axis=(1,2)))
    img = img[good_y,...]
    img[np.all(np.isnan(img),axis=-1),:] = 0
    img[np.isfinite(img) == False] = 0
    return img

def find_closest(array, value):
    return np.argmin(np.abs(np.array(array) - value))



def main():
    parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
    parser.add_argument('data_file')
    parser.add_argument('-wavelength_file',default=None, type=str)
    parser.add_argument('-rows',nargs=2, default=[-1,-1], type=int)
    parser.add_argument('-cols',nargs=2, default=[-1,-1], type=int)
    parser.add_argument('-smooth_spectra',default=0, type=int)
    parser.add_argument('-remove_water_features',default=0, type=int)
    parser.add_argument('-cube_type',default='radiance', choices=['radiance','reflectance'],type=str)
    parser.add_argument('-populate_defaults',default=1, type=int)
    parser.add_argument('-extraction_points',default=0, type=int)
    parser.add_argument('-output_directory',default='figs', type=str)
    parser.add_argument('-dpi', default=300, type=int)
    parser.add_argument('-front_wavelengths',default=[647,547,457], type=int, nargs=3)
    args = parser.parse_args()

    if args.populate_defaults == 1:
        if args.cube_type == 'radiance':
            args.remove_water_features = 0
            args.smooth_spectra = 0
            args.extraction_points = 1
        elif args.cube_type == 'reflectance':
            args.remove_water_features = 1
            args.smooth_spectra = 1
            args.extraction_points = 1


    if os.path.isfile(args.data_file + '.hdr'):
        ds = envi.open(args.data_file + '.hdr')
    else:
        ds = envi.open(args.data_file.replace('.img','.hdr'))
    data_cube = ds.open_memmap(interleave='bip')


    wavelengths = None
    if 'wavelength' in ds.metadata.keys():
        wavelengths = np.array([float(x) for x in ds.metadata['wavelength']])

    #print(data_cube[..., 1:-1:4].shape, data_cube[..., ::4].shape)
    #data_cube = (data_cube[..., :-1:4] +  data_cube[..., 1:-1:4] + data_cube[..., 2::4] + data_cube[...,3::4]) /4.
    #print(data_cube.shape)
    #wavelengths = (wavelengths[::4]+ wavelengths[1::4] + wavelengths[2::4] + wavelengths[3::4])/4.
    #from scipy import interpolate
    #finterp = interpolate.interp1d(wavelengths, data_cube, fill_value="extrapolate")
    #wavelengths = wavelengths[::4]
    #data_cube = data_cube[...,::6]
    #wavelengths = wavelengths[::6]


    if args.wavelength_file is not None:
        wavelengths = np.genfromtxt(args.wavelength_file)

    if wavelengths is None:
        raise AttributeError('Data cube has no property "wavelengths" and wavelengths file is not specified.  Please provide one of the two')

    if args.rows[0] == -1:
        args.rows[0] = 0
    if args.rows[1] == -1:
        args.rows[1] = data_cube.shape[0]

    if args.cols[0] == -1:
        args.cols[0] = 0
    if args.cols[1] == -1:
        args.cols[1] = data_cube.shape[1]

    data_cube = data_cube[args.rows[0]:args.rows[1],args.cols[0]:args.cols[1],:]
    #data_cube = data_cube[:,::-1,:]
    #data_cube = data_cube[::-1,:,:]
    #data_cube = np.transpose(data_cube.copy(),(1,0,2))

    water_edges = [[find_closest(wavelengths, 1268), find_closest(wavelengths, 1473)],
                   [find_closest(wavelengths, 1794), find_closest(wavelengths, 1944)],
                   ]

    cm_list = plt.get_cmap('jet')(np.linspace(0,1,256))[:,:3].tolist()
    cm = LinearSegmentedColormap.from_list('cm_list',cm_list,N=len(cm_list))

    front_bands = np.array([find_closest(wavelengths,x) for x in args.front_wavelengths])
    rgb = data_cube[:,:,front_bands].copy().astype(float )

    #rgb /= np.sum(rgb,axis=2)[:,:,np.newaxis]
    #rgb -= np.percentile(rgb ,2 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    #rgb /= np.percentile(rgb ,70,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    
    #rgb[np.any(rgb) > 1] = 0.999
    #rgb[np.any(rgb) < 0] = 0
    rgb -= np.nanmin(rgb ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    rgb /= np.nanmax(rgb ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
    rgb *= 255
    rgb[rgb > 255] = 255
    rgb[np.isnan(rgb)] = 0
    rgb = rgb.astype('uint8')
    for _b in range(rgb.shape[-1]):
        rgb[...,_b] = cv.equalizeHist(rgb[...,_b])
    rgb = rgb.astype(np.float32)/255.



    ###### Get right data
    right = data_cube[:,-5,:].copy().astype(float )
    #right = (right[:,2::2] + right[:,1::2])/2.
    #wavelengths = (wavelengths[2::2] + wavelengths[1::2])/2.
    right[right < 0] = 0.001
    if args.cube_type == 'radiance':
        right[right!=0] = np.log(right[right!=0])

    # Top
    top = data_cube[0,...].copy().astype(float )
    #top = (top[:,2::2] + top[:,1::2])/2.
    top[top < 0] = 0
    if args.cube_type == 'radiance':
        top[top!=0] = np.log(top[top!=0])

    # Scale right-top together
    water_edges = [[find_closest(wavelengths, 1268), find_closest(wavelengths,1500)],
                   [find_closest(wavelengths, 1760), find_closest(wavelengths,2040)],
                  ]
    scale_array = []
    for im in [right, top]:
        tmp_im = im.copy()
        if len(tmp_im.shape) > 2:
            tmp_im = np.reshape((tmp_im.shape[0]*tmp_im.shape[1],tmp_im.shape[2]))
        for we in water_edges:
            tmp_im[:,we[0]:we[1]] = np.nan
        scale_array.append(tmp_im)
    scale_array = np.vstack(scale_array)
    offset = np.nanpercentile(scale_array,3)
    scale = np.nanpercentile(scale_array - offset,99)

    if args.remove_water_features:
        for we in water_edges:
            top[..., we[0]:we[1]] = 0
            right[..., we[0]:we[1]] = 0

    # Color transform and skew right
    right = (right-offset)/scale
    right[right < 0] = 0
    right = cm(right)[...,:3]
    

    skew_rate = 0.9
    squeeze_rate = 1
    right_x_offset = right.shape[1]*2
    right_y_offset = right.shape[1]*2

    affine_tf = skimage.transform.AffineTransform(shear=-skew_rate,
                                                translation=[0,-right_y_offset], 
                                                rotation=skew_rate)
    right = skimage.transform.warp(right, inverse_map=affine_tf, order=0, clip=True, 
                                   output_shape=(right.shape[0] + right_y_offset,right.shape[1]+right_x_offset,3))

    right = trim_image(right)
    right = skimage.transform.resize(right, (right.shape[0],int(right.shape[1]*squeeze_rate),3),order=0)
    right[np.all(right == 0,axis=-1),:] = np.nan


    # Color transform and skew top
    top = cm((top-offset)/scale)[...,:3]
    top = top.transpose([1,0,2])
    top = top[::-1,:,:]

    top_offset = top.shape[1]
    affine_tf = skimage.transform.AffineTransform(shear=-1.01, translation=[-2*top_offset,0])
    top = skimage.transform.warp(top, inverse_map=affine_tf, clip=False, 
                                   output_shape=(top_offset*4, top_offset*4,3))

    top = trim_image(top)

    locations = [[300, 800], [400,1200 ], [700,700] , [1150, 600]]
    front = rgb.copy()
    #if args.extraction_points:
    if False:
        colors=[[31, 119, 180], [255, 126, 14], [44, 160, 44],[214, 39, 40]]
        #for _i in range(1,5):
        for _i, loc in enumerate(locations):
            yc = int(front.shape[0]/5*_i)
            xc = int(front.shape[1]/2)
            yc = loc[0]
            xc = loc[1]
            #front[yc-30:yc+30,xc-30:xc+30,:] = np.array(colors[_i-1])/255.
            #front[yc-30:yc+30,xc-30:xc-25,:] = np.array(colors[_i])/255.
            #front[yc-30:yc+30,xc+25:xc+30,:] = np.array(colors[_i])/255.
            #front[yc-30:yc-25,xc-30:xc+30,:] = np.array(colors[_i])/255.
            #front[yc+25:yc+30,xc-30:xc+30,:] = np.array(colors[_i])/255.

            front[yc-30:yc+30,xc-30:xc-25,:] = np.array(colors[_i])/255.
            front[yc-30:yc+30,xc+25:xc+30,:] = np.array(colors[_i])/255.
            front[yc-30:yc-25,xc-30:xc+30,:] = np.array(colors[_i])/255.
            front[yc+25:yc+30,xc-30:xc+30,:] = np.array(colors[_i])/255.

    # Set up output image
    output_image = np.zeros((max(front.shape[0],right.shape[0]), front.shape[1]+right.shape[1], 3))
    output_image[...] = np.nan

    # size top to accomodate
    top = skimage.transform.resize(top, (int(output_image.shape[0] - front.shape[0]),int(output_image.shape[1]),3))
    top[np.all(top ==0,axis=-1),:] = np.nan
    #top[np.all(top <0.1,axis=-1),:] = np.nan


    output_image[output_image.shape[0]-front.shape[0]:,:front.shape[1],:] = front
    output_image[output_image.shape[0]-right.shape[0]:,front.shape[1]:front.shape[1]+right.shape[1],:] = right

    good_top = np.where(np.logical_and(np.all(np.isnan(top),axis=-1) == False, np.all(np.isnan(output_image[:top.shape[0],:top.shape[1],:]),axis=-1) ))
    output_image[good_top[0],good_top[1],:] = top[good_top[0],good_top[1],:]

    output_image[np.isnan(output_image)] = 1
    output_image[np.all(output_image == 0,axis=-1),:] = 1


    figsize = (25,10)
    fig = plt.figure(figsize=figsize)
    spec = gridspec.GridSpec(ncols=4, nrows=10,left=0.01, wspace=0.01)

    ax = fig.add_subplot(spec[:,0:2])

    # Apply transform to image data
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')
    plt.imshow(output_image)
    plt.axis('off')

    #if args.extraction_points:
    if False:
        ax = fig.add_subplot(spec[:,2:])

        #for _i in range(1,5):
            #spectrum = data_cube[int(front.shape[0]/5*_i),int(front.shape[1]/2),:].copy()
        for _i, loc in enumerate(locations):
            yc = loc[0]
            xc = loc[1]
            spectrum = data_cube[loc[0],loc[1],:].copy()
            #if args.smooth_spectra == 1:
            if False:
                spectrum[...,:water_edges[0][0]] = savgol_filter(spectrum[...,:water_edges[0][0]], window_length=9, polyorder=3)
                for _w in range(len(water_edges)-1):
                    spectrum[...,water_edges[_w][1]:water_edges[_w+1][0]] = savgol_filter(spectrum[...,water_edges[_w][1]:water_edges[_w+1][0]], window_length=9, polyorder=3)
                spectrum[...,water_edges[-1][1]:] = savgol_filter(spectrum[...,water_edges[-1][1]:], window_length=9, polyorder=3)

            if args.remove_water_features:
                for we in water_edges:
                    spectrum[...,we[0]:we[1]] = np.nan

            spectrum[...,-5:] = np.nan
            plt.plot(wavelengths,spectrum, c=np.array(colors[_i])/255.)
        if args.cube_type == 'reflectance':
            plt.ylabel('Reflectance')
        elif args.cube_type == 'radiance':
            plt.ylabel('Radiance [W sr$^{−1}$ m$^{-2}$ nm$^{−1}$]')

        plt.xlabel('Wavelength [nm]')
        plt.grid(True)

    if os.path.isdir(args.output_directory) is False:
        os.mkdir(args.output_directory)
    plt.savefig('{}/data_cube_{}.png'.format(args.output_directory,os.path.basename(args.data_file)),dpi=args.dpi,bbox_inches='tight')

if __name__ == '__main__':
    main()
