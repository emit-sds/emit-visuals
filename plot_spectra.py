import numpy as np
import scipy
from spectral.io import envi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os
import matplotlib


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

def wl2band(w: float, wl: np.array) -> int:
  return np.argmin(abs(wl-w))

def smooth(x:np.array, window_length:int = 3) -> np.array:
  """Moving average smoother
  Args:
      x (np.array): Input spectrum
      window_length (int, optional): Window size for smoothing. Defaults to 3.

  Returns:
      np.array: smoothed spectra
  """
  q=np.r_[x[window_length-1:0:-1],x,x[-1:-window_length:-1]]
  w=np.ones(window_length,'d')/float(window_length)
  y=np.convolve(w,q,mode='valid')
  y= y[int(window_length/2):-int(window_length/2)]
  return y



def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='reflectance data')
    parser.add_argument('--rad_file', type=str, default=None)
    parser.add_argument('--output_directory',default='figs', type=str)
    parser.add_argument('--style',default='random', type=str)
    parser.add_argument('--num_spectra',default=10, type=int)
    parser.add_argument('--seed',default=13, type=int)
    parser.add_argument('--cherrypick',action='store_true')
    parser.add_argument('--radonly',action='store_true')
    args = parser.parse_args()


    np.random.seed(args.seed)
    rfl_ds = envi.open(envi_header(args.input_file))
    rfl = rfl_ds.open_memmap(interleave='bip')
    rfl = rfl.reshape((rfl.shape[0]*rfl.shape[1],rfl.shape[2]))

    if args.rad_file is not None:
        rad_ds = envi.open(envi_header(args.rad_file))
        rad = rad_ds.open_memmap(interleave='bip')
        rad = rad.reshape((rad.shape[0]*rad.shape[1],rad.shape[2]))

    wl = np.array([float(x) for x in rfl_ds.metadata['wavelength']])

    #s940,e940 = wl2band(910,wl), wl2band(990,wl)
    #s1140,e1140 = wl2band(1090,wl), wl2band(1180,wl) 
    #srefA,erefA = wl2band(1010,wl), wl2band(1080,wl)
    #srefB,erefB = wl2band(780,wl), wl2band(900,wl)

    #errsA = spectrum[s940:e940] - ctm[s940:e940]
    #errsB = spectrum[s1140:e1140] - ctm[s1140:e1140]
    #referenceA = spectrum[srefA:erefA] - ctm[srefA:erefA]
    #referenceB = spectrum[srefB:erefB] - ctm[srefB:erefB]

    ## calcualte the root mean squared error of each interval
    #errsA = np.sqrt(np.mean(pow(errsA,2)))
    #errsB = np.sqrt(np.mean(pow(errsB,2)))
    #referenceA = np.sqrt(np.mean(pow(referenceA,2)))
    #referenceB = np.sqrt(np.mean(pow(referenceB,2)))

    ## We use the better of two reference regions and two 
    ## water regions for robustness
    #errs = min(errsA,errsB)
    #reference = min(referenceA,referenceB)
    #excess_error = errs/reference

    font = {'size'   : 18}

    matplotlib.rc('font', **font)
    
    fig = plt.figure(figsize = (10,14))
    spec = gridspec.GridSpec(ncols=1, nrows=2,left=0.01, wspace=0.01)
    ax = fig.add_subplot(spec[0,0])
    wl_nan = wl.copy()
    wl_nan[np.logical_and(wl > 1325, wl < 1490)] = np.nan
    wl_nan[np.logical_and(wl > 1770, wl < 1965)] = np.nan
    #wl_nan[wl > 2470] = np.nan

    if args.style == 'random':
        coord = np.random.permutation(rfl.shape[0])[:args.num_spectra]

    if args.radonly:
        plt.plot(wl,rfl[coord,:].T)
        plt.ylabel('Radiance [$\mu$W sr$^{-1}$ cm$^{-2}$ nm$^{-1}$]')
    else:
        plt.plot(wl_nan,rfl[coord,:].T)
        plt.ylabel('Reflectance')
    plt.xlabel('Wavelength [nm]')
    plt.grid(True)


    if args.rad_file is not None:
        ax = fig.add_subplot(spec[1,0])
        plt.plot(wl,rad[coord,:].T)
        plt.ylabel('Radiance [$\mu$W sr$^{-1}$ cm$^{-2}$ nm$^{-1}$]')
        plt.xlabel('Wavelength [nm]')
            
        plt.grid(True)


    plt.savefig('{}/spectra_{}.png'.format(args.output_directory,os.path.basename(args.input_file)),dpi=200,bbox_inches='tight')


 
if __name__ == "__main__":
    main()
