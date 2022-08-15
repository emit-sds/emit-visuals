

import os
import subprocess
import argparse
import numpy as np
import glob


def main():

    parser = argparse.ArgumentParser(description="Run visuals workflow")
    #parser.add_argument('input_file_list', type=str)
    parser.add_argument('type',type=str, choices=['sa','sma','rgb'])
    parser.add_argument('--sa_tile_out',default='tiled_visuals/sa_mosaic_RGB', type=str)
    parser.add_argument('--sma_tile_out',default='tiled_visuals/sma_mosaic_RGB', type=str)
    parser.add_argument('--rgb_tile_out',default='tiled_visuals/rgb_mosaic_RGB', type=str)
    args = parser.parse_args()


    #loclist = np.genfromtxt(args.input_file_list,dtype=str)
    loclist = glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_loc_*.img')

    run_file = 'already_run_locs.txt'
    if os.path.isfile(run_file):
        prerun_loc_list = np.genfromtxt(run_file,dtype=str).tolist()
    else:
        prerun_loc_list = []



    for loc in loclist:
      if loc not in prerun_loc_list:
    
        rdn = loc.replace('loc','rdn')
        mask = loc.replace('l1b','l2a').replace('loc','mask')
        sa = loc.replace('l1b','l2b').replace('loc','abun')
        sma = loc.replace('l1b','l3').replace('loc','cover')
        
        glt = f'glts/{os.path.basename(loc).replace("loc","glt")}'
        ortho_sa = f'ortho/sa/{os.path.basename(sa)}'
        ortho_sma = f'ortho/sma/{os.path.basename(sma)}'
        ortho_rgb = f'ortho/rgb/{os.path.basename(rdn)}'
        ortho_mask = f'ortho/rgb/{os.path.basename(mask)}'
        
        ortho_sa_ql = f'ortho/sa_ql/{os.path.basename(sa)}'
        ortho_sma_ql = f'ortho/sma_ql/{os.path.basename(sma)}'
        ortho_rgb_ql = f'ortho/rgb_ql/{os.path.basename(rdn)}'

        if args.type == 'sa':
            cmd_str= f'gdal2tiles.py {ortho_sa_ql} {args.sa_tile_out} -z 4-11 --srcnodata 0 --processes=40'
        elif args.type == 'sma':
            cmd_str= f'gdal2tiles.py {ortho_sma_ql} {args.sma_tile_out} -z 4-11 --srcnodata 0 --processes=40'
        elif args.type == 'rgb':
            cmd_str= f'gdal2tiles.py {ortho_rgb_ql} {args.rgb_tile_out} -z 4-11 --srcnodata 0 --processes=40 -r antialias'


        cmd_str += f'&& echo {loc} >> {run_file}'
        #cmd_str = f'srun -N 1 -c 40 --mem=180G --pty {cmd_str}'
        cmd_str = f'sbatch -N 1 -c 40 --mem=180G --wrap="{cmd_str}"'
        print(cmd_str)
        subprocess.call(cmd_str,shell=True)

        

if __name__ == "__main__":
    main()
