

import os
import subprocess
import argparse
import numpy as np
import glob
from spectral.io import envi
from emit_utils.file_checks import envi_header


def main():

    parser = argparse.ArgumentParser(description="Run visuals workflow")
    #parser.add_argument('input_file_list', type=str)
    args = parser.parse_args()


    #loclist = np.genfromtxt(args.input_file_list,dtype=str)
    loclist = sorted(glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_loc_*.img'))

    badlist = ['emit20220731t095903_o21207_s000','emit20220731t052033_o21204_s000','emit20220728t025143_o20901_s001']


    for loc in loclist:

        flagged = False
        for bf in badlist:
            if bf in loc:
                flagged = True
                break
        if flagged:
            continue
    
        rdn = loc.replace('loc','rdn')
        #rdn = loc.replace('l1b','l2a').replace('loc','rfl')
        mask = loc.replace('l1b','l2a').replace('loc','mask')
        sa = loc.replace('l1b','l2b').replace('loc','abun')
        sma = loc.replace('l1b','l3').replace('loc','cover')
        
        use_glt = False
        if '20220728' not in rdn and \
        '20220730' not in rdn and \
        '20220731' not in rdn and \
        '20220803' not in rdn and \
        '20220804' not in rdn:
            use_glt = True

        if use_glt:
            glt = loc.replace('loc','glt')
        else:
            glt = f'glts/{os.path.basename(loc).replace("loc","glt")}'
        ortho_sa = f'ortho/sa/{os.path.basename(sa)}'
        ortho_sma = f'ortho/sma/{os.path.basename(sma)}'
        ortho_rgb = f'ortho/rgb/{os.path.basename(rdn)}'
        ortho_mask = f'ortho/rgb/{os.path.basename(mask)}'
        
        ortho_sa_ql = f'ortho/sa_ql/{os.path.basename(sa)}'
        ortho_sma_ql = f'ortho/sma_ql/{os.path.basename(sma)}'
        ortho_rgb_ql = f'ortho/rgb_ql/{os.path.basename(rdn)}'
        ortho_mask_ql = f'ortho/mask_ql/{os.path.basename(mask)}'

        if envi.open(envi_header(rdn)).metadata['emit acquisition daynight'] != 'Day':
            continue
        
        cmd_str = ''
        if use_glt is False and os.path.isfile(glt) is False:
            cmd_str+=f'julia /beegfs/scratch/brodrick/emit/emit-sds-l3/build_mosaic_glt.jl {glt} {loc} 0.0005454 0.0005454 --mosaic 0 --output_epsg 4326; '
        #if os.path.isfile(ortho_sa) is False:
        #    cmd_str+= f'python /beegfs/store/emit/ops/repos/emit-sds-l3/apply_glt.py {glt} {sa} {ortho_sa} -n_cores 40 -one_based_glt 1; '

        #if os.path.isfile(ortho_sma) is False:
        #    cmd_str+= f'python /beegfs/store/emit/ops/repos/emit-sds-l3/apply_glt.py {glt} {sma} {ortho_sma} -n_cores 40 -one_based_glt 1; '
        #if os.path.isfile(ortho_rgb) is False:
        if os.path.isfile(ortho_rgb) is False:
            cmd_str+= f'python /beegfs/store/emit/ops/repos/emit-sds-l3/apply_glt.py {glt} {rdn} {ortho_rgb} -n_cores 40 -one_based_glt 1 -b 37 25 13; '
        #if os.path.isfile(ortho_mask) is False:
        #    cmd_str+= f'python /beegfs/store/emit/ops/repos/emit-sds-l3/apply_glt.py {glt} {mask} {ortho_mask} -n_cores 40 -one_based_glt 1; '

        if os.path.isfile(ortho_rgb_ql) is False:
            # preferred....but neeeds mask
            #cmd_str+= f'python quicklooks/rgb_ql.py {ortho_rgb} {ortho_rgb_ql} --mask_file {ortho_mask};'
            cmd_str+= f'python quicklooks/rgb_ql.py {ortho_rgb} {ortho_rgb_ql};'

        #if os.path.isfile(ortho_mask_ql) is False:
        #    cmd_str+= f'python quicklooks/mask_ql.py {ortho_mask} {ortho_mask_ql};'

        #if os.path.isfile(ortho_sma_ql) is False:
        #    cmd_str+= f'python quicklooks/cover_ql.py {ortho_sma} {ortho_sma_ql} --mask_file {ortho_mask}; '
        #if os.path.isfile(ortho_sa_ql) is False:
        #    cmd_str+= f'python quicklooks/sa_ql.py {ortho_sa} {ortho_sa_ql} --mask_file {ortho_mask}'
    
        if cmd_str != "":
            cmd_str = f'sbatch -N 1 -c 40 -p standard --mem=180G --wrap="{cmd_str}"'
    
            print(cmd_str)
            if cmd_str != "":
                subprocess.call(cmd_str,shell=True)




if __name__ == "__main__":
    main()
