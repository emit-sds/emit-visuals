






from spectral.io import envi
import argparse
import numpy as np
import subprocess
import os
import yaml
import glob
import json



def main():

    glt_files = glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_glt_*.hdr')
    out_file = 'track_coverage.json'


    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},"features":[],"name":"aquisitions","type":"FeatureCollection" }


    for glt in glt_files:
        meta = envi.open(glt).metadata
        gring = meta['gring']
        gring.pop(0)
        gring = [float(x.strip()) for x in gring] 
        gring = [[gring[x], gring[x+1]] for x in range(0,len(gring),2)]
        gring[0].append([gring[0][0],gring[0][1]])
        fid = os.path.basename(glt).split('_l1b_glt')[0]

        #emit pge input files = { frames_dir=/beegfs/store/emit/ops/data/data_collections/by_dcid/13434/1343493746/1343493746_frames_b0105_v01 , flexcodec_exe_path=/beegfs/store/emit/ops/repos/EMIT_FLEX_codec/flexcodec , constants_path=/beegfs/store/emit/ops/repos/emit-sds-l1a/decompression/constants.txt , init_data_path=/beegfs/store/emit/ops/repos/emit-sds-l1a/decompression/FPGA_Data_Initialization_File_CREATE_COMPRESSION_INIT_DATA_328_e0.bin }
        raw_meta = envi.open(glt.replace('l1b','l1a').replace('glt','raw')).metadata
        dcid = os.path.basename(raw_meta['emit pge input files'][0]).split('_')[0]
        
        outdict['features'].append({"geometry": {"coordinates": [gring],"type":"Polygon"},"properties":{"name": f"{fid}\n{dcid}", "fid":fid, "dcid": dcid}, "type":"Feature"})


    with open(out_file, 'w') as fout:
        fout.write(json.dumps(outdict, indent=2, sort_keys=True)) 
        













if __name__ == "__main__":
    main()
