






from spectral.io import envi
import argparse
import numpy as np
import subprocess
import os
import yaml
import glob
import json
import ray

@ray.remote
def get_single_footprint(glt, index, length, previous_build, overwrite, read_clouds=False):
        print(f'{index}/{length-1}')
        fid = os.path.basename(glt).split('_l1b_glt')[0]
        # If we have this key and it's already been sent to the DAAC, no need to rebuild
        if not overwrite and previous_build is not None and fid in list(previous_build.keys()) and previous_build[fid]['properties']['style']['color'] == "#0000FF":
            return previous_build[fid], index

        #meta = envi.open(glt).metadata
        #gring = meta['gring']
        #gring.pop(0)
        #gring = [float(x.strip()) for x in gring] 
        #gring = [[gring[x], gring[x+1]] for x in range(0,len(gring),2)]
        #gring.append([gring[0][0],gring[0][1]])


        raw_meta = envi.open(glt.replace('l1b','l1a').replace('glt','raw')).metadata
        dcid = os.path.basename(raw_meta['emit pge input files'][0]).split('_')[0]
        st = os.path.basename(raw_meta['emit acquisition start time'])[:19] + 'Z'
        et = os.path.basename(raw_meta['emit acquisition stop time'])[:19] + 'Z'

        aod, h2o, ele, l2a_quality  = None, None, None, None

        l2a_file = glt.replace('l1b','l2a').replace('glt','statesubs')
        mask_file = glt.replace('l1b','l2a').replace('glt','mask')
        segmentation_file = glt.replace('l1b','l2a').replace('glt','lbl')
        l2a_quality_file = glt.replace('l1b','l2a').replace('glt','quality').replace('.hdr','.txt')
        if os.path.isfile(l2a_file):
            l2a_state_subs_ds = envi.open(l2a_file)
            l2a_state_subs = l2a_state_subs_ds.open_memmap(interleave='bip')
            l2a_names = l2a_state_subs_ds.metadata['band names']
            if 'AOT550' in l2a_names:
                aod = l2a_state_subs[...,l2a_names.index('AOT550')]

            if 'H2OSTR' in l2a_names:
                h2o = l2a_state_subs[...,l2a_names.index('H2OSTR')]

            if 'GNDALT' in l2a_names:
                ele = l2a_state_subs[...,l2a_names.index('GNDALT')]

            l2a_quality = np.genfromtxt(l2a_quality_file)

        cirrus, cloud, buff, screened, total_clouds = None, None, None, None, None
        if read_clouds and os.path.isfile(mask_file):
            masks = envi.open(mask_file).open_memmap(interleave='bip')
            seg = np.squeeze(envi.open(segmentation_file).open_memmap(interleave='bip'))
            cloud = masks[...,0].copy()
            cirrus = masks[...,1].copy()
            buff = masks[...,4].copy()

            pixel_count = buff.shape[0]*buff.shape[1]

            cirrus = np.sum(np.logical_and(seg != 0, cloud + cirrus > 0)) 
            cloud = np.sum(np.logical_and(seg != 0, cloud  > 0)) 
            buff = np.sum(np.logical_and(seg != 0, buff  > 0)) 

            screened = np.sum(seg == 0)

            total_clouds = buff + screened

            cirrus /= pixel_count
            cloud /= pixel_count
            buff /= pixel_count
            screened /= pixel_count
            total_clouds /= pixel_count
            


        obs_file = glt.replace('glt','obs')
        obs_ds = envi.open(obs_file).open_memmap(interleave='bip')
        obs_centerline = np.mean(obs_ds[int(obs_ds.shape[0]/2),:,:].copy(), axis=0)
        obs_names = envi.open(obs_file).metadata['band names']

        loc_file = glt.replace('glt','loc')
        loc_ds = envi.open(loc_file).open_memmap(interleave='bip')
        loc_centerline = np.mean(loc_ds[int(loc_ds.shape[0]/2),:,:].copy(), axis=0)
        loc_names = envi.open(loc_file).metadata['band names']
        gring = []
        gring.append(loc_ds[0,0,:2].copy().tolist())
        gring.append(loc_ds[0,-1,:2].copy().tolist())
        gring.append(loc_ds[-1,-1,:2].copy().tolist())
        gring.append(loc_ds[-1,0,:2].copy().tolist())
        gring.append(gring[0])

        #try:
        #    mask_meta = envi.open(glt.replace('l1b','l2a').replace('glt','mask')).metadata
        #    cloud_fraction = mask_meta['emit acquisition cloudfraction']
        #except:
        #    cloud_fraction = 'NaN'

        loc_entry = {"geometry": {"coordinates": [gring],"type":"Polygon"},
                     "properties":{"fid":fid, "dcid": dcid,"start_time":st, "end_time":et}, "type":"Feature"}

        for _o, on in enumerate(obs_names):
            loc_entry['properties'][on] = obs_centerline[_o]

        for _o, on in enumerate(loc_names):
            loc_entry['properties'][on] = loc_centerline[_o]

        if aod is not None:
            loc_entry['properties']['Retrieved AOT Median'] = np.median(aod)
            loc_entry['properties']['Retrieved AOT Min'] = np.min(aod)
            loc_entry['properties']['Retrieved AOT Max'] = np.max(aod)
        if h2o is not None:
            loc_entry['properties']['Retrieved WV Median'] = np.median(h2o)
            loc_entry['properties']['Retrieved WV Min'] = np.min(h2o)
            loc_entry['properties']['Retrieved WV Max'] = np.max(h2o)
        if ele is not None:
            loc_entry['properties']['Retrieved Ele. Median'] = np.median(ele)
            loc_entry['properties']['Retrieved Ele. Min'] = np.min(ele)
            loc_entry['properties']['Retrieved Ele. Max'] = np.max(ele)
        if l2a_quality is not None:
            loc_entry['properties']['L2A Quality 50'] = l2a_quality[0]
            loc_entry['properties']['L2A Quality 95'] = l2a_quality[1]
            loc_entry['properties']['L2A Quality 99.9'] = l2a_quality[2]
        if cirrus is not None:
            loc_entry['properties']['Cloud Fraction'] = cloud
            loc_entry['properties']['Cloud + Cirrus Fraction'] = cirrus
            loc_entry['properties']['Clouds & Buffer Fraction'] = buff
            loc_entry['properties']['Screened Onboard Fraction'] = screened
            loc_entry['properties']['Total Cloud Fraction'] = total_clouds

        loc_entry['properties']['Request Link'] = f'https://docs.google.com/forms/d/e/1FAIpQLSc9pitlAVhrOhjkZrgo2At2DN3L6-wWhje8qEglGnSfcUnGcg/viewform?usp=pp_url&entry.1512332345={fid}'
        loc_entry['properties']['Orbit'] = fid.split('_')[1][1:]
        loc_entry['properties']['Orbit Segment'] = fid.split('_')[2][1:]
        #loc_entry['properties']['Cloud Fraction'] = cloud_fraction

        #l1b_rad = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/EMIT_L1B_RAD*_cnm.out')
        cmr_json_file = glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/emit*.cmr.json')

        on_daac = False
        if len(cmr_json_file) > 0:

          cmr = json.load(open(cmr_json_file[0]))
          cmr = cmr['DataGranule']['ArchiveAndDistributionInformation']
          for fn in cmr:
            if 'EMIT_L1B_RAD' in fn['Name']:
                on_daac = True
                l1b_rad = os.path.splitext(fn['Name'])[0]
                l1b_obs = l1b_rad.replace('RAD','OBS')

                l2a_rfl = l1b_rad.replace('L1B_RAD','L2A_RFL')
                l2a_rfluncert = l1b_rad.replace('L1B_RAD','L2A_RFLUNCERT')
                l2a_mask = l1b_rad.replace('L1B_RAD','L2A_MASK')

                loc_entry['properties']['L1B Radiance Download'] = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{l1b_rad}/{l1b_rad}.nc'
                loc_entry['properties']['L1B Observation Download'] = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{l1b_rad}/{l1b_obs}.nc'

                loc_entry['properties']['L2A Reflectance Download'] = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{l2a_rfl}/{l2a_rfl}.nc'
                loc_entry['properties']['L2A Reflectance Uncertainty Download'] = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{l2a_rfl}/{l2a_rfluncert}.nc'
                loc_entry['properties']['L2A Mask Download'] = f'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{l2a_rfl}/{l2a_mask}.nc'
                #loc_entry['properties']['LP DAAC Download'] = '#00F'
                loc_entry['properties']['style'] = {"weight":1,"opacity":1,"fillColor": "#0000FF", "color": "#0000FF"}
                break

        if on_daac is False:
            #loc_entry['properties']['LP DAAC Download'] = '#fc0'
            loc_entry['properties']['style'] = {"weight":1,"opacity":1,"fillColor": "#f6c409", "color": "#f6c409"}

        return loc_entry, index
 

class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)



def main():

    parser = argparse.ArgumentParser(description="Run coverage vector tiling")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--out_base', type=str, default='track_coverage')
    parser.add_argument('--read_clouds', type=int, default='1')
    args = parser.parse_args()

    glt_files = sorted(glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_glt_*.hdr'))
    out_file = f'{args.out_base}.json'
    out_file_pub = f'{args.out_base}_pub.json'

    previous_build = None
    if os.path.isfile(out_file):
        previous_build_init = json.load(open(out_file))
        previous_build = {}
        for feat in previous_build_init['features']:
            previous_build[feat['properties']['fid']] = feat
        del previous_build_init


    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},"features":[],"name":"aquisitions","type":"FeatureCollection" }


    ray.init()
    previous_build_ray = ray.put(previous_build)
    jobs = []
    for _glt, glt in enumerate(glt_files):
        jobs.append(get_single_footprint.remote(glt, _glt, len(glt_files), previous_build_ray, args.overwrite, args.read_clouds == 1))

    rreturn = [ray.get(jid) for jid in jobs]
    daac_index = 1
    for _res, res in enumerate(rreturn):
        res[0]['properties']['FID_index'] = _res + 1
        res[0]['properties']['DAAC_index'] = daac_index
        if 'L1B Radiance Download' in list(res[0]['properties'].keys()):
            daac_index += 1

        outdict['features'].append(res[0])

    with open(out_file, 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder, indent=2, sort_keys=True)) 
        
    subprocess.call(f'cat {out_file} | grep -v "Request Link" > {out_file_pub}',shell=True)













if __name__ == "__main__":
    main()
