






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
def get_single_footprint(glt, index, length):
        print(f'{index}/{length-1}')
        meta = envi.open(glt).metadata
        gring = meta['gring']
        gring.pop(0)
        gring = [float(x.strip()) for x in gring] 
        gring = [[gring[x], gring[x+1]] for x in range(0,len(gring),2)]
        gring[0].append([gring[0][0],gring[0][1]])
        fid = os.path.basename(glt).split('_l1b_glt')[0]


        raw_meta = envi.open(glt.replace('l1b','l1a').replace('glt','raw')).metadata
        dcid = os.path.basename(raw_meta['emit pge input files'][0]).split('_')[0]
        st = os.path.basename(raw_meta['emit acquisition start time'])[:19]
        et = os.path.basename(raw_meta['emit acquisition stop time'])[:19]

        aod, h2o, ele, l2a_quality  = None, None, None, None

        l2a_file = glt.replace('l1b','l2a').replace('glt','statesubs')
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


        obs_file = glt.replace('glt','obs')
        obs_ds = envi.open(obs_file).open_memmap(interleave='bip')
        obs_centerline = np.mean(obs_ds[int(obs_ds.shape[0]/2),:,:].copy(), axis=0)
        obs_names = envi.open(obs_file).metadata['band names']

        loc_file = glt.replace('glt','loc')
        loc_ds = envi.open(loc_file).open_memmap(interleave='bip')
        loc_centerline = np.mean(loc_ds[int(loc_ds.shape[0]/2),:,:].copy(), axis=0)
        loc_names = envi.open(loc_file).metadata['band names']

        loc_entry = {"geometry": {"coordinates": [gring],"type":"Polygon"},
                     "properties":{"name": f"{fid}\n{dcid}", "fid":fid, "dcid": dcid,"start_time":st, "end_time":et}, "type":"Feature"}

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

        loc_entry['properties']['Request Link'] = f'https://docs.google.com/forms/d/e/1FAIpQLSc9pitlAVhrOhjkZrgo2At2DN3L6-wWhje8qEglGnSfcUnGcg/viewform?usp=pp_url&entry.1512332345={fid}'

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

    glt_files = glob.glob('/beegfs/store/emit/ops/data/acquisitions/*/*/l1b/*_l1b_glt_*.hdr')
    out_file = 'track_coverage.json'


    outdict = {"crs": {"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}, "type": "name"},"features":[],"name":"aquisitions","type":"FeatureCollection" }


    ray.init()
    jobs = []
    for _glt, glt in enumerate(glt_files):
        jobs.append(get_single_footprint.remote(glt, _glt, len(glt_files)))

    rreturn = [ray.get(jid) for jid in jobs]
    for res in rreturn:
        outdict['features'].append(res[0])

    with open(out_file, 'w') as fout:
        fout.write(json.dumps(outdict, cls=SerialEncoder, indent=2, sort_keys=True)) 
        













if __name__ == "__main__":
    main()
