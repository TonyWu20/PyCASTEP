# /usr/local/bin/python3.7
import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np
import timeit
import tables
import warnings
import fire
from p_tqdm import p_map
from itertools import chain
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


def extract_dos(_band, _key):
    band = _band
    get = {
        'e': "[float(item.attrib['XY'].split(',')[0]) for item in band]",
        'dos': "[float(item.attrib['XY'].split(',')[1]) for item in band]"
    }
    return np.array(eval(get[_key]))


def dcenter(x, y):
    pdE = np.trapz(y, x)
    pEdE = np.trapz(y*x, x)
    dc = pEdE/pdE
    return dc


def check_ab(_alpha, _beta):
    alpha = _alpha
    beta = _beta
    x_a, x_b = alpha['dE_alpha'], beta['dE_beta']
    y_a, y_b = alpha['d_alpha'], beta['d_beta']

    dc_alpha = dcenter(x_a, y_a)
    dc_beta = dcenter(x_b, y_b)
    return dc_alpha > dc_beta


def band_extract(_root):
    root = _root
    bands = root.findall('.//SERIES_2D')
    if len(bands) == 4:
        bands = {
            's_alpha': bands[0],
            's_beta': bands[1],
            'p_alpha': bands[2],
            'p_beta': bands[3]
        }
        d_band_exists = False
    elif len(bands) == 6:
        bands = {
            's_alpha': bands[0],
            's_beta': bands[1],
            'p_alpha': bands[2],
            'p_beta': bands[3],
            'd_alpha': bands[4],
            'd_beta': bands[5]
        }
        d_band_exists = True
    return bands, d_band_exists


def parse_xcd(_path):
    tree = ET.parse(_path)
    root = tree.getroot()
    bands, d_band_exists = band_extract(root)
    if d_band_exists == False:
        DOS = {
            's_alpha': {
                'sE_alpha': extract_dos(bands['s_alpha'], 'e'),
                's_alpha': extract_dos(bands['s_alpha'], 'dos')
            },
            's_beta': {
                'sE_beta': extract_dos(bands['s_beta'], 'e'),
                's_beta': extract_dos(bands['s_beta'], 'dos')*-1
            },
            'p_alpha': {
                'pE_alpha': extract_dos(bands['p_alpha'], 'e'),
                'p_alpha': extract_dos(bands['p_alpha'], 'dos')
            },
            'p_beta': {
                'pE_beta': extract_dos(bands['p_beta'], 'e'),
                'p_beta': extract_dos(bands['p_beta'], 'dos')*-1
            }
        }
    else:
        DOS = {
            's_alpha': {
                'sE_alpha': extract_dos(bands['s_alpha'], 'e'),
                's_alpha': extract_dos(bands['s_alpha'], 'dos')
            },
            's_beta': {
                'sE_beta': extract_dos(bands['s_beta'], 'e'),
                's_beta': extract_dos(bands['s_beta'], 'dos')*-1
            },
            'p_alpha': {
                'pE_alpha': extract_dos(bands['p_alpha'], 'e'),
                'p_alpha': extract_dos(bands['p_alpha'], 'dos')
            },
            'p_beta': {
                'pE_beta': extract_dos(bands['p_beta'], 'e'),
                'p_beta': extract_dos(bands['p_beta'], 'dos')*-1
            },
            'd_alpha': {
                'dE_alpha': extract_dos(bands['d_alpha'], 'e'),
                'd_alpha': extract_dos(bands['d_alpha'], 'dos')
            },
            'd_beta': {
                'dE_beta': extract_dos(bands['d_beta'], 'e'),
                'd_beta': extract_dos(bands['d_beta'], 'dos')*-1
            }
        }

        if check_ab(DOS['d_alpha'], DOS['d_beta']):
            DOS['d_alpha'] = {
                'dE_alpha': extract_dos(bands['d_beta'], 'e'),
                'd_alpha': extract_dos(bands['d_beta'], 'dos')*-1
            }
            DOS['d_beta'] = {
                'dE_beta': extract_dos(bands['d_alpha'], 'e'),
                'd_beta': extract_dos(bands['d_alpha'], 'dos')
            }

    df = {key:pd.DataFrame.from_dict(DOS[key]) for key in DOS.keys()}
    total_df = pd.concat(list(df.values()), axis=1)
    return total_df


def get_dos(_path, _target, _dir):
    filelist = os.listdir(_path)
    filelist.sort()
    if 'subsurface' in filelist[1] and 'bulk' in filelist[0]:
        compressive = filelist[:12]
        compressive.sort(reverse=True)
        tensile = filelist[12:]
        sf = tensile[2::3]
        subsf = tensile[1::3]
        bk = tensile[::3]
        new_ten = [[sf[i], subsf[i], bk[i]] for i in range(len(sf))]
        ten_array = np.array(new_ten)
        flatten_ten = ten_array.flatten()
        new_ten = flatten_ten.tolist()
        new_filelist = compressive + new_ten
    elif 'surface' in filelist[1]:
        compressive = filelist[:8]
        compressive.sort(reverse=True)
        tensile = filelist[8:]
        sf = tensile[1::2]
        bk = tensile[::2]
        new_ten = list(chain.from_iterable(zip(sf, bk)))
        new_filelist = compressive + new_ten
    elif 'surface' not in filelist[1]:
        compressive = filelist[:4]
        compressive.sort(reverse=True)
        tensile = filelist[4:]
        new_filelist = compressive + tensile

    key_list = [item.replace('_PDOS.xcd', '') for item in new_filelist]
    pathlist = [f"{_path}/{item}" for item in new_filelist]
    dataframes = p_map(parse_xcd, pathlist)
    for key, item in zip(key_list, dataframes):
        item.to_hdf(f"{_dir}/DOS_of_{_target}.h5", key, format='table')
    index_df = pd.Series(key_list)
    index_df.to_hdf(f"{_dir}/DOS_of_{_target}.h5", "index", format="table")
    print('Done!')


def main(_target, _path="Raw data"):
    start = timeit.default_timer()
    target = _target
    path = f"{_path}/{target}_dos"
    out_path = f"Spin_data_analysis/data of {target}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    get_dos(path, target, out_path)
    end = timeit.default_timer()
    print(f"Took {end-start:.2f} seconds")


if __name__ == '__main__':
    fire.Fire(main)
