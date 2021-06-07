'''
Parse DOS xcd data
'''
import xml.etree.ElementTree as ET
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


class XcdFile:
    '''
    Parse the xcd and return DOS data
    Initiate instance with a passed xcd path

    ...

    Attributes:
        path (Path): Path object directing to the XCD file
        pwd (Path): Current working directory when the script is running
        bands (dict): dictionary of parsed electronic bands
        name_stem (Path): Path stem of xcd path
    '''
    def __init__(self, path: Path):
        self.path = path
        self.pwd = Path.cwd()
        try:
            tree: ET.ElementTree = ET.parse(path)
        except ET.ParseError as empty_file:
            raise AttributeError(
                f"Error in {path}: no element found") from empty_file
        else:
            root = tree.getroot()
            bands: List[ET.Element] = root.findall('.//SERIES_2D')
            self.bands = {band.attrib["Name"][0]: band for band in bands}
            self.name_stem = path.stem

    @property
    def name(self):
        '''
        return attribute: name
        Read-only
        '''
        return self.name_stem.replace('_DOS', '')

    def get_xy(self, band: str) -> Tuple[np.ndarray, np.ndarray]:
        ''' Parse the coordinates stored in xcd
        Args:
            band (str): ['s', 'p', 'd'] or ['s', 'p', 'd', 'f']
        '''
        xy = [item.attrib["XY"] for item in self.bands[band]]  #pylint: disable=invalid-name
        x = np.array([float(item.split(',')[0]) for item in xy])  #pylint: disable=invalid-name
        y = np.array([float(item.split(',')[1]) for item in xy])  #pylint: disable=invalid-name
        return x, y

    def get_dos_df(self):
        '''
        Sort parsed DOS bands, return DOS dataframe
        '''
        s_e, s_dos = self.get_xy('s')
        p_e, p_dos = self.get_xy('p')
        d_e, d_dos = self.get_xy('d')
        f_e, f_dos = self.get_xy('f')
        dos_dict = dict(se=s_e,
                        sdos=s_dos,
                        pe=p_e,
                        pdos=p_dos,
                        de=d_e,
                        ddos=d_dos,
                        fe=f_e,
                        fdos=f_dos)
        df = pd.DataFrame.from_dict(  #pylint: disable=invalid-name
            {k: pd.Series(v)
             for k, v in dos_dict.items()})
        return df

    def band_center(self, band: str):
        """ Calculate band center
        Args:
            band (str): band to calculate, ['s', 'p', 'd', 'f']
        """
        if band not in ["s", "p", "d", "f"]:
            raise ValueError(
                f"Input {band} is not supported! Please choose from s, p, d and f"
            )
        np.seterr(invalid="ignore")
        x, y = self.get_xy(band)  #pylint: disable=invalid-name
        pdE = np.trapz(y, x)  #pylint: disable=invalid-name
        pEdE = np.trapz(y * x, x)  #pylint: disable=invalid-name
        center = pEdE / pdE
        return center
