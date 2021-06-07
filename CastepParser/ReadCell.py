"""
Methods to parse CASTEP file
"""
import re
from typing import (Union, Tuple)
from pathlib import Path
import pandas as pd
import numpy as np


class CastepFile:
    """
    A class to parse CASTEP file

    ...

    Attributes:
        filepath : Path
            Path object of the CASTEP file
        body : str
            Total string of the CASTEP file
        final_energy: float
            The last result of Final energy in .castep
        atom_num: int
            Total number of atoms
        spec_num: int
            Total number of species
        area: float
            Lattice surface area (XY plane)
        lattice_param: Tuple(float, float, float)
            lattice parameters

    Methods:
        extract_pressure:
            Extract pressure if computed
        extract_cell :
            Extract cell configuration
        extract_gamma:
            Extract gamma angle
        extract_volume:
            Extract volume
    """
    def __init__(self, filepath: Path):
        self.filepath: Path = filepath
        self.body = self.filepath.read_text()

    @property
    def final_energy(self) -> float:
        """
        Extract the last result of Final energy
        """
        energy = re.compile(r'(?<=Final\senergy,\sE\s{13}=\s{2}).*\d+')
        text = self.body
        energy_value = float(energy.findall(text)[-1])
        return energy_value

    @property
    def atom_num(self) -> int:
        '''
        Read total number of atoms in cell
        '''
        text = self.body
        number = re.compile(r'Total\snumber\sof\sions\sin\scell\s=\s+(\d+)')
        try:
            nums = int(number.findall(text)[0])
        except IndexError:
            '''
            For debugging
            '''
            print(self.filepath)
        return nums

    @property
    def spec_num(self) -> int:
        '''
        Read total number of species in cell
        '''
        text = self.body
        species = re.compile(r"Total number of species in cell =\s+([0-9]+)")
        number = int(species.search(text).group(1))
        return number

    @property
    def area(self) -> np.float64:
        '''
        attribute: lattice surface area
        returns:
            lattice surface area
        '''
        text = self.body
        param_pattern = re.compile(r'\s+[a-c]\s\=\s+(\S+)')
        a, b = [float(item) for item in param_pattern.findall(text)][:2]  # pylint: disable=invalid-name
        gamma = re.search(r"gamma\W+(\S+)", text).group(1)
        gm_angle = float(gamma)
        area: np.float64 = a * b * np.sin(np.radians(gm_angle))
        return area

    @property
    def lattice_param(self) -> Tuple[float, float, float]:
        """
        attribute: lattice parameters
        """
        text = self.body
        param_pattern = re.compile(r'\s+[a-c]\s\=\s+(\S+)')
        a, b, c = [float(item) for item in param_pattern.findall(text)]  # pylint: disable=invalid-name
        return a, b, c

    def extract_pressure(self) -> Union[float, str]:  #pylint: disable=unsubscriptable-object
        """
        Parse pressure from CASTEP body text
        returns:
            pressure value (float): return pressure if exists
        or:
            Warning message (str): when the computation did not compute stress
        """
        text = self.body
        pre = re.compile(
            r'(?<=Pressure:\s{3})\-\d+\.\d+|(?<=Pressure:\s{3})\d+\.\d+|(?<=Pressure:\s{2})\d+\.\d+|(?<=Pressure:\s{4})\d+\.\d+'  #pylint: disable=line-too-long
        )
        if pre.search(text):
            return float(pre.findall(text)[0])
        return f"Pressure in {self.filepath.name} is missing!"

    def extract_cell(self) -> pd.DataFrame:
        """
        Extract cell configuration from CASTEP file
        returns:
            cell_df (pd.DataFrame): Dataframe of cell configuration
        """
        cell = re.compile(r"(?<=\s{3}x\s).*(?=\s{2,3}x)")
        raw_lines = cell.findall(self.body)[3::]
        purged_lines = [[s for s in line.split(" ") if s != '']
                        for line in raw_lines]
        cell_df = pd.DataFrame.from_records(
            purged_lines, columns=["Element", "Atom Number", "u", "v", "w"])
        cell_df[["u", "v", "w"]] = cell_df[["u", "v", "w"]].astype('float')
        cell_df["Atom Number"] = cell_df["Atom Number"].astype('int')
        return cell_df

    def extract_gamma(self) -> float:
        """
        Extract gamma angle of cell from CASTEP file
        Args:
            body (str): read from CASTEP file
        returns:
            gamma (float)
        """
        gamma_re = re.compile(r"(?<=gamma\s=)\s+[0-9.]+")
        gamma = gamma_re.search(self.body).group()  #type: ignore
        return float(gamma)

    def extract_volume(self) -> float:
        """
        Extract cell volume from CASTEP file
        returns:
            volume (float)
        """
        volume_re = re.compile(r"(?<=Current cell volume =  )[0-9.]+")
        volume = volume_re.search(self.body).group()  #type: ignore
        return float(volume)
