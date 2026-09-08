import os
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import f90nml
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import plotly 
import plotly.graph_objects as go

from simsopt.mhd import Vmec


@dataclass
class FourierDat:
    """
    Represents the mode structure for a STELLGAP run as specified in a fourier.dat file.

    Attributes:
        Nfp (int): Number of field periods. For tokamaks, typically 1.
        ith (int): Number of zeta grid points, maximum mpol is calculated as ith * 2 / 5.
        izt (int): Number of theta grid points, maximum ntor is calculated as izt * 2 / 5.
        mode_family (int): Identifier for the mode family, not currently used in calculations.
        mode_definitions (List[Tuple[int, int, int]]): List of mode definitions with (nw, mwl, mwu).
        original_content (str): The original text content of the fourier.dat file.
    """
    Nfp: int
    ith: int
    izt: int
    mode_family: int
    mode_definitions: List[Tuple[int, int, int]] = field(default_factory=list)
    original_content: str = ""

    def __post_init__(self):
        self.mpol_max = self.ith * 2 // 5
        self.ntor_max = self.izt * 2 // 5

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
        original_content = "".join(lines)

        header = lines[0].split()
        Nfp = int(header[0])
        ith = int(header[1])
        izt = int(header[2])
        mode_family = int(header[3])
        nt_col = int(lines[1].strip())

        mode_definitions = []
        for i in range(2, 2 + nt_col):
            parts = lines[i].split()
            nw = int(parts[0])
            mwl = int(parts[1])
            mwu = int(parts[2])
            mode_definitions.append((nw, mwl, mwu))

        return cls(Nfp, ith, izt, mode_family, mode_definitions, original_content)

    def __str__(self):
        return self.explain()

    def explain(self):
        return f"""***Original Content***
{self.original_content}

Explanation:
Nfp = {self.Nfp} - single field period as in a tokamak
ith = {self.ith} - zero grid has {self.ith} points, therefore maximum mpol that can be resolved is {self.mpol_max}
izt = {self.izt} - theta grid has {self.izt} points, therefore maximum ntor that can be resolved is {self.ntor_max}
nt_col = {len(self.mode_definitions)} - number of mode definitions; determines how many lines of fourer.dat to read
mode_family = {self.mode_family} - this number is currently not used in calculation, but read by code
Mode Definitions: {self.mode_definitions}
        """


@dataclass
class PlasmaDat:
    """
    Represents the plasma configuration for a STELLGAP run as specified in a plasma.dat file.
    """

    ion_to_proton_mass: float
    ion_density_0: float
    ion_profile: int
    jdqz_data: bool
    egnout_form: str
    nion: list
    aion: float = None
    bion: float = None
    cion: float = None
    original_content: str = ""

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r") as file:
            content = file.read()
        namelist = f90nml.reads(content)
        plasma_input = namelist["plasma_input"]

        aion = plasma_input.get("aion", 0.0)
        bion = plasma_input.get("bion", 0.0)
        cion = plasma_input.get("cion", 0.0)

        return cls(
            ion_to_proton_mass=plasma_input["ion_to_proton_mass"],
            ion_density_0=plasma_input["ion_density_0"],
            ion_profile=plasma_input["ion_profile"],
            jdqz_data=plasma_input["jdqz_data"],
            egnout_form=plasma_input["egnout_form"],
            nion=plasma_input["nion"],
            aion=aion,
            bion=bion,
            cion=cion,
            original_content=content,
        )

    def __str__(self):
        ion_profile_explanation = self.get_ion_profile_explanation()
        return f"""***Original Content***
{self.original_content}

Explanation:
ion_to_proton_mass = {self.ion_to_proton_mass}: Mass ratio of ion to proton.
ion_density_0 = {self.ion_density_0}: Ion density (m**-3) at the magnetic axis.
ion_profile = {self.ion_profile}: {ion_profile_explanation}
nion = {self.nion}: Coefficients for polynomial fit (ion_profile = 1) or parameters for other profiles.
aion = {self.aion}, bion = {self.bion}, cion = {self.cion}: Parameters for ion density profile when ion_profile = 3.
jdqz_data = {self.jdqz_data}: Used in AE3D, not in STELLGAP; Whether to include data for JDQZ solve
egnout_form = {self.egnout_form}: Output form, used in AE3D, not in STELLGAP; Whether save in binary ("binr") or ASCII ("asci") format.
"""

    def get_ion_profile_explanation(self):
        profiles = {
            0: "Ion density is proportional to [iota(rho)/iota(0)]**2; rho is normalized tor. flux;",
            1: "Ion density is a polynomial fit nion(1) + ... + nion(9)*(rho**8); rho is normalized tor. flux;",
            2: "Ion density is constant;",
            3: "Ion density is [1 - aion*(rho**bion)]**cion.; rho is normalized tor. flux;",
        }
        explanation = profiles.get(self.ion_profile, "Unknown profile")
        additional_info = "Available profiles are: 0 (iota/iota on axis squared), 1 (10th order polynomial fit), 2 (constant), 3 (formula based)."
        return f"{explanation} {additional_info}"


@dataclass
class Mode:
    n: int
    m: int
    s: np.ndarray
    freq: np.ndarray


@dataclass
class TaeDataBoozer:
    sim_dir: str
    file_path: str = field(init=False)
    nznt: int = field(init=False)
    surfs: int = field(init=False)
    array: np.ndarray = field(init=False)
    field_descriptions: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.initialize_from_dir(self.sim_dir)

    @classmethod
    def from_dir(cls, sim_dir):
        instance = cls(sim_dir)
        instance.initialize_from_dir(sim_dir)
        return instance

    def initialize_from_dir(self, sim_dir):
        self.file_path = f"{sim_dir}/tae_data_boozer"
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")
        if not os.path.isfile(os.path.join(sim_dir, "fourier.dat")):
            raise FileNotFoundError("Fourier data file not found in the simulation directory.")
        self.nznt = self.get_nznt()
        self.surfs = self.get_number_of_surfaces()
        self.initialize_descriptions()
        self.array = self.load_data()
        self.check_data()

    def get_nznt(self):
        fd = FourierDat.from_file(os.path.join(self.sim_dir, "fourier.dat"))
        return fd.ith * fd.izt

    def get_number_of_surfaces(self):
        with open(self.file_path, "r") as file:
            return sum(1 for _ in file) // (1 + 2 * self.nznt)

    def initialize_descriptions(self):
        self.field_descriptions = {
            "ks": "Index of the flux surface",
            "iota": "Iota on the flux surface",
            "phip": "Toroidal magnetic flux on the flux surface",
            "jtor": "Toroidal current density on the flux surface",
            "jpol": "Poloidal current density on the flux surface",
            "thetang": "Theta angle",
            "zetang": "Zeta angle",
            "bfield": "Magnetic field strength",
            "gsssup": "g^ss metric element",
            "gtzsup": "g^tz metric element",
            "gttsup": "g^tt metric element",
            "gzzsup": "g^zz metric element",
            "gstsup": "g^st metric element",
            "gszsup": "g^sz metric element",
            "rjacob": "Jacobian from cylindrical to Boozer coordinates (sqrt(g))",
        }

    def load_data(self):
        dtype = np.dtype(
            [
                ("ks", int),
                ("iota", "f8"),
                ("phip", "f8"),
                ("jtor", "f8"),
                ("jpol", "f8"),
                ("thetang", ("f8", self.nznt)),
                ("zetang", ("f8", self.nznt)),
                ("bfield", ("f8", self.nznt)),
                ("gsssup", ("f8", self.nznt)),
                ("gtzsup", ("f8", self.nznt)),
                ("gttsup", ("f8", self.nznt)),
                ("gzzsup", ("f8", self.nznt)),
                ("gstsup", ("f8", self.nznt)),
                ("gszsup", ("f8", self.nznt)),
                ("rjacob", ("f8", self.nznt)),
            ]
        )
        data = np.zeros(self.surfs, dtype=dtype)
        with open(self.file_path, "r") as file:
            for i in range(self.surfs):
                tae_data_block0 = np.loadtxt(
                    file,
                    max_rows=1,
                    dtype=np.dtype([("ks", int), ("iota", "f8"), ("phip", "f8"), ("jtor", "f8"), ("jpol", "f8")]),
                )
                tae_data_block1 = []
                tae_data_block2 = []
                for _ in range(self.nznt):
                    tae_data_block1.append(
                        np.loadtxt(
                            file,
                            max_rows=1,
                            dtype=np.dtype([("thetang", "f8"), ("zetang", "f8"), ("bfield", "f8"), ("gsssup", "f8"), ("gtzsup", "f8")]),
                        )
                    )
                    tae_data_block2.append(
                        np.loadtxt(
                            file,
                            max_rows=1,
                            dtype=np.dtype([("gttsup", "f8"), ("gzzsup", "f8"), ("gstsup", "f8"), ("gszsup", "f8"), ("rjacob", "f8")]),
                        )
                    )
                tae_data_block1 = np.array(tae_data_block1)
                tae_data_block2 = np.array(tae_data_block2)
                for field in dtype.names[:5]:
                    data[i][field] = tae_data_block0[field]
                for field in dtype.names[5:10]:
                    data[i][field] = tae_data_block1[field]
                for field in dtype.names[10:]:
                    data[i][field] = tae_data_block2[field]
        return data

    def check_data(self):
        for field in self.array.dtype.fields:
            if np.issubdtype(self.array[field].dtype, np.number):
                if np.any(np.isnan(self.array[field])):
                    warnings.warn(f"NaN values found in field '{field}'")
                if np.any(np.isinf(self.array[field])):
                    warnings.warn(f"Infinite values found in field '{field}'")
        if self.array.size == 0:
            warnings.warn(f"File {self.file_path} is empty")


@dataclass
class AeMetricData:
    sim_dir: str
    file_path: str = field(init=False)
    surfs: int = field(init=False)
    izeta: int = field(init=False)
    itheta: int = field(init=False)
    nznt: int = field(init=False)
    field_descriptions: dict = field(default_factory=dict, init=False)
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "ae_metric.dat")
        self.initialize_descriptions()
        self.array = self.load_data()

    def initialize_descriptions(self):
        self.field_descriptions = {
            "iota": "iota",
            "iotapf": "radial iota derivative",
            "jpol": "poloidal current density",
            "jpolpf": "radial derivative poloidal current density",
            "jtor": "toroidal current density",
            "jtorpf": "radial derivative toroidal current density",
            "phip": "toroidal flux",
            "phippf": "radial derivative toroidal flux",
            "rjacob": "Jacobian from cylindrical to Boozer coordinates (sqrt(g))",
            "bfield": "Magnetic field strength",
            "gsssup": "g^ss metric element (s = radial, t = theta, z = zeta)",
            "gttsup": "g^tt metric element",
            "gzzsup": "g^zz metric element",
            "gstsup": "g^st metric element",
            "gszsup": "g^sz metric element",
            "gtzsup": "g^tz metric element",
            "bfields": "radial derivative of magnetic field strength",
            "bfieldth": "theta derivative of magnetic field strength",
            "bfieldze": "zeta derivative of magnetic field strength",
            "jprl_coef0": "J_parallel/B coefficient 0",
            "jprl_coef1": "J_parallel/B coefficient 1",
            "jprl_coef2": "J_parallel/B coefficient 2",
            "prespf": "radial pressure gradient",
        }

    def load_data(self):
        with open(self.file_path, "r") as file:
            first_line = file.readline()
            numbers = list(map(int, first_line.split()))
            self.surfs, self.izeta, self.itheta, self.nznt = numbers[0:4]

            surface_data = np.loadtxt(
                file,
                max_rows=self.surfs,
                dtype=np.dtype(
                    [
                        ("iota", "f8"),
                        ("iotapf", "f8"),
                        ("jpol", "f8"),
                        ("jpolpf", "f8"),
                        ("jtor", "f8"),
                        ("jtorpf", "f8"),
                        ("phip", "f8"),
                        ("phippf", "f8"),
                    ]
                ),
            )

            dtype = np.dtype(
                [
                    ("iota", "f8"),
                    ("iotapf", "f8"),
                    ("jpol", "f8"),
                    ("jpolpf", "f8"),
                    ("jtor", "f8"),
                    ("jtorpf", "f8"),
                    ("phip", "f8"),
                    ("phippf", "f8"),
                    ("rjacob", ("f8", self.nznt)),
                    ("bfield", ("f8", self.nznt)),
                    ("gsssup", ("f8", self.nznt)),
                    ("gttsup", ("f8", self.nznt)),
                    ("gzzsup", ("f8", self.nznt)),
                    ("gstsup", ("f8", self.nznt)),
                    ("gszsup", ("f8", self.nznt)),
                    ("gtzsup", ("f8", self.nznt)),
                    ("bfields", ("f8", self.nznt)),
                    ("bfieldth", ("f8", self.nznt)),
                    ("bfieldze", ("f8", self.nznt)),
                    ("jprl_coef0", "f8"),
                    ("jprl_coef1", "f8"),
                    ("jprl_coef2", "f8"),
                    ("prespf", "f8"),
                    ("brho", ("f8", self.nznt)),
                ]
            )

            data = np.zeros(self.surfs, dtype=dtype)

            for field in surface_data.dtype.names:
                data[field] = surface_data[field]

            for i in range(self.surfs):
                surf_metric_data_block = np.loadtxt(
                    file,
                    max_rows=self.nznt,
                    dtype=np.dtype(
                        [
                            ("rjacob", "f8"),
                            ("bfield", "f8"),
                            ("gsssup", "f8"),
                            ("gttsup", "f8"),
                            ("gzzsup", "f8"),
                            ("gstsup", "f8"),
                            ("gszsup", "f8"),
                            ("gtzsup", "f8"),
                            ("bfields", "f8"),
                            ("bfieldth", "f8"),
                            ("bfieldze", "f8"),
                        ]
                    ),
                )
                for field in surf_metric_data_block.dtype.names:
                    data[i][field] = surf_metric_data_block[field]

            jprl_data_block = np.loadtxt(
                file,
                max_rows=self.surfs,
                dtype=np.dtype([("jprl_coef0", "f8"), ("jprl_coef1", "f8"), ("jprl_coef2", "f8"), ("prespf", "f8")]),
            )

            for field in jprl_data_block.dtype.names:
                data[field] = jprl_data_block[field]

            for i in range(self.surfs):
                brho_data_block = np.loadtxt(file, max_rows=self.nznt, dtype=np.dtype([("brho", "f8")]))
                for field in brho_data_block.dtype.names:
                    data[i][field] = brho_data_block[field]

        return data


@dataclass
class ModesOutput:
    equilibrium_modes: np.ndarray = field(default_factory=lambda: np.array([], dtype=[("meq", int), ("neq", int)]))
    eigenvector_modes: np.ndarray = field(default_factory=lambda: np.array([], dtype=[("m", int), ("n", int)]))

    @classmethod
    def from_file(cls, file_path: str = "modes"):
        with open(file_path, "r") as file:
            lines = file.readlines()

        eq_start = lines.index("Equilibrium modes:\n") + 3
        eig_start = lines.index("Eigenvector modes:\n") + 3
        eq_end = eig_start - 3

        equilibrium_modes = np.array(
            [tuple(map(int, line.strip().split())) for line in lines[eq_start:eq_end] if line.strip()],
            dtype=[("meq", int), ("neq", int)],
        )

        eigenvector_modes = np.array(
            [tuple(map(int, line.strip().split())) for line in lines[eig_start:]],
            dtype=[("m", int), ("n", int)],
        )

        return cls(equilibrium_modes=equilibrium_modes, eigenvector_modes=eigenvector_modes)


class FindGaps:
    """Methods for the automatic finding of continuum crossings and gaps."""

    modes: List[Mode]
    delta_freq: float
    delta_s: float
    normalized: bool = False
    ylims: List[float] = None
    vmec: Vmec

    ends_of_segments = {"m": [], "n": [], "s": [], "freq": []}
    matching_pairs = {"m1": [], "n1": [], "m2": [], "n2": [], "s_avg": [], "freq_avg": []}
    gap_widths = {"deltamn": [], "width": [], "s_avg": [], "freq_avg": [], "deltam": [], "deltan": [], "count": []}
    crossings = {"deltamn": [], "deltam": [], "deltan": [], "s_avg": [], "freq_avg": [], "width": []}

    calculated_ends_of_segments = False
    calculated_matching_pairs = False
    calculated_gap_widths = False

    def __init__(
        self,
        modes: List[Mode],
        normalized: bool = False,
        ylims: List[float] = None,
        vmec_file: str = "",
        delta_freq: float = 0.02,
        delta_s: float = None,
    ):
        self.modes = modes
        self.delta_freq = delta_freq
        self.delta_s = delta_s
        self.normalized = normalized
        self.ylims = ylims

        if normalized:
            if str == "":
                raise AttributeError("VMEC file required and not provided")
            else:
                self.vmec = Vmec(vmec_file)

        if self.delta_s is None:
            self.delta_s = np.ceil((modes[0].s[1] - modes[0].s[0]) * 100000) / 100000

    def calculate_ends_of_segments(self):
        self.ends_of_segments = {"m": [], "n": [], "s": [], "freq": []}
        ends_of_segments = {"m": [], "n": [], "s": [], "freq": []}

        if self.ylims is not None:
            y_max = self.ylims[1]
            y_min = self.ylims[0]

        for md in self.modes:
            segments = []

            if self.normalized:
                freqs = get_normalized_frequencies(self.vmec, md)
            else:
                freqs = md.freq

            if self.ylims is None or (np.min(freqs) <= y_max and np.max(freqs) >= y_min):
                for k in range(0, len(md.s)):
                    if len(segments) == 0:
                        new_segment = {"s": [md.s[k]], "freq": [freqs[k]]}
                        segments.append(new_segment)
                    else:
                        seg_found = False
                        for seg in segments:
                            if abs(seg["freq"][-1] - freqs[k]) <= self.delta_freq and abs(seg["s"][-1] - md.s[k]) <= self.delta_s:
                                if seg["s"][-1] != md.s[k]:
                                    seg["freq"].append(freqs[k])
                                    seg["s"].append(md.s[k])

                                seg_found = True
                                break

                        if not seg_found:
                            new_segment = {"s": [md.s[k]], "freq": [freqs[k]]}
                            segments.append(new_segment)

                for seg in segments:
                    if len(seg["s"]) > 1:
                        if seg["s"][0] >= 0.05:
                            ends_of_segments["m"].append(md.m)
                            ends_of_segments["n"].append(md.n)
                            ends_of_segments["s"].append(seg["s"][0])
                            ends_of_segments["freq"].append(seg["freq"][0])

                        if seg["s"][-1] <= 0.95:
                            ends_of_segments["m"].append(md.m)
                            ends_of_segments["n"].append(md.n)
                            ends_of_segments["s"].append(seg["s"][-1])
                            ends_of_segments["freq"].append(seg["freq"][-1])
                    elif len(seg["s"]) == 1 and seg["s"][0] >= 0.05 and seg["s"][0] <= 0.95:
                        ends_of_segments["m"].append(md.m)
                        ends_of_segments["n"].append(md.n)
                        ends_of_segments["s"].append(seg["s"][0])
                        ends_of_segments["freq"].append(seg["freq"][0])

        self.calculated_ends_of_segments = True
        self.ends_of_segments = ends_of_segments

    def calculate_matching_pairs(self):
        if not self.calculated_ends_of_segments:
            self.calculate_ends_of_segments()

        self.matching_pairs = {"m1": [], "n1": [], "m2": [], "n2": [], "s_avg": [], "freq_avg": []}
        matching_pairs = {"m1": [], "n1": [], "m2": [], "n2": [], "s_avg": [], "freq_avg": []}

        for i in range(0, len(self.ends_of_segments["m"]) - 1):
            m_test = self.ends_of_segments["m"][i]
            n_test = self.ends_of_segments["n"][i]
            s_test = self.ends_of_segments["s"][i]
            freq_test = self.ends_of_segments["freq"][i]

            for j in range(i + 1, len(self.ends_of_segments["m"])):
                if (
                    abs(freq_test - self.ends_of_segments["freq"][j]) <= self.delta_freq
                    and abs(s_test - self.ends_of_segments["s"][j]) <= self.delta_s
                    and (m_test != self.ends_of_segments["m"][j] or n_test != self.ends_of_segments["n"][j])
                ):
                    matching_pairs["m1"].append(m_test)
                    matching_pairs["n1"].append(n_test)

                    matching_pairs["m2"].append(self.ends_of_segments["m"][j])
                    matching_pairs["n2"].append(self.ends_of_segments["n"][j])

                    s_avg = (s_test + self.ends_of_segments["s"][j]) / 2.0
                    freq_avg = (freq_test + self.ends_of_segments["freq"][j]) / 2.0

                    matching_pairs["s_avg"].append(s_avg)
                    matching_pairs["freq_avg"].append(freq_avg)

        self.calculated_matching_pairs = True
        self.matching_pairs = matching_pairs

    def calculate_gap_widths(self):
        if not self.calculated_matching_pairs:
            self.calculate_matching_pairs()

        self.gap_widths = {"deltamn": [], "width": [], "s_avg": [], "freq_avg": [], "deltam": [], "deltan": [], "count": []}
        gap_widths = {"deltamn": [], "width": [], "s_avg": [], "freq_avg": [], "deltam": [], "deltan": [], "count": []}

        self.crossings = {
            "deltamn": [],
            "deltam": [],
            "deltan": [],
            "s_avg": [],
            "freq_avg": [],
            "width": [],
            "pair1_freq": [],
            "pair1_s": [],
            "pair2_freq": [],
            "pair2_s": [],
        }
        crossings = {
            "deltamn": [],
            "deltam": [],
            "deltan": [],
            "s_avg": [],
            "freq_avg": [],
            "width": [],
            "pair1_freq": [],
            "pair1_s": [],
            "pair2_freq": [],
            "pair2_s": [],
        }

        matching_pairs = self.matching_pairs.copy()

        i = 0
        while i < len(matching_pairs["m1"]) - 1:
            j_keeper = -1
            s_avg_diff = -1

            j = i + 1
            while j < len(matching_pairs["m1"]):
                if (
                    (
                        matching_pairs["m1"][i] == matching_pairs["m1"][j]
                        and matching_pairs["m2"][i] == matching_pairs["m2"][j]
                        and matching_pairs["n1"][i] == matching_pairs["n1"][j]
                        and matching_pairs["n2"][i] == matching_pairs["n2"][j]
                    )
                    or (
                        matching_pairs["m1"][i] == matching_pairs["m2"][j]
                        and matching_pairs["m2"][i] == matching_pairs["m1"][j]
                        and matching_pairs["n1"][i] == matching_pairs["n2"][j]
                        and matching_pairs["n2"][i] == matching_pairs["n1"][j]
                    )
                ):
                    s_avg_diff_inner = np.abs(matching_pairs["s_avg"][i] - matching_pairs["s_avg"][j])
                    if j_keeper == -1 or s_avg_diff_inner < s_avg_diff:
                        j_keeper = j
                        s_avg_diff = s_avg_diff_inner

                j += 1

            if j_keeper != -1:
                j = j_keeper

                s_avg = (matching_pairs["s_avg"][i] + matching_pairs["s_avg"][j]) / 2.0
                freq_avg = (matching_pairs["freq_avg"][i] + matching_pairs["freq_avg"][j]) / 2.0

                deltam_here = np.array([np.abs(matching_pairs["m1"][i] - matching_pairs["m2"][i])])
                deltan_here = np.array([matching_pairs["n1"][i] - matching_pairs["n2"][i]])

                deltam_sign = 1
                gap_modes = np.array([])

                if deltam_here[0] == 0 and deltan_here[0] != 0:
                    deltan_temp = np.abs(matching_pairs["n1"][i] - matching_pairs["n2"][i])
                    deltam_temp = np.abs(matching_pairs["m1"][i] - matching_pairs["m2"][i])

                    deltan_here = np.array([deltan_temp, -1 * deltan_temp])
                    deltam_here = np.array([deltam_temp, deltam_temp])

                    gap_modes = np.array(
                        [
                            "{},{}".format(deltam_here[0], deltan_here[0]),
                            "{},{}".format(deltam_here[1], deltan_here[1]),
                        ]
                    )
                else:
                    if matching_pairs["freq_avg"][i] > matching_pairs["freq_avg"][j]:
                        deltan_here = np.array([matching_pairs["n1"][i] - matching_pairs["n2"][i]])
                        deltam_sign = int((matching_pairs["m1"][i] - matching_pairs["m2"][i]) / np.abs(deltam_here[0]))
                    else:
                        deltan_here = np.array([matching_pairs["n1"][j] - matching_pairs["n2"][j]])
                        deltam_sign = int((matching_pairs["m1"][j] - matching_pairs["m2"][j]) / np.abs(deltam_here[0]))

                    deltan_here = deltam_sign * deltan_here
                    gap_modes = np.array(["{},{}".format(deltam_here[0], deltan_here[0])])

                gap_width = np.abs(matching_pairs["freq_avg"][i] - matching_pairs["freq_avg"][j])

                for iter in range(0, len(gap_modes)):
                    gap_mode = gap_modes[iter]
                    if gap_mode in gap_widths["deltamn"]:
                        ind = gap_widths["deltamn"].index(gap_mode)
                        prev_count = gap_widths["count"][ind]
                        gap_widths["width"][ind] = (gap_widths["width"][ind] * prev_count + gap_width) / (prev_count + 1.0)
                        gap_widths["freq_avg"][ind] = (gap_widths["freq_avg"][ind] * prev_count + freq_avg) / (prev_count + 1.0)
                        gap_widths["s_avg"][ind] = (gap_widths["s_avg"][ind] * prev_count + s_avg) / (prev_count + 1.0)
                        gap_widths["count"][ind] = prev_count + 1
                    else:
                        gap_widths["deltamn"].append(gap_mode)
                        gap_widths["width"].append(gap_width)
                        gap_widths["freq_avg"].append(freq_avg)
                        gap_widths["s_avg"].append(s_avg)
                        gap_widths["deltam"].append(deltam_here[iter])
                        gap_widths["deltan"].append(deltan_here[iter])
                        gap_widths["count"].append(1)

                    crossings["deltamn"].append(gap_mode)
                    crossings["s_avg"].append(s_avg)
                    crossings["freq_avg"].append(freq_avg)
                    crossings["deltam"].append(deltam_here[iter])
                    crossings["deltan"].append(deltan_here[iter])
                    crossings["width"].append(gap_width)
                    crossings["pair1_freq"].append(matching_pairs["freq_avg"][i])
                    crossings["pair1_s"].append(matching_pairs["s_avg"][i])
                    crossings["pair2_freq"].append(matching_pairs["freq_avg"][j])
                    crossings["pair2_s"].append(matching_pairs["s_avg"][j])

                del matching_pairs["m1"][j]
                del matching_pairs["n1"][j]
                del matching_pairs["m2"][j]
                del matching_pairs["n2"][j]
                del matching_pairs["s_avg"][j]
                del matching_pairs["freq_avg"][j]

                del matching_pairs["m1"][i]
                del matching_pairs["n1"][i]
                del matching_pairs["m2"][i]
                del matching_pairs["n2"][i]
                del matching_pairs["s_avg"][i]
                del matching_pairs["freq_avg"][i]

                i -= 1

            i += 1

        do_filter = True
        if do_filter:
            gap_widths_to_remove = []

            for i in range(0, len(gap_widths["deltamn"])):
                if gap_widths["count"][i] > 1:
                    freqs = []
                    indices = []
                    widths = []

                    for j in range(0, len(crossings["deltamn"])):
                        if crossings["deltamn"][j] == gap_widths["deltamn"][i]:
                            freqs.append(crossings["freq_avg"][j])
                            widths.append(crossings["width"][j])
                            indices.append(j)

                    freqs = np.array(freqs)
                    widths = np.array(widths)

                    temp = np.diff(np.sort(freqs))
                    min_difference = np.min(temp)

                    if min_difference > np.max(widths):
                        gap_widths_to_remove.append(i)

                        indices = np.array(indices)
                        indices = indices[::-1]

                        crossing_keys = list(crossings.keys())

                        for index in indices:
                            for key in crossing_keys:
                                del crossings[key][index]

            gap_widths_to_remove = np.array(gap_widths_to_remove)
            gap_widths_to_remove = gap_widths_to_remove[::-1]

            for i in gap_widths_to_remove:
                gap_width_keys = list(gap_widths.keys())
                for key in gap_width_keys:
                    del gap_widths[key][i]

            gap_widths_to_remove = []

            for i in range(0, len(gap_widths["deltamn"])):
                if gap_widths["count"][i] > 1:
                    indices = []
                    freqs = []
                    widths = []
                    s = []

                    for j in range(0, len(crossings["deltamn"])):
                        if crossings["deltamn"][j] == gap_widths["deltamn"][i]:
                            indices.append(j)
                            freqs.append(crossings["freq_avg"][j])
                            widths.append(crossings["width"][j])
                            s.append(crossings["s_avg"][j])

                    indices = np.array(indices)
                    freqs = np.array(freqs)
                    widths = np.array(widths)
                    s = np.array(s)

                    indices_to_remove = []

                    largest_freq_diff = 0
                    largest_gap_width = -1
                    ind_of_largest_diff = -1
                    motion = True
                    while motion and len(freqs) > 1:
                        largest_gap_width = np.max(widths)
                        largest_freq_diff = 0
                        counter = 0
                        temp_ind = -1
                        for k in indices:
                            freqs_test = np.delete(freqs, counter)

                            if np.min(np.abs(freqs_test - crossings["freq_avg"][k])) > largest_freq_diff:
                                largest_freq_diff = np.min(np.abs(freqs_test - crossings["freq_avg"][k]))
                                ind_of_largest_diff = k
                                temp_ind = counter
                            counter += 1

                        if largest_freq_diff > largest_gap_width:
                            motion = True
                            indices_to_remove.append(ind_of_largest_diff)

                            indices = np.delete(indices, temp_ind)
                            freqs = np.delete(freqs, temp_ind)
                            widths = np.delete(widths, temp_ind)
                            s = np.delete(s, temp_ind)
                        else:
                            motion = False

                    if len(freqs) == 0:
                        gap_widths_to_remove.append(i)

                    gap_widths["count"][i] = len(freqs)
                    gap_widths["width"][i] = np.sum(widths) / len(widths)
                    gap_widths["freq_avg"][i] = np.sum(freqs) / len(freqs)
                    gap_widths["s_avg"][i] = np.sum(s) / len(s)

                    indices_to_remove = np.sort(indices_to_remove)
                    indices_to_remove = indices_to_remove[::-1]

                    crossing_keys = list(crossings.keys())

                    for index in indices_to_remove:
                        for key in crossing_keys:
                            del crossings[key][index]

            gap_widths_to_remove = np.array(gap_widths_to_remove)
            gap_widths_to_remove = gap_widths_to_remove[::-1]

            for i in gap_widths_to_remove:
                gap_width_keys = list(gap_widths.keys())
                for key in gap_width_keys:
                    del gap_widths[key][i]

            for i in range(0, len(gap_widths["deltamn"])):
                if gap_widths["count"][i] > 1:
                    indices = []
                    freqs = []
                    widths = []
                    s = []

                    for j in range(0, len(crossings["deltamn"])):
                        if crossings["deltamn"][j] == gap_widths["deltamn"][i]:
                            indices.append(j)
                            freqs.append(crossings["freq_avg"][j])
                            widths.append(crossings["width"][j])
                            s.append(crossings["s_avg"][j])

                    indices = np.array(indices)
                    freqs = np.array(freqs)
                    widths = np.array(widths)
                    s = np.array(s)

                    indices_to_remove = []

                    inn = 0
                    while inn < len(indices):
                        if widths[inn] > 3.0:
                            indices_to_remove.append(indices[inn])

                            indices = np.delete(indices, inn)
                            freqs = np.delete(freqs, inn)
                            widths = np.delete(widths, inn)
                            s = np.delete(s, inn)

                            inn -= 1

                        inn += 1

                    gap_widths["count"][i] = len(freqs)
                    gap_widths["width"][i] = np.sum(widths) / len(widths)
                    gap_widths["freq_avg"][i] = np.sum(freqs) / len(freqs)
                    gap_widths["s_avg"][i] = np.sum(s) / len(s)

                    indices_to_remove = np.sort(indices_to_remove)
                    indices_to_remove = indices_to_remove[::-1]

                    crossing_keys = list(crossings.keys())

                    for index in indices_to_remove:
                        for key in crossing_keys:
                            del crossings[key][index]

            required_min_count = 3
            gap_widths_to_remove = []

            for i in range(0, len(gap_widths["deltamn"])):
                if gap_widths["count"][i] < required_min_count:
                    indices = []

                    for j in range(0, len(crossings["deltamn"])):
                        if crossings["deltamn"][j] == gap_widths["deltamn"][i]:
                            indices.append(j)

                    indices.sort(reverse=True)

                    for index in indices:
                        for key in crossing_keys:
                            del crossings[key][index]

                    gap_widths_to_remove.append(i)

            gap_widths_to_remove = np.array(gap_widths_to_remove)
            gap_widths_to_remove = gap_widths_to_remove[::-1]

            for i in gap_widths_to_remove:
                gap_width_keys = list(gap_widths.keys())
                for key in gap_width_keys:
                    del gap_widths[key][i]

        self.calculated_gap_widths = True
        self.crossings = crossings
        self.gap_widths = gap_widths

    def get_ends_of_segments(self):
        if not self.calculated_ends_of_segments:
            self.calculate_ends_of_segments()
        return self.ends_of_segments

    def get_matching_pairs(self):
        if not self.calculated_matching_pairs:
            self.calculate_matching_pairs()
        return self.matching_pairs

    def get_crossings(self):
        if not self.calculated_gap_widths:
            self.calculate_gap_widths()
        return self.crossings

    def get_gap_widths(self):
        if not self.calculated_gap_widths:
            self.calculate_gap_widths()
        return self.gap_widths

    def print_gap_widths(self):
        gap_widths = self.get_gap_widths()

        print("{:<18}||   {:<22}||    {:<20}".format("delta m, delta n", "Average Width [kHz]", "Average Freq [kHz]"))

        n = 20 + 6 + 25 + 20
        line = ""

        for i in range(0, n):
            line = "{}{}".format(line, "=")

        print(line)

        for i in range(0, len(gap_widths["deltamn"])):
            print("{:<20}   {:<25}   {:<20}".format(gap_widths["deltamn"][i], gap_widths["width"][i], gap_widths["freq_avg"][i]))


class AlfvenSpecData(np.ndarray):
    """Subclass of numpy.ndarray with dtype specific to STELLGAP output in alfven_spec files."""

    def __new__(cls, filenames: List[str]):
        if not filenames:
            raise ValueError("No filenames provided")

        data = np.vstack(
            [
                np.loadtxt(
                    fname,
                    dtype=[("s", float), ("ar", float), ("ai", float), ("beta", float), ("m", int), ("n", int)],
                )
                for fname in filenames
            ]
        )
        obj = np.asarray(data).view(cls)
        return obj

    @classmethod
    def from_dir(cls, directory: str):
        files = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.startswith("alfven_spec")]
        if not files:
            raise ValueError(f"No alfven_spec files found in the directory {directory}")
        return cls(files)

    def nonzero_beta(self):
        return self[self["beta"] != 0]

    def sort_by_s(self):
        return self[np.argsort(self["s"])]

    def get_modes(self) -> List[Mode]:
        data = self.nonzero_beta()
        modes = [
            Mode(
                n=n,
                m=m,
                s=(filtered_data := np.sort(data[(data["n"] == n) & (data["m"] == m)], order="s"))["s"],
                freq=np.sqrt(np.abs(filtered_data["ar"] / filtered_data["beta"])),
            )
            for n, m in {(a["n"], a["m"]) for a in data}
        ]
        return modes

    def condition_number(self):
        data = self.nonzero_beta().sort_by_s()
        s = np.unique(data["s"])
        condition_numbers = np.array(
            [
                np.max(np.abs(data[data["s"] == s_]["ar"])) / np.min(np.abs(data[data["s"] == s_]["ar"]))
                if np.min(np.abs(data[data["s"] == s_]["ar"])) != 0
                else np.inf
                for s_ in s
            ]
        )
        return s, condition_numbers


def data_from_dir(directory: str) -> AlfvenSpecData:
    files = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.startswith("alfven_spec")]
    assert len(files) > 0, f"No alfven_spec files found in the dir {directory}"
    return AlfvenSpecData(files)


def continuum_from_dir(directory: str) -> go.Figure:
    modes = data_from_dir(directory).get_modes()
    assert len(modes) > 0, "No Alfven mode data found the AlvfenSpecData"
    fig = plot_continuum(modes)
    return fig


def get_omega_A(vmec: Vmec, s_vals: List):
    m_proton = 1.67e-27
    mu_0 = 1.26e-6

    ion_to_proton_mass_ratio = 2
    n_0 = 4.8e20

    B_0 = vmec.wout.volavgB

    bvco = vmec.wout.bvco
    bvco = bvco[bvco != 0]

    G_0 = bvco[0]

    rho = n_0 * m_proton * ion_to_proton_mass_ratio * (1.0 - 1.0 * np.power(s_vals, 4))

    omega_A = [0.0] * len(s_vals)

    for i in range(0, len(np.sqrt(rho))):
        if np.sqrt(rho)[i] == 0.0:
            omega_A[i] = 0
        else:
            freq_A = B_0**2.0 / (np.sqrt(mu_0 * rho[i]) * G_0)
            freq_A = freq_A * pow(10, -3)
            freq_A = freq_A / (2 * np.pi)
            omega_A[i] = freq_A

    return omega_A


def get_normalized_frequencies(vmec: Vmec, md: Mode):
    m_proton = 1.67e-27
    mu_0 = 1.26e-6

    ion_to_proton_mass_ratio = 2
    n_0 = 4.8e20

    B_0 = vmec.wout.volavgB

    bvco = vmec.wout.bvco
    bvco = bvco[bvco != 0]

    G_0 = bvco[0]

    s_vals = md.s

    if 0 in s_vals:
        print("Zero-value in flux surfaces, s")

    rho = n_0 * m_proton * ion_to_proton_mass_ratio * (1.0 - 1.0 * np.power(s_vals, 4))

    freq_normalized = np.copy(md.freq)

    for i in range(0, len(np.sqrt(rho))):
        if np.sqrt(rho)[i] == 0.0:
            freq_normalized[i] = 0
        else:
            freq_A = B_0**2.0 / (np.sqrt(mu_0 * rho[i]) * G_0)
            freq_A = freq_A * pow(10, -3)
            freq_A = freq_A / (2 * np.pi)
            freq_normalized[i] = abs(md.freq[i] / freq_A)

    return freq_normalized


def add_crossing_labels(
    figure,
    modes: List[Mode],
    delta_freq: float = 0.02,
    delta_s: float = None,
    vmec_file: str = "",
    normalized: bool = False,
    ylims: List[float] = None,
    specific_crossing_labels: List[List[float]] = None,
):
    fig = figure

    finder = FindGaps(modes=modes, delta_freq=delta_freq, delta_s=delta_s, vmec_file=vmec_file, normalized=normalized, ylims=ylims)

    print(finder.get_crossings())

    crossings = finder.get_crossings()

    for i in range(0, len(crossings["deltamn"])):
        if specific_crossing_labels is None:
            fig.add_trace(
                go.Scatter(
                    x=[crossings["s_avg"][i]],
                    y=[crossings["freq_avg"][i]],
                    mode="markers+text",
                    name="Markers and Text",
                    text=[crossings["deltamn"][i]],
                    textposition="bottom center",
                )
            )
        else:
            specific_label = [crossings["deltam"][i], crossings["deltan"][i]]
            if specific_label in specific_crossing_labels:
                fig.add_trace(
                    go.Scatter(
                        x=[crossings["s_avg"][i]],
                        y=[crossings["freq_avg"][i]],
                        mode="markers+text",
                        name="Markers and Text",
                        text=[crossings["deltamn"][i]],
                        textposition="bottom center",
                    )
                )

    return fig


def plot_gap_width_grid(
    modes: List[Mode],
    delta_freq: float = 0.02,
    delta_s: float = None,
    vmec_file: str = "",
    normalized: bool = False,
    ylims: List[float] = None,
    slims: List[float] = None,
    title: str = None,
    select_gaps: List[List[float]] = None,
    normalize_by: float = None,
    plot_xlims: List[float] = None,
    plot_ylims: List[float] = None,
):
    vmec = Vmec(vmec_file)

    finder = FindGaps(modes=modes, delta_freq=delta_freq, delta_s=delta_s, vmec_file=vmec_file, normalized=normalized, ylims=ylims)

    if slims is None:
        gap_widths = finder.get_gap_widths()
    else:
        crossings = finder.get_crossings()

        gap_widths = {"deltamn": [], "width": [], "s_avg": [], "freq_avg": [], "deltam": [], "deltan": [], "count": []}

        print("slims exists: {}, {}".format(slims[0], slims[1]))

        print_widths = crossings.copy()
        sort_key = "width"
        sorted_indices = sorted(range(len(print_widths[sort_key])), key=lambda i: print_widths[sort_key][i])

        sorted_data = {key: [values[i] for i in sorted_indices] for key, values in print_widths.items()}

        print("Crossings sorted by gap width")
        print(sorted_data)

        for iter in range(0, len(crossings["deltam"])):
            if crossings["s_avg"][iter] >= slims[0] and crossings["s_avg"][iter] <= slims[1] and crossings["freq_avg"][iter] < ylims[1]:
                crossing = crossings["deltamn"][iter]
                if crossing in gap_widths["deltamn"]:
                    ind = gap_widths["deltamn"].index(crossing)

                    prev_count = gap_widths["count"][ind]
                    gap_widths["width"][ind] = (gap_widths["width"][ind] * prev_count + crossings["width"][iter]) / (prev_count + 1.0)
                    gap_widths["freq_avg"][ind] = (gap_widths["freq_avg"][ind] * prev_count + crossings["freq_avg"][iter]) / (prev_count + 1.0)
                    gap_widths["s_avg"][ind] = (gap_widths["s_avg"][ind] * prev_count + crossings["s_avg"][iter]) / (prev_count + 1.0)
                    gap_widths["count"][ind] = prev_count + 1
                else:
                    gap_widths["deltamn"].append(crossing)
                    gap_widths["width"].append(crossings["width"][iter])
                    gap_widths["freq_avg"].append(crossings["freq_avg"][iter])
                    gap_widths["s_avg"].append(crossings["s_avg"][iter])
                    gap_widths["deltam"].append(crossings["deltam"][iter])
                    gap_widths["deltan"].append(crossings["deltan"][iter])
                    gap_widths["count"].append(1)

    data = np.zeros(
        (
            (np.max(gap_widths["deltam"]) - np.min(gap_widths["deltam"]) + 1),
            (np.max(gap_widths["deltan"]) - np.min(gap_widths["deltan"]) + 1),
        )
    )

    for i in range(0, len(gap_widths["deltamn"])):
        nfp = vmec.wout.nfp
        indm = gap_widths["deltam"][i] - np.min(gap_widths["deltam"])
        indn = int(gap_widths["deltan"][i] / nfp) - (np.min(gap_widths["deltan"]))

        if gap_widths["deltam"][i] == 1 and gap_widths["deltan"][i] == 0:
            print("width: {}".format(gap_widths["width"][i]))

        gap_width_test = gap_widths["deltan"][i] / nfp

        if gap_width_test - int(gap_width_test) != 0:
            raise AssertionError("Not a multiple of Nfp")

        if data[indm, indn] != 0:
            raise AssertionError("Overriding a gap width value with another")

        if select_gaps is None:
            data[indm, indn] = gap_widths["width"][i]
        else:
            this_gap = np.abs([gap_widths["deltam"][i], gap_widths["deltan"][i]]).tolist()
            check_gaps = np.abs(select_gaps).tolist()

            if this_gap in check_gaps:
                data[indm, indn] = gap_widths["width"][i]

    data = np.array(data)
    data_unnormalized = data

    maximum_value = np.max(data)
    print("Maximum width: {}".format(maximum_value))
    print("Maximum value: {}".format(maximum_value))

    plt_fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.imshow(data, cmap="plasma", aspect="auto")

    data3 = data

    for i in range(data3.shape[0]):
        for j in range(data3.shape[1]):
            ax.text(j, i, f"{data3[i, j]:.2f}", ha="center", va="center", color="white", fontsize=10)

    ax.set_yticks(np.arange(0, np.max(gap_widths["deltam"]) - np.min(gap_widths["deltam"]) + 1))
    ax.set_xticks(np.arange(0, np.max(gap_widths["deltan"]) - np.min(gap_widths["deltan"]) + 1))
    ax.set_yticklabels(np.arange(np.min(gap_widths["deltam"]), np.max(gap_widths["deltam"]) + 1))
    ax.set_xticklabels(np.arange(np.min(gap_widths["deltan"]), np.max(gap_widths["deltan"]) + 1))

    if plot_xlims is not None:
        ax.set_xlim(plot_xlims)

    if plot_ylims is not None:
        ax.set_ylim(plot_ylims)

    ax.invert_yaxis()
    ax.set_xlabel(r"$\delta n / N_{fp}$")
    ax.set_ylabel(r"$\delta m$")
    if title is not None:
        plt.title(title)

    if normalize_by is None:
        clbar = plt.colorbar(c, ax=ax)
        clbar.set_label("Normalized $\Delta \omega$")
    else:
        ax2 = ax.twinx()
        ax2.set_ylabel(r"Stellgap $\Delta \overline{\omega}$")
        ax2.set_yticks([])

    plt.show()

    return [plt_fig, data_unnormalized]


def plot_gap_width_grid_not_divided_by_deltan(
    modes: List[Mode],
    delta_freq: float = 0.02,
    delta_s: float = None,
    vmec_file: str = "",
    normalized: bool = False,
    ylims: List[float] = None,
    title: str = None,
):
    finder = FindGaps(modes=modes, delta_freq=delta_freq, delta_s=delta_s, vmec_file=vmec_file, normalized=normalized, ylims=ylims)
    gap_widths = finder.get_gap_widths()

    vmec = Vmec(vmec_file)

    data = np.zeros(
        (
            (np.max(gap_widths["deltam"]) - np.min(gap_widths["deltam"]) + 1),
            (np.max(gap_widths["deltan"]) - np.min(gap_widths["deltan"]) + 1),
        )
    )

    modes_delta = np.empty(
        [
            (np.max(gap_widths["deltam"]) - np.min(gap_widths["deltam"]) + 1),
            (np.max(gap_widths["deltan"]) - np.min(gap_widths["deltan"]) + 1),
        ],
        dtype=str,
    )

    for i in range(0, len(gap_widths["deltamn"])):
        indm = gap_widths["deltam"][i] - np.min(gap_widths["deltam"])
        indn = int(gap_widths["deltan"][i]) - (np.min(gap_widths["deltan"]))

        if data[indm, indn] != 0:
            raise AssertionError("Overriding a gap width value with another")

        data[indm, indn] = gap_widths["width"][i]
        modes_delta = "{},{}".format(gap_widths["deltam"][i], gap_widths["deltan"][i])

    data = np.array(data)
    data_unnormalized = data

    maximum_value = np.max(data)
    data = data / maximum_value

    plt_fig, ax = plt.subplots(figsize=(10, 5))
    c = ax.imshow(data, cmap="plasma", aspect="auto")

    ax.set_yticks(np.arange(0, np.max(gap_widths["deltam"]) - np.min(gap_widths["deltam"]) + 1))
    ax.set_xticks(np.arange(0, np.max(gap_widths["deltan"]) - np.min(gap_widths["deltan"]) + 1))
    ax.set_yticklabels(np.arange(np.min(gap_widths["deltam"]), np.max(gap_widths["deltam"]) + 1))
    ax.set_xticklabels(np.arange(np.min(gap_widths["deltan"]), np.max(gap_widths["deltan"]) + 1))

    ax.invert_yaxis()
    ax.set_xlabel(r"$\delta n$")
    ax.set_ylabel(r"$\delta m$")
    if title is not None:
        plt.title(title)

    clbar = plt.colorbar(c, ax=ax)
    clbar.set_label("$\Delta \omega$ [kHz]")

    plt.show()

    return [plt_fig, data_unnormalized]


def plot_histogram(modes: List[Mode], vmec_file: str = "", ylims: List[float] = None, normalized: bool = False, save_name: str = None):
    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            vmec = Vmec(vmec_file)

    freq_all = []

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    for md in modes:
        if normalized:
            freqs = get_normalized_frequencies(vmec, md)
        else:
            freqs = md.freq

        if ylims is None:
            freqs_here = freqs
        else:
            freqs_here = freqs[freqs <= y_max]
            freqs_here = freqs[freqs >= y_min]

        for freq_here in freqs_here:
            freq_all.append(freq_here)

    if ylims is None:
        bins = np.linspace(np.min(freq_all), np.max(freq_all), 50)
    else:
        bins = np.linspace(ylims[0], ylims[1], 50)

    plt.figure()
    plt.hist(freq_all, bins=bins, density=True, orientation="horizontal")
    plt.ylabel("$\omega / \omega_A$")
    plt.title("QH Ku-Boozer")

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    if save_name is not None:
        plt.savefig(save_name, dpi=2000)


def plot_2_histograms(
    modes: List[Mode],
    modes2: List[Mode],
    vmec_file: str = "",
    ylims: List[float] = None,
    normalized: bool = False,
    save_name: str = None,
    label_name: str = None,
):
    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            vmec = Vmec(vmec_file)

    freq_all = []
    freq_all2 = []

    number_of_bins = 25

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    for md in modes:
        if normalized:
            freqs = get_normalized_frequencies(vmec, md)
        else:
            freqs = md.freq

        if ylims is None:
            freqs_here = freqs
        else:
            freqs_here = freqs[freqs <= y_max]
            freqs_here = freqs[freqs >= y_min]

        for freq_here in freqs_here:
            freq_all.append(freq_here)

    for md in modes2:
        if normalized:
            freqs2 = get_normalized_frequencies(vmec, md)
        else:
            freqs2 = md.freq

        if ylims is None:
            freqs_here2 = freqs2
        else:
            freqs_here2 = freqs2[freqs2 <= y_max]
            freqs_here2 = freqs2[freqs2 >= y_min]

        for freq_here2 in freqs_here2:
            freq_all2.append(freq_here2)

    if ylims is None:
        bins = np.linspace(np.min(freq_all), np.max(freq_all), number_of_bins)
    else:
        bins = np.linspace(ylims[0], ylims[1], number_of_bins)

    counts1, _ = np.histogram(freq_all, bins=bins)
    counts2, _ = np.histogram(freq_all2, bins=bins)

    norm_factor = counts2.max()
    counts1_norm = counts1 / norm_factor
    counts2_norm = counts2 / norm_factor

    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2

    plt.figure()
    plt.barh(bin_centers, counts1_norm, height=bin_widths, alpha=0.5, label=label_name)
    plt.barh(bin_centers, counts2_norm, height=bin_widths, hatch="//", facecolor="none", edgecolor="black", label="block")

    plt.xlim(0, np.max(counts2_norm) * 1.55)
    plt.ylim(ylims[0], ylims[1])
    plt.ylabel("$\overline{\omega}$")
    plt.legend(loc="upper right")

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    if save_name is not None:
        plt.savefig(save_name, dpi=500)


def plot_2_KDEs_orig(
    modes: List[Mode],
    modes2: List[Mode],
    vmec_file: str = "",
    ylims: List[float] = None,
    normalized: bool = False,
    save_name: str = None,
    label_name: str = None,
):
    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            vmec = Vmec(vmec_file)

    freq_all = []
    freq_all2 = []
    number_of_bins = 25

    if ylims is not None:
        y_max = ylims[1] + 1.0
        y_min = ylims[0]

    for md in modes:
        if normalized:
            freqs = get_normalized_frequencies(vmec, md)
        else:
            freqs = md.freq

        if ylims is None:
            freqs_here = freqs
        else:
            freqs_here = freqs[freqs <= y_max]
            freqs_here = freqs_here[freqs_here >= y_min]

        for freq_here in freqs_here:
            freq_all.append(freq_here)

    for md in modes2:
        if normalized:
            freqs2 = get_normalized_frequencies(vmec, md)
        else:
            freqs2 = md.freq

        if ylims is None:
            freqs_here2 = freqs2
        else:
            freqs_here2 = freqs2[freqs2 <= y_max]
            freqs_here2 = freqs_here2[freqs_here2 >= y_min]

        for freq_here2 in freqs_here2:
            freq_all2.append(freq_here2)

    if ylims is None:
        bins = np.linspace(np.min(freq_all), np.max(freq_all), number_of_bins)
    else:
        bins = np.linspace(ylims[0], ylims[1], number_of_bins)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure()
    sns.kdeplot(y=freq_all, label=label_name, fill=True, color="#3D74B6", alpha=0.4, bw_adjust=0.4)
    sns.kdeplot(y=freq_all2, label="block", fill=True, color="#DC3C22", alpha=0.4, bw_adjust=0.4)

    plt.ylabel("$\overline{\omega}$")
    plt.xlabel("amplitude")
    plt.legend(loc="upper right")
    plt.xlim([0, 0.6])
    plt.ylim([0, 3])

    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    if save_name is not None:
        plt.savefig(save_name, dpi=500)


def plot_2_KDEs(
    modes: List[Mode],
    modes2: List[Mode],
    vmec_file: str = "",
    ylims: List[float] = None,
    normalized: bool = False,
    save_name: str = None,
    label_name: str = None,
):
    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            vmec = Vmec(vmec_file)

    freq_all = []
    freq_all2 = []
    number_of_bins = 25

    if ylims is not None:
        y_max = ylims[1] + 1.0
        y_min = ylims[0]

    for md in modes:
        if normalized:
            freqs = get_normalized_frequencies(vmec, md)
        else:
            freqs = md.freq

        if ylims is None:
            freqs_here = freqs
        else:
            freqs_here = freqs[freqs <= y_max]
            freqs_here = freqs_here[freqs_here >= y_min]

        for freq_here in freqs_here:
            freq_all.append(freq_here)

    for md in modes2:
        if normalized:
            freqs2 = get_normalized_frequencies(vmec, md)
        else:
            freqs2 = md.freq

        if ylims is None:
            freqs_here2 = freqs2
        else:
            freqs_here2 = freqs2[freqs2 <= y_max]
            freqs_here2 = freqs_here2[freqs_here2 >= y_min]

        for freq_here2 in freqs_here2:
            freq_all2.append(freq_here2)

    if ylims is None:
        bins = np.linspace(np.min(freq_all), np.max(freq_all), number_of_bins)
    else:
        bins = np.linspace(ylims[0], ylims[1], number_of_bins)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure()

    ax = sns.kdeplot(y=freq_all, label=label_name, fill=False, color="#3D74B6", alpha=0.4, bw_adjust=0.4)
    line1 = ax.lines[0]
    density_vals1 = line1.get_xdata()
    y_vals1 = line1.get_ydata()

    ax = sns.kdeplot(y=freq_all2, label="block", fill=False, color="#DC3C22", alpha=0.4, bw_adjust=0.4)
    plt.ylim([0, 3])

    line2 = ax.lines[1]
    density_vals2 = line2.get_xdata()

    fig = plt.figure()
    plt.plot((density_vals2 - density_vals1), y_vals1)
    plt.ylim([0, 3])
    plt.show()


def plot_continuum(
    modes: List[Mode],
    show_legend: bool = False,
    normalized: bool = False,
    ylims: List[float] = None,
    vmec_file: str = "",
    show_crossing_labels: bool = False,
    specific_crossing_labels: List[List[float]] = None,
    delta_freq: float = 0.02,
    delta_s: float = None,
    title: str = "",
) -> go.Figure:
    fig = None

    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            fig = plot_continuum_normalized(
                vmec_file=vmec_file,
                modes=modes,
                show_legend=show_legend,
                ylims=ylims,
                show_crossing_labels=show_crossing_labels,
                specific_crossing_labels=specific_crossing_labels,
                delta_freq=delta_freq,
                delta_s=delta_s,
                title=title,
            )
    else:
        fig = go.Figure()
        for md in modes:
            fig.add_trace(
                go.Scatter(
                    x=md.s,
                    y=md.freq,
                    mode="markers",
                    name=f"m={md.m}, n={md.n}",
                    marker=dict(size=3),
                    line=dict(width=0.5),
                )
            )

        if ylims is None:
            fig.update_layout(
                autosize=True,
                title=title,
                xaxis_title=r"$s$",
                yaxis_title=r"$\text{Frequency }\omega\text{ [kHz]}$",
                xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
                legend=dict(title=r"$\text{Mode: }$", yanchor="top", y=1.4, xanchor="center", x=0.5, orientation="h"),
                showlegend=show_legend,
            )
        else:
            fig.update_layout(
                autosize=True,
                title=title,
                xaxis_title=r"$\text{Normalized flux }s$",
                yaxis_title=r"$\text{Frequency }\omega\text{ [kHz]}$",
                xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
                yaxis=dict(range=ylims),
                legend=dict(title=r"$\text{Mode: }$", yanchor="top", y=1.4, xanchor="center", x=0.5, orientation="h"),
                showlegend=show_legend,
            )

        if show_crossing_labels:
            fig = add_crossing_labels(
                fig,
                modes=modes,
                vmec_file=vmec_file,
                normalized=False,
                ylims=ylims,
                delta_freq=delta_freq,
                delta_s=delta_s,
                specific_crossing_labels=specific_crossing_labels,
            )

    return fig


def plot_continuum_normalized_by_prop_direction(
    vmec_file: str,
    modes: List[Mode],
    show_legend: bool = False,
    ylims: List[float] = None,
    show_crossing_labels: bool = False,
    specific_crossing_labels: List[List[float]] = None,
    delta_freq: float = 0.02,
    delta_s: float = None,
    title: str = "",
) -> go.Figure:
    vmec = Vmec(vmec_file)
    iotas = vmec.wout.iotas
    delta_s = 1.0 / (len(iotas) - 1)

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    fig = go.Figure()
    for md in modes:
        freq_normalized = get_normalized_frequencies(vmec, md)

        if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
            marker_colors = [""] * len(freq_normalized)

            for i in range(0, len(freq_normalized)):
                s_ind = int(np.round(md.s[i] / delta_s))
                test_val = iotas[s_ind] * md.m - md.n
                if test_val >= 0:
                    marker_colors[i] = "blue"
                else:
                    marker_colors[i] = "red"

            fig.add_trace(
                go.Scatter(
                    x=md.s,
                    y=freq_normalized,
                    mode="markers",
                    name=f"m={md.m}, n={md.n}",
                    marker=dict(size=3),
                    line=dict(width=0.5),
                    marker_color=marker_colors,
                )
            )

    if show_crossing_labels:
        fig = add_crossing_labels(
            fig,
            modes=modes,
            vmec_file=vmec_file,
            normalized=True,
            ylims=ylims,
            specific_crossing_labels=specific_crossing_labels,
            delta_freq=delta_freq,
            delta_s=delta_s,
        )

    if ylims is None:
        fig.update_layout(
            autosize=True,
            title=title,
            xaxis_title=r"$s$",
            yaxis_title=r"$\overline{\omega}$",
            xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
            legend=dict(title=r"$\text{Modes: }$"),
            showlegend=show_legend,
        )
    else:
        fig.update_layout(
            autosize=True,
            title=title,
            xaxis_title=r"$s$",
            yaxis_title=r"$\overline{\omega}$",
            xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
            yaxis=dict(range=[ylims[0], ylims[1]]),
            legend=dict(title=r"$\text{Modes: }$"),
            showlegend=show_legend,
        )

    return fig


def plot_continuum_normalized(
    vmec_file: str,
    modes: List[Mode],
    show_legend: bool = False,
    ylims: List[float] = None,
    show_crossing_labels: bool = False,
    specific_crossing_labels: List[List[float]] = None,
    delta_freq: float = 0.02,
    delta_s: float = None,
    title: str = "",
) -> go.Figure:
    vmec = Vmec(vmec_file)

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    fig = go.Figure()
    for md in modes:
        freq_normalized = get_normalized_frequencies(vmec, md)

        if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
            fig.add_trace(
                go.Scatter(
                    x=md.s,
                    y=freq_normalized,
                    mode="markers",
                    name=f"m={md.m}, n={md.n}",
                    marker=dict(size=3),
                    line=dict(width=0.5),
                )
            )

    if show_crossing_labels:
        fig = add_crossing_labels(
            fig,
            modes=modes,
            vmec_file=vmec_file,
            normalized=True,
            ylims=ylims,
            specific_crossing_labels=specific_crossing_labels,
            delta_freq=delta_freq,
            delta_s=delta_s,
        )

    if ylims is None:
        fig.update_layout(
            autosize=True,
            title=title,
            xaxis_title=r"$s$",
            yaxis_title=r"$\text{Normalized frequency }\omega / \omega_A$",
            xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
            legend=dict(title=r"$\text{Modes: }$"),
            showlegend=show_legend,
        )
    else:
        fig.update_layout(
            autosize=True,
            title=title,
            xaxis_title=r"$s$",
            yaxis_title=r"$\overline{\omega}$",
            xaxis=dict(range=[np.min([np.min(md.s) for md in modes]), np.max([np.max(md.s) for md in modes])]),
            yaxis=dict(range=[ylims[0], ylims[1]]),
            legend=dict(title=r"$\text{Modes: }$"),
            showlegend=show_legend,
        )

    return fig


def plot_continuum_matplotlib(
    modes: List[Mode],
    normalized: bool = False,
    ylims: List[float] = None,
    xlims: List[float] = None,
    vmec_file: str = "",
    insert_inset=False,
    inset_xlims=[0, 1],
    N_hel: float = None,
    gap_labels: dict = None,
    **kwargs,
):
    vmec = Vmec(vmec_file)
    figure, ax = plt.subplots(1, figsize=(7, 4))

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    if xlims is not None:
        x_max = xlims[1]
        x_min = xlims[0]
    else:
        x_max = 1.0
        x_min = modes[0].s[0]
        xlims = [x_min, x_max]

    if insert_inset:
        x_bound = [0.03, 0.3]
        y_bound = [0.4, 1.4]
        box_location = [1.25, 0.15, 0.7, 0.7]

        ax_inset = ax.inset_axes(box_location, xlim=(x_bound[0], x_bound[1]), ylim=(y_bound[0], y_bound[1]))
        yticks = ax_inset.get_yticks()
        ax_inset.set_yticks(yticks[1:-1])

    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            for md in modes:
                freq_normalized = get_normalized_frequencies(vmec, md)

                if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
                    ax.scatter(md.s, freq_normalized, s=1)

                    if insert_inset:
                        ax_inset.scatter(md.s, freq_normalized, s=1)

        ax.set_ylabel(r"$\overline{\omega}$")
    else:
        for md in modes:
            freqs = md.freq

            if ylims is None or (np.min(freqs) <= y_max and np.max(freqs) >= y_min):
                ax.scatter(md.s, freqs, s=1)

                if insert_inset:
                    ax_inset.scatter(md.s, freqs, s=1)

        ax.set_ylabel(r"Frequency $\omega [kHz]$")

    if N_hel is not None:
        iota_vals = vmec.wout.iotas
        if iota_vals[5] < 0:
            iota_vals = -1 * iota_vals

        iota_s = np.abs(iota_vals - N_hel) / 2.0
        s_vals = np.linspace(0, 1, len(iota_s))

        ax.plot(s_vals, iota_s, linestyle="dashed", linewidth=3, color="black")
        ax.text(s_vals[-1] + 0.01, iota_s[-1], "|\iota—N|/2", fontsize=12, va="center", ha="left", style="italic", fontweight="bold")

    if ylims is not None:
        ax.set_ylim(y_min, y_max)

    if xlims is not None:
        ax.set_xlim(x_min, x_max)

    ax.set_xlabel(r"$s$")

    if insert_inset:
        from matplotlib.patches import ConnectionPatch, Rectangle

        inset_rect = Rectangle(
            (x_bound[0], y_bound[0]),
            x_bound[1] - x_bound[0],
            y_bound[1] - y_bound[0],
            facecolor=(0.5, 0.5, 0.5, 0.40),
            edgecolor="black",
            linewidth=2,
            zorder=100,
        )

        ax.add_patch(inset_rect)

        for spine in ax_inset.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)

        connector = ConnectionPatch(
            xyA=(x_bound[1], y_bound[1]),
            coordsA=ax.transData,
            xyB=(0, 1),
            coordsB=ax_inset.transAxes,
            color="black",
            linewidth=2,
            zorder=100,
        )
        figure.add_artist(connector)

        connector = ConnectionPatch(
            xyA=(x_bound[1], y_bound[0]),
            coordsA=ax.transData,
            xyB=(0, 0),
            coordsB=ax_inset.transAxes,
            color="black",
            linewidth=2,
            zorder=100,
        )
        figure.add_artist(connector)

    if gap_labels is not None:
        for i in range(0, len(gap_labels["s"])):
            label = ""
            delta_m = gap_labels["m"][i]
            delta_n = gap_labels["n"][i]

            if delta_n == 0:
                if delta_m == 0:
                    label = "GAE"
                elif abs(delta_m) == 1:
                    label = "TAE"
                elif abs(delta_m) == 2:
                    label = "EAE"
                else:
                    label = "NAE ({},{})".format(delta_m, delta_n)
            else:
                if delta_m == 0:
                    label = "MAE ({},{})".format(delta_m, delta_n)
                else:
                    label = "HAE ({},{})".format(delta_m, delta_n)

            text_fontsize = 12
            box_alpha = 0.9
            font_weight = "bold"

            props = dict(boxstyle="round", facecolor="white", alpha=box_alpha)
            ax.text(gap_labels["s"][i], gap_labels["freq"][i], label, bbox=props, fontsize=text_fontsize, fontweight=font_weight)

            is_QI = True

            if insert_inset and not is_QI:
                if (
                    gap_labels["s"][i] >= x_bound[0]
                    and gap_labels["s"][i] <= x_bound[1]
                    and gap_labels["freq"][i] >= y_bound[0]
                    and gap_labels["freq"][i] <= y_bound[1]
                ):
                    ax_inset.text(gap_labels["s"][i], gap_labels["freq"][i], label, bbox=props, fontsize=text_fontsize, fontweight=font_weight)

    if insert_inset:
        props = dict(boxstyle="round", facecolor="yellow", alpha=box_alpha)
        ax_inset.text(0.04, 0.9, "m=29,n=32", bbox=props, fontsize=text_fontsize, fontweight=font_weight)
        ax_inset.text(0.17, 0.8, "m=30,n=32", bbox=props, fontsize=text_fontsize, fontweight=font_weight)

    return [figure, ax]


def plot_continuum_matplotlib_overlapping(
    modes: List[Mode],
    modes_2: List[Mode],
    normalized: bool = False,
    ylims: List[float] = None,
    xlims: List[float] = None,
    vmec_file: str = "",
    insert_inset=False,
    inset_xlims=[0, 1],
    label_name: str = None,
    gap_labels: dict = None,
    **kwargs,
):
    vmec = Vmec(vmec_file)
    vmec_2 = Vmec(vmec_file_2)
    figure, ax = plt.subplots(1, figsize=(7, 4))

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    if xlims is not None:
        x_max = xlims[1]
        x_min = xlims[0]
    else:
        x_max = 1.0
        x_min = modes[0].s[0]
        xlims = [x_min, x_max]

    if insert_inset:
        x_bound = [0.07, 0.53]
        y_bound = [0.4, 1.0]
        box_location = [1.1, 0.07, 0.7, 0.5]

        ax_inset = ax.inset_axes(box_location, xlim=(x_bound[0], x_bound[1]), ylim=(y_bound[0], y_bound[1]))
        ax_inset.set_yticklabels([])
        ax_inset.tick_params(axis="y", length=0)

    show_label_1 = True
    show_label_2 = True

    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            for md in modes_2:
                alpha_val2 = 0.25
                freq_normalized = get_normalized_frequencies(vmec_2, md)

                if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
                    if show_label_2:
                        ax.scatter(-5, -100, s=100, color="#3D74B6", alpha=alpha_val2, label=label_name, marker="x")
                        show_label_2 = False

                    ax.scatter(md.s, freq_normalized, s=1, color="#3D74B6", alpha=alpha_val2, marker="x")

                    if insert_inset:
                        ax_inset.scatter(md.s, freq_normalized, s=1)

            for md in modes:
                freq_normalized = get_normalized_frequencies(vmec, md)

                if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
                    alpha_val1 = 0.25

                    if show_label_1:
                        ax.scatter(-5, -100, s=100, color="#DC3C22", alpha=alpha_val1, label="block", marker="x")
                        show_label_1 = False

                    ax.scatter(md.s, freq_normalized, s=1, color="#DC3C22", alpha=alpha_val1, marker="x")

                    if insert_inset:
                        ax_inset.scatter(md.s, freq_normalized, s=1)

        ax.set_ylabel(r"$\overline{\omega}$")
    else:
        for md in modes:
            freqs = md.freq

            if ylims is None or (np.min(freqs) <= y_max and np.max(freqs) >= y_min):
                ax.scatter(md.s, freqs, s=1)

                if insert_inset:
                    ax_inset.scatter(md.s, freqs, s=1)

        ax.set_ylabel(r"Frequency $\omega [kHz]$")

    if ylims is not None:
        ax.set_ylim(y_min, y_max)

    if xlims is not None:
        ax.set_xlim(x_min, x_max)

    ax.set_xlabel(r"$s$")

    if insert_inset:
        linewidth = 2
        alpha = 1

        rect_patch, connector_lines = ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=linewidth, alpha=alpha)

        for line in connector_lines:
            line.set_alpha(alpha)
            line.set_linewidth(linewidth)

        connector_lines[0].set_visible(True)
        connector_lines[2].set_visible(False)
        connector_lines[1].set_visible(True)
        connector_lines[3].set_visible(False)

    if gap_labels is not None:
        for i in range(0, len(gap_labels["s"])):
            label = ""
            delta_m = gap_labels["m"][i]
            delta_n = gap_labels["n"][i]

            if delta_n == 0:
                if delta_m == 0:
                    label = "GAE"
                elif abs(delta_m) == 1:
                    label = "TAE"
                elif abs(delta_m) == 2:
                    label = "EAE"
                else:
                    label = "NAE ({},{})".format(delta_m, delta_n)
            else:
                if delta_m == 0:
                    label = "MAE ({},{})".format(delta_m, delta_n)
                else:
                    label = "HAE ({},{})".format(delta_m, delta_n)

            text_fontsize = 12
            box_alpha = 0.9
            font_weight = "bold"

            props = dict(boxstyle="round", facecolor="white", alpha=box_alpha)
            ax.text(gap_labels["s"][i], gap_labels["freq"][i], label, bbox=props, fontsize=text_fontsize, fontweight=font_weight)

            is_QI = True

            if insert_inset and not is_QI:
                if (
                    gap_labels["s"][i] >= x_bound[0]
                    and gap_labels["s"][i] <= x_bound[1]
                    and gap_labels["freq"][i] >= y_bound[0]
                    and gap_labels["freq"][i] <= y_bound[1]
                ):
                    ax_inset.text(gap_labels["s"][i], gap_labels["freq"][i], label, bbox=props, fontsize=text_fontsize, fontweight=font_weight)

    if insert_inset:
        from matplotlib.patches import FancyArrowPatch

        props = dict(boxstyle="round", facecolor="white", alpha=box_alpha)
        ax_inset.text(0.10, 0.78, "HAE (3,4)", bbox=props, fontsize=text_fontsize, fontweight=font_weight)
        ax_inset.text(0.10, 0.61, "EAE", bbox=props, fontsize=text_fontsize, fontweight=font_weight)

        arrow = FancyArrowPatch(
            posA=(arrow_pos[0], arrow_pos[1]),
            posB=(arrow_pos[0] + arrow_len[0], arrow_pos[1] + arrow_len[1]),
            arrowstyle="simple,head_width=4,head_length=4",
            color="#DC3C22",
            linewidth=1.5,
            transform=ax_inset.transData,
            joinstyle="miter",
            capstyle="butt",
            mutation_scale=1,
        )
        ax_inset.add_patch(arrow)

        arrow = FancyArrowPatch(
            posA=(arrow_pos_2[0], arrow_pos_2[1]),
            posB=(arrow_pos_2[0] + arrow_len_2[0], arrow_pos_2[1] + arrow_len_2[1]),
            arrowstyle="simple,head_width=4,head_length=4",
            color="#3D74B6",
            linewidth=1.5,
            transform=ax_inset.transData,
            joinstyle="miter",
            capstyle="butt",
            mutation_scale=1,
        )
        ax_inset.add_patch(arrow)

    return figure


def plot_continuum_matplotlib_by_prop_direction(
    modes: List[Mode],
    normalized: bool = False,
    ylims: List[float] = None,
    xlims: List[float] = None,
    vmec_file: str = "",
    gap_labels: dict = None,
):
    vmec = Vmec(vmec_file)

    iotas = vmec.wout.iotas
    delta_s = 1.0 / (len(iotas) - 1)

    figure, ax = plt.subplots(1, figsize=(7, 4))

    if ylims is not None:
        y_max = ylims[1]
        y_min = ylims[0]

    if xlims is not None:
        x_max = xlims[1]
        x_min = xlims[0]

    if normalized:
        if str == "":
            raise AttributeError("VMEC file required and not provided")
        else:
            for md in modes:
                freq_normalized = get_normalized_frequencies(vmec, md)

                if ylims is None or (np.min(freq_normalized) <= y_max and np.max(freq_normalized) >= y_min):
                    marker_colors = [""] * len(freq_normalized)

                    for i in range(0, len(freq_normalized)):
                        s_ind = int(np.round(md.s[i] / delta_s))

                        test_val = iotas[s_ind] * md.m - md.n
                        if test_val >= 0:
                            marker_colors[i] = "blue"
                        else:
                            marker_colors[i] = "red"

                    ax.scatter(md.s, freq_normalized, s=1, c=marker_colors)

        ax.set_ylabel(r"$\overline{\omega}$")
    else:
        for md in modes:
            freqs = md.freq
            marker_colors = [""] * len(freqs)

            for i in range(0, len(freqs)):
                s_ind = int(np.round(md.s[i] / delta_s))
                test_val = iotas[s_ind] * md.m - md.n
                if test_val >= 0:
                    marker_colors[i] = "blue"
                else:
                    marker_colors[i] = "red"

            if ylims is None or (np.min(freqs) <= y_max and np.max(freqs) >= y_min):
                ax.scatter(md.s, freqs, s=1, c=marker_colors)

        ax.set_ylabel(r"Frequency $\omega [kHz]$")

    if ylims is not None:
        ax.set_ylim(y_min, y_max)

    if xlims is not None:
        ax.set_xlim(x_min, x_max)

    ax.set_xlabel(r"$s$")

    if gap_labels is not None:
        props = dict(boxstyle="round", facecolor="white")

        for i in range(0, len(gap_labels["s"])):
            label = ""
            delta_m = gap_labels["m"][i]
            delta_n = gap_labels["n"][i]

            if delta_n == 0:
                if delta_m == 0:
                    label = "GAE"
                elif abs(delta_m) == 1:
                    label = "TAE"
                elif abs(delta_m) == 2:
                    label = "EAE"
                else:
                    label = "NAE ({},{})".format(delta_m, delta_n)
            else:
                if delta_m == 0:
                    label = "MAE ({},{})".format(delta_m, delta_n)
                else:
                    label = "HAE ({},{})".format(delta_m, delta_n)

            ax.text(gap_labels["s"][i], gap_labels["freq"][i], label, bbox=props, fontsize=6)

    return figure


def plot_condition_numbers(s, condition_numbers):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=condition_numbers, mode="lines+markers", name="Condition Number"))
    fig.update_layout(
        title=r"$\text{Condition Number in STELLGAP output}$",
        xaxis_title=r"$\text{Normalized Flux }s$",
        yaxis_title=r"$\text{Condition Number }|\lambda_{\rm max}/\lambda_{\rm min}|$",
        legend_title="Legend",
        xaxis=dict(range=[np.min(s), np.max(s)]),
        yaxis=dict(type="log", tickformat=".0e", exponentformat="e", showexponent="all"),
    )
    return fig


@dataclass
class DataPost:
    iopt: int
    eigenvector_mode_num: int
    radial_points_num: int
    isym_opt: int

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as file:
            line = file.readline().strip()
        values = list(map(int, line.split()))
        if len(values) != 4:
            raise ValueError("File does not contain exactly four integers.")
        return cls(*values)


@dataclass
class ProfilesDat:
    sim_dir: str
    file_path: str = field(init=False)
    field_descriptions: dict = field(default_factory=dict, init=False)
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "profiles.dat")
        self.initialize_descriptions()
        self.array = self.load_data()

    def initialize_descriptions(self):
        self.field_descriptions = {
            "rho": "Normalized radial position",
            "den_ion": "Ion density",
            "iota": "Iota",
            "iotap": "Radial iota derivative",
            "jpol": "Poloidal current density",
            "jpolp": "Radial derivative poloidal current density",
            "jtor": "Toroidal current density",
            "jtorp": "Radial derivative toroidal current density",
            "presp": "Radial pressure gradient",
            "phip": "Toroidal flux",
            "phipp": "Radial derivative toroidal flux",
            "jprl0": "J_parallel/B coefficient 0",
            "jprl1": "J_parallel/B coefficient 1",
            "jprl2": "J_parallel/B coefficient 2",
        }

    def load_data(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")
        with open(self.file_path, "r") as file:
            header = file.readline().strip().split()
            dtype = [(name, "f8") for name in header]
            data = np.loadtxt(file, dtype=np.dtype(dtype))

        return data


@dataclass
class IonProfile:
    array: np.ndarray

    @classmethod
    def from_file(cls, file_path: str):
        ion_profile_data = np.loadtxt(file_path)
        dtype = [
            ("normalized_flux", "f8"),
            ("ion_number_density", "f8"),
            ("rotation_transform_iota", "f8"),
            ("alfven_speed", "f8"),
        ]
        array = np.core.records.fromarrays(
            [
                ion_profile_data[:, 0],
                ion_profile_data[:, 1],
                ion_profile_data[:, 2],
                ion_profile_data[:, 3],
            ],
            dtype=dtype,
        )
        return cls(array=array)


@dataclass
class EgnvaluesDat:
    sim_dir: str
    file_path: str = field(init=False)
    field_descriptions: dict = field(default_factory=dict, init=False)
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "egn_values.dat")
        self.initialize_descriptions()
        self.array = self.load_data()

    def initialize_descriptions(self):
        self.field_descriptions = {
            "eigenvalue": "The eigenvalue of the AE3D mode (either squared if positive or as is if negative)",
            "electrostatic (inertial) energy": "The electrostatic or inertial energy component of the mode",
            "electromagnetic (field bending)": "The electromagnetic or field bending energy component of the mode",
        }

    def load_data(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")
        data = np.loadtxt(
            self.file_path,
            dtype=[
                ("eigenvalue", "f8"),
                ("electrostatic (inertial) energy", "f8"),
                ("electromagnetic (field bending)", "f8"),
            ],
        )

        positive_indices = data["eigenvalue"] > 0
        data["eigenvalue"][positive_indices] = np.square(data["eigenvalue"][positive_indices])

        return data

    def condition_number(self):
        eigenvalues = self.array["eigenvalue"]
        abs_eigenvalues = np.abs(eigenvalues)
        max_eigenvalue = np.max(abs_eigenvalues)
        min_eigenvalue = np.min(abs_eigenvalues)
        if min_eigenvalue == 0:
            raise ValueError("The smallest absolute eigenvalue is zero, condition number is undefined.")
        return max_eigenvalue / min_eigenvalue


@dataclass
class FieldBendingMatrix:
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Field bending (sparce) matrix A from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "a_matrix.dat")
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, "r") as file:
            data = np.loadtxt(file, dtype=[("i", int), ("j", int), ("value", float)])

        rows = data["i"] - 1
        cols = data["j"] - 1
        values = data["value"]

        size = max(np.max(rows), np.max(cols)) + 1
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))


@dataclass
class InertiaMatrix:
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Inertia matrix B from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "b_matrix.dat")
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, "r") as file:
            data = np.loadtxt(file, dtype=[("i", int), ("j", int), ("value", float)])

        rows = data["i"] - 1
        cols = data["j"] - 1
        values = data["value"]

        size = max(np.max(rows), np.max(cols)) + 1
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))


@dataclass
class JDQZData:
    sim_dir: str
    file_path: str = field(init=False)
    ns: int = field(init=False)
    mn_col: int = field(init=False)
    mode_numbers: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "jdqz_data.dat")
        self.load_data()

    def load_data(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, "r") as file:
            first_line = file.readline().split()
            self.ns, self.mn_col = int(first_line[0]), int(first_line[1])
            mode_data = np.loadtxt(file, max_rows=self.mn_col, dtype=[("m", int), ("n", int)])
            self.mode_numbers = mode_data
            self.s = np.loadtxt(file, dtype="f8")


@dataclass
class Omega2Dat:
    sim_dir: str
    file_path: str = field(init=False)
    array: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "omega2.dat")
        self.array = self.load_data()

    def load_data(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        dtype = [
            ("index", int),
            ("alphar", "f8"),
            ("alphai", "f8"),
            ("betar", "f8"),
            ("dm", "f8"),
        ]
        data = np.loadtxt(self.file_path, dtype=dtype)

        if np.any(np.abs(data["alphai"]) > 1e-10):
            raise ValueError("Non-zero imaginary components found in an ideal MHD scenario.")

        return data


@dataclass
class EigModeASCI:
    sim_dir: str
    file_path: str = field(init=False)
    num_eigenmodes: int = field(init=False)
    num_fourier_modes: int = field(init=False)
    num_radial_points: int = field(init=False)
    modes: np.ndarray = field(init=False)
    egn_values: np.ndarray = field(init=False)
    s_coords: np.ndarray = field(init=False)
    egn_vectors: np.ndarray = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, "egn_mode_asci.dat")
        self.load_data()

    def load_data(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        data = np.loadtxt(self.file_path)
        it = iter(data)
        self.num_eigenmodes = int(next(it))
        self.num_fourier_modes = int(next(it))
        self.num_radial_points = int(next(it))

        self.modes = np.array([(next(it), next(it)) for _ in range(self.num_fourier_modes)], dtype=[("m", "int32"), ("n", "int32")])
        self.egn_values = np.array([next(it) for _ in range(self.num_eigenmodes)])
        self.s_coords = np.array([next(it) for _ in range(self.num_radial_points)])
        self.egn_vectors = np.array(
            [next(it) for _ in range(self.num_eigenmodes * self.num_radial_points * self.num_fourier_modes)]
        ).reshape(self.num_eigenmodes, self.num_radial_points, self.num_fourier_modes)

    def get_nearest_eigenvector(self, target_eigenvalue):
        data = [(self.egn_values[I], self.egn_vectors[I]) for I in range(len(self.egn_values))]
        data.sort(key=lambda a: np.abs(a[0] - target_eigenvalue))
        nearest_egn_value, nearest_vector = data[0]
        sort_by_energy = np.argsort(np.sum(-nearest_vector**2, axis=0))
        egn_vector_sorted = nearest_vector[:, sort_by_energy]
        modes_sorted = self.modes[sort_by_energy]
        normalized_egn_vector = egn_vector_sorted / egn_vector_sorted[np.argmax(np.abs(egn_vector_sorted[:, 0])), 0]
        return nearest_egn_value, normalized_egn_vector, modes_sorted

    def condition_number(self):
        eigenvalues = self.egn_values
        abs_eigenvalues = np.abs(eigenvalues)
        max_eigenvalue = np.max(abs_eigenvalues)
        min_eigenvalue = np.min(abs_eigenvalues)
        if min_eigenvalue == 0:
            raise ValueError("The smallest absolute eigenvalue is zero, condition number is undefined.")
        return max_eigenvalue / min_eigenvalue


@dataclass
class Harmonic:
    m: int
    n: int
    amplitudes: np.ndarray


@dataclass
class AE3DEigenvector:
    eigenvalue: float
    s_coords: np.ndarray
    harmonics: List[Harmonic]

    @staticmethod
    def from_eig_mode_asci(eig_mode_asci: EigModeASCI, target_eigenvalue: float):
        egn_value, egn_vector_sorted, modes_sorted = eig_mode_asci.get_nearest_eigenvector(target_eigenvalue)
        harmonics = [
            Harmonic(m=modes_sorted["m"][i], n=modes_sorted["n"][i], amplitudes=egn_vector_sorted[:, i])
            for i in range(len(modes_sorted))
        ]
        return AE3DEigenvector(eigenvalue=egn_value, s_coords=eig_mode_asci.s_coords, harmonics=harmonics)


def plot_ae3d_eigenmode(mode: AE3DEigenvector, harmonics: int = 5):
    fig = go.Figure()

    num_harmonics_to_plot = min(harmonics, len(mode.harmonics))

    for i in range(num_harmonics_to_plot):
        harmonic = mode.harmonics[i]
        fig.add_trace(go.Scatter(x=mode.s_coords, y=harmonic.amplitudes, mode="lines", name=f"(m={harmonic.m}, n={harmonic.n})"))

    fig.update_yaxes(title_text=r"$\text{Electrostatic Potential }\varphi$")
    fig.update_xaxes(title_text=r"$\text{Normalized Flux }s$")
    fig.update_layout(title=f"Eigenvalue: {mode.eigenvalue}")

    return fig


@dataclass
class FAR3DEigenproblem:
    sim_dir: str
    a_matrix_path: str = field(init=False)
    b_matrix_path: str = field(init=False)
    jdqz_path: str = field(init=False)
    matrix_size: int = field(init=False)
    matrix_A: sp.coo_matrix = field(init=False)
    matrix_B: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.a_matrix_path = os.path.join(self.sim_dir, "a_matrix.dat")
        self.b_matrix_path = os.path.join(self.sim_dir, "b_matrix.dat")
        self.jdqz_path = os.path.join(self.sim_dir, "jdqz.dat")
        self.matrix_size = self.calculate_matrix_size()
        self.matrix_A = self.load_matrix(self.a_matrix_path)
        self.matrix_B = self.load_matrix(self.b_matrix_path)

    def calculate_matrix_size(self):
        if not os.path.isfile(self.jdqz_path):
            raise FileNotFoundError(f"Data file {self.jdqz_path} not found.")

        with open(self.jdqz_path, "r") as file:
            first_line = file.readline().strip()
            mjm1, lmaxn, noeqn = map(int, first_line.split())
            matrix_size = mjm1 * lmaxn * noeqn
        return matrix_size

    def load_matrix(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found.")

        with open(file_path, "r") as file:
            data = np.loadtxt(file, dtype=[("i", int), ("j", int), ("value", float)])

        rows = data["i"] - 1
        cols = data["j"] - 1
        values = data["value"]

        return sp.coo_matrix((values, (rows, cols)), shape=(self.matrix_size, self.matrix_size))
