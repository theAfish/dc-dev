# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
from typing import (
    List,
    Optional,
    Tuple, Union, Callable,
)

import numpy as np
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
    fitting_check_output,
)
from deepmd.dpmodel.utils import AtomExcludeMask
from deepmd.pt.model.network.network import (
    ResidualDeep,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
    GeneralFitting,
)
from deepmd.pt.model.task.invar_fitting import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.pt.utils.utils import to_numpy_array
from deepmd.utils.out_stat import compute_stats_from_redu
from deepmd.utils.path import DPPath
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("rho")
class RhoFittingNet(InvarFitting):
    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: List[int] = [128, 128, 128],
        bias_atom_e: Optional[torch.Tensor] = None,
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        **kwargs,
    ):
        super().__init__(
            "rho",
            ntypes,
            dim_descrpt,
            1,
            neuron=neuron,
            bias_atom_e=bias_atom_e,
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            **kwargs,
        )

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("var_name")
        data.pop("dim_out")
        return super().deserialize(data)

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        return {
            **super().serialize(),
            "type": "rho",
        }

    def compute_output_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        bias_atom_e = self._compute_output_stats(
            merged, self.ntypes, stat_file_path, self.rcond, self.atom_ener
        )
        self.bias_atom_e.copy_(
            torch.tensor(bias_atom_e, device=env.DEVICE).view(
                [self.ntypes, self.dim_out]
            )
        )
    def _compute_output_stats(
            self,
            merged: Union[Callable[[], List[dict]], List[dict]],
            ntypes: int,
            stat_file_path: Optional[DPPath] = None,
            rcond: Optional[float] = None,
            atom_ener: Optional[List[float]] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        ntypes : int
            The number of atom types.
        stat_file_path : DPPath, optional
            The path to the stat file.
        rcond : float, optional
            The condition number for the regression of atomic energy.
        atom_ener : List[float], optional
            Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.

        """
        if stat_file_path is not None:
            stat_file_path = stat_file_path / "bias_atom_e"
        if stat_file_path is not None and stat_file_path.is_file():
            bias_atom_e = stat_file_path.load_numpy()
        else:
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
            energy = [item["rho"].sum(axis=1) for item in sampled]
            data_mixed_type = "real_natoms_vec" in sampled[0]
            natoms_key = "natoms" if not data_mixed_type else "real_natoms_vec"
            for system in sampled:
                if "atom_exclude_types" in system:
                    type_mask = AtomExcludeMask(
                        ntypes, system["atom_exclude_types"]
                    ).get_type_mask()
                    system[natoms_key][:, 2:] *= type_mask.unsqueeze(0)
            input_natoms = [item[natoms_key] for item in sampled]
            # shape: (nframes, ndim)
            merged_energy = to_numpy_array(torch.cat(energy))
            # shape: (nframes, ntypes)
            merged_natoms = to_numpy_array(torch.cat(input_natoms)[:, 2:])
            if atom_ener is not None and len(atom_ener) > 0:
                assigned_atom_ener = np.array(
                    [ee if ee is not None else np.nan for ee in atom_ener]
                )
            else:
                assigned_atom_ener = None
            bias_atom_e, _ = compute_stats_from_redu(
                merged_energy,
                merged_natoms,
                assigned_bias=assigned_atom_ener,
                rcond=rcond,
            )
            if stat_file_path is not None:
                stat_file_path.save_numpy(bias_atom_e)
        assert all(x is not None for x in [bias_atom_e])
        return torch.tensor(bias_atom_e, device=env.DEVICE)

    # make jit happy with torch 2.0.0
    exclude_types: List[int]