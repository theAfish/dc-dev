# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
from typing import (
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    DescriptorBlock,
    prod_env_mat_rho,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

try:
    from typing import (
        Final,
    )
except ImportError:
    from torch.jit import Final

from deepmd.dpmodel.utils import EnvMat as DPEnvMat
from deepmd.pt.model.network.mlp import (
    EmbeddingNet,
    NetworkCollection,
)
from deepmd.pt.model.network.network import (
    TypeFilter,
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,
)

from .se_a import (
    DescrptSeA,
)
from .base_descriptor import (
    BaseDescriptor,
)

@BaseDescriptor.register("se_e2_a_rho")
@BaseDescriptor.register("se_a_rho")
class DescrptSeARho(DescrptSeA, torch.nn.Module):
    def __init__(
        self,
        rcut,
        rcut_smth,
        sel,
        neuron=[25, 50, 100],
        axis_neuron=16,
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "float64",
        resnet_dt: bool = False,
        exclude_types: List[Tuple[int, int]] = [],
        env_protection: float = 0.0,
        old_impl: bool = False,
        type_one_side: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.sea = DescrptSeA(
            rcut,
            rcut_smth,
            sel,
            neuron=neuron,
            axis_neuron=axis_neuron,
            set_davg_zero=set_davg_zero,
            activation_function=activation_function,
            precision=precision,
            resnet_dt=resnet_dt,
            exclude_types=exclude_types,
            env_protection=env_protection,
            old_impl=old_impl,
            type_one_side=type_one_side,
            **kwargs,
        )



    def compute_input_stats(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        path: Optional[DPPath] = None,
    ):
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        return self.sea.compute_input_stats(merged, path)


    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        nlist
            The neighbor list. shape: nf x nloc x nnei
        mapping
            The index mapping, not required by this descriptor.

        Returns
        -------
        descriptor
            The descriptor. shape: nf x nloc x (ng x axis_neuron)
        gr
            The rotationally equivariant and permutationally invariant single particle
            representation. shape: nf x nloc x ng x 3
        g2
            The rotationally invariant pair-partical representation.
            this descriptor returns None
        h2
            The rotationally equivariant pair-partical representation.
            this descriptor returns None
        sw
            The smooth switch function.

        """
        return self.sea.forward(nlist, coord_ext, atype_ext, None, mapping)



    def serialize(self) -> dict:
        obj = self.sea
        return {
            "@class": "Descriptor",
            "type": "se_e2_a_rho",
            "@version": 1,
            "rcut": obj.rcut,
            "rcut_smth": obj.rcut_smth,
            "sel": obj.sel,
            "neuron": obj.neuron,
            "axis_neuron": obj.axis_neuron,
            "resnet_dt": obj.resnet_dt,
            "set_davg_zero": obj.set_davg_zero,
            "activation_function": obj.activation_function,
            # make deterministic
            "precision": RESERVED_PRECISON_DICT[obj.prec],
            "embeddings": obj.filter_layers.serialize(),
            "env_mat": DPEnvMat(obj.rcut, obj.rcut_smth).serialize(),
            "exclude_types": obj.exclude_types,
            "env_protection": obj.env_protection,
            "@variables": {
                "davg": obj["davg"].detach().cpu().numpy(),
                "dstd": obj["dstd"].detach().cpu().numpy(),
            },
            ## to be updated when the options are supported.
            "trainable": True,
            "type_one_side": obj.type_one_side,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptSeARho":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        embeddings = data.pop("embeddings")
        env_mat = data.pop("env_mat")
        obj = cls(**data)

        def t_cvt(xx):
            return torch.tensor(xx, dtype=obj.sea.prec, device=env.DEVICE)

        obj.sea["davg"] = t_cvt(variables["davg"])
        obj.sea["dstd"] = t_cvt(variables["dstd"])
        obj.sea.filter_layers = NetworkCollection.deserialize(embeddings)
        return obj

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
    ):
        """Calculate decoded embedding for each atom.

        Args:
        - coord: Tell atom coordinates with shape [nframes, natoms[1]*3].
        - atype: Tell atom types with shape [nframes, natoms[1]].
        - natoms: Tell atom count and element count. Its shape is [2+self.ntypes].
        - box: Tell simulation box with shape [nframes, 9].

        Returns
        -------
        - `torch.Tensor`: descriptor matrix with shape [nframes, natoms[0]*self.filter_neuron[-1]*self.axis_neuron].
        """
        del extended_atype_embd, mapping
        nloc = nlist.shape[1]
        atype = extended_atype[:, :nloc]
        dmatrix, diff, sw = prod_env_mat_rho(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.rcut,
            self.rcut_smth,
            protection=self.env_protection,
        )

        if self.old_impl:
            assert self.filter_layers_old is not None
            dmatrix = dmatrix.view(
                -1, self.ndescrpt
            )  # shape is [nframes*nall, self.ndescrpt]
            xyz_scatter = torch.empty(
                1,
                device=env.DEVICE,
            )
            ret = self.filter_layers_old[0](dmatrix)
            xyz_scatter = ret
            for ii, transform in enumerate(self.filter_layers_old[1:]):
                # shape is [nframes*nall, 4, self.filter_neuron[-1]]
                ret = transform.forward(dmatrix)
                xyz_scatter = xyz_scatter + ret
        else:
            assert self.filter_layers is not None
            dmatrix = dmatrix.view(-1, self.nnei, 4)
            dmatrix = dmatrix.to(dtype=self.prec)
            nfnl = dmatrix.shape[0]
            # pre-allocate a shape to pass jit
            xyz_scatter = torch.zeros(
                [nfnl, 4, self.filter_neuron[-1]],
                dtype=self.prec,
                device=extended_coord.device,
            )
            # nfnl x nnei
            exclude_mask = self.emask(nlist, extended_atype).view(nfnl, -1)
            for embedding_idx, ll in enumerate(self.filter_layers.networks):
                if self.type_one_side:
                    ii = embedding_idx
                    # torch.jit is not happy with slice(None)
                    # ti_mask = torch.ones(nfnl, dtype=torch.bool, device=dmatrix.device)
                    # applying a mask seems to cause performance degradation
                    ti_mask = None
                else:
                    # ti: center atom type, ii: neighbor type...
                    ii = embedding_idx // self.ntypes
                    ti = embedding_idx % self.ntypes
                    ti_mask = atype.ravel().eq(ti)
                # nfnl x nt
                if ti_mask is not None:
                    mm = exclude_mask[ti_mask, self.sec[ii] : self.sec[ii + 1]]
                else:
                    mm = exclude_mask[:, self.sec[ii] : self.sec[ii + 1]]
                # nfnl x nt x 4
                if ti_mask is not None:
                    rr = dmatrix[ti_mask, self.sec[ii] : self.sec[ii + 1], :]
                else:
                    rr = dmatrix[:, self.sec[ii] : self.sec[ii + 1], :]
                rr = rr * mm[:, :, None]
                ss = rr[:, :, :1]
                # nfnl x nt x ng
                gg = ll.forward(ss)
                # nfnl x 4 x ng
                gr = torch.matmul(rr.permute(0, 2, 1), gg)
                if ti_mask is not None:
                    xyz_scatter[ti_mask] += gr
                else:
                    xyz_scatter += gr

        xyz_scatter /= self.nnei
        xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)
        rot_mat = xyz_scatter_1[:, :, 1:4]
        xyz_scatter_2 = xyz_scatter[:, :, 0 : self.axis_neuron]
        result = torch.matmul(
            xyz_scatter_1, xyz_scatter_2
        )  # shape is [nframes*nall, self.filter_neuron[-1], self.axis_neuron]
        result = result.view(-1, nloc, self.filter_neuron[-1] * self.axis_neuron)
        rot_mat = rot_mat.view([-1, nloc] + list(rot_mat.shape[1:]))  # noqa:RUF005
        return (
            result.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            rot_mat.to(dtype=env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            sw,
        )