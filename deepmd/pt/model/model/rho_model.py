# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from .dp_model import (
    DPModel,
)

from deepmd.pt.utils.nlist_field import (
    extend_input_and_build_neighbor_list,
)

from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)


class RhoModel(DPModel):
    model_type = "rho"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    # cannot use the name forward. torch script does not work
    def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return model prediction.

        Parameters
        ----------
        coord
            The coordinates of the atoms.
            shape: nf x (nloc x 3)
        atype
            The type of atoms. shape: nf x nloc
        box
            The simulation box. shape: nf x 9
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            If calculate the atomic virial.

        Returns
        -------
        ret_dict
            The result dict of type Dict[str,torch.Tensor].
            The keys are defined by the `ModelOutputDef`.

        """
        cc, bb, fp, ap, input_prec = self.input_type_cast(
            coord, box=box, fparam=fparam, aparam=aparam
        )
        del coord, box, fparam, aparam
        (
            extended_coord,
            extended_atype,
            mapping,
            nlist,
        ) = extend_input_and_build_neighbor_list(
            cc,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=self.mixed_types(),
            box=bb,
        )
        model_predict_lower = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            do_atomic_virial=do_atomic_virial,
            fparam=fp,
            aparam=ap,
        )
        model_predict = communicate_extended_output(
            model_predict_lower,
            self.model_output_def(),
            mapping,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = self.output_type_cast(model_predict, input_prec)
        return model_predict

    def forward_common_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
    ):
        """Return model prediction. Lower interface that takes
        extended atomic coordinates and types, nlist, and mapping
        as input, and returns the predictions on the extended region.
        The predictions are not reduced.

        Parameters
        ----------
        extended_coord
            coodinates in extended region. nf x (nall x 3)
        extended_atype
            atomic type in extended region. nf x nall
        nlist
            neighbor list. nf x nloc x nsel.
        mapping
            mapps the extended indices to local indices. nf x nall.
        fparam
            frame parameter. nf x ndf
        aparam
            atomic parameter. nf x nloc x nda
        do_atomic_virial
            whether calculate atomic virial.

        Returns
        -------
        result_dict
            the result dict, defined by the `FittingOutputDef`.

        """
        nframes, nall = extended_atype.shape[:2]
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlist = self.format_nlist(extended_coord, extended_atype, nlist)
        cc_ext, _, fp, ap, input_prec = self.input_type_cast(
            extended_coord, fparam=fparam, aparam=aparam
        )
        del extended_coord, fparam, aparam
        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
        )
        model_predict = fit_output_to_model_output(
            atomic_ret,
            self.atomic_output_def(),
            cc_ext,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = self.output_type_cast(model_predict, input_prec)
        return model_predict

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["rho"] = model_ret["rho"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["rho"] = model_ret["rho"]
        else:
            model_predict = model_ret
        return model_predict
