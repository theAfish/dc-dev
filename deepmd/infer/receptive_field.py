# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    overload,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)

from .deep_eval import (
    DeepEval,
)


class ReceptiveField(DeepEval):

    @property
    def output_def(self) -> ModelOutputDef:
        """Get the output definition of this model."""
        return ModelOutputDef(
            FittingOutputDef(
                [
                    OutputVariableDef(
                        "energy",
                        shape=[1],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                        receptive_field=True,
                        atomic=True
                    ),
                    OutputVariableDef(
                        "rec_field",
                        shape=[-1],
                        reducible=False,
                        r_differentiable=False,
                        c_differentiable=False,
                        receptive_field=False,
                        atomic=True,
                        category=32,
                    ),
                ]
            )
        )

    @overload
    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: Literal[True],
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
        mixed_type: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @overload
    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: Literal[False],
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
        mixed_type: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @overload
    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: bool,
        fparam: Optional[np.ndarray],
        aparam: Optional[np.ndarray],
        mixed_type: bool,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        pass

    def eval(
        self,
        coords: np.ndarray,
        cells: Optional[np.ndarray],
        atom_types: Union[list[int], np.ndarray],
        atomic: bool = False,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
        mixed_type: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        (
            coords,
            cells,
            atom_types,
            fparam,
            aparam,
            nframes,
            natoms,
        ) = self._standard_input(coords, cells, atom_types, fparam, aparam, mixed_type)

        results = self.deep_eval.eval(
            coords,
            cells,
            atom_types,
            atomic,
            fparam=fparam,
            aparam=aparam,
            **kwargs,
        )
        rec_field = results["rec_field"]
        result = (
            rec_field
        )
        return result


__all__ = ["ReceptiveField"]
