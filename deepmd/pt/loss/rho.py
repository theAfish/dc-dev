# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class RhoLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_rho: float = 1.0,
        limit_pref_rho: float = 1.0,
        use_l1_all: bool = False,
        inference=False,
        **kwargs,
    ):
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_e : float
            The prefactor of energy loss at the start of the training.
        limit_pref_e : float
            The prefactor of energy loss at the end of the training.
        start_pref_f : float
            The prefactor of force loss at the start of the training.
        limit_pref_f : float
            The prefactor of force loss at the end of the training.
        start_pref_v : float
            The prefactor of virial loss at the start of the training.
        limit_pref_v : float
            The prefactor of virial loss at the end of the training.
        start_pref_ae : float
            The prefactor of atomic energy loss at the start of the training.
        limit_pref_ae : float
            The prefactor of atomic energy loss at the end of the training.
        start_pref_pf : float
            The prefactor of atomic prefactor force loss at the start of the training.
        limit_pref_pf : float
            The prefactor of atomic prefactor force loss at the end of the training.
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate

        self.has_rho = (start_pref_rho != 0.0 and limit_pref_rho != 0.0) or inference

        self.start_pref_rho = start_pref_rho
        self.limit_pref_rho = limit_pref_rho
        self.use_l1_all = use_l1_all
        self.inference = inference

    def forward(self, model_pred, label, natoms, learning_rate, atype=None, mae=False):
        """Return loss on loss and force.

        Args:
        - natoms: Tell atom count.
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.

        Returns
        -------
        - loss: Loss to minimize.
        """
        coef = learning_rate / self.starter_learning_rate
        pref_rho = self.limit_pref_rho + (self.start_pref_rho - self.limit_pref_rho) * coef
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms

        indices_pseudo = (atype.flatten() == 0).nonzero().flatten()

        if self.has_rho and "rho" in model_pred and "rho" in label:
            if not self.use_l1_all:
                l2_rho_loss = torch.mean(
                    torch.square(model_pred["rho"][:, indices_pseudo] - label["rho"][:, indices_pseudo])
                )
                if not self.inference:
                    more_loss["l2_rho_loss"] = l2_rho_loss.detach()
                loss += atom_norm * (pref_rho * l2_rho_loss)
                rmse_rho = l2_rho_loss.sqrt() * atom_norm
                more_loss["rmse_rho"] = rmse_rho.detach()
            else:
                l1_ener_loss = F.l1_loss(
                    model_pred["rho"][:, indices_pseudo].reshape(-1),
                    label["rho"][:, indices_pseudo].reshape(-1),
                    reduction="sum",
                )
                loss += pref_rho * l1_ener_loss
                more_loss["mae_rho"] = F.l1_loss(
                    model_pred["rho"].reshape(-1),
                    label["rho"].reshape(-1),
                    reduction="mean",
                ).detach()

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_rho:
            label_requirement.append(
                DataRequirementItem(
                    "rho",
                    ndof=1,
                    atomic=True,
                    must=True,
                    high_prec=False,
                )
            )
        return label_requirement
