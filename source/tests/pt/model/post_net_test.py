# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from argparse import (
    Namespace,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch
import torch.nn.functional as F

from deepmd.infer.deep_pot import DeepPot as DeepPotUni
from deepmd.pt.entrypoints.main import (
    get_trainer,
)


from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pt.model.network.mlp import (
    FittingNet,
)


CUR_DIR = os.path.dirname(__file__)


def get_data(trainer):
    input_dict, label_dict, log_dict = trainer.get_data(is_train=False, task_key="Default")

    (
        extended_coord,
        extended_atype,
        mapping,
        nlist,
    ) = extend_input_and_build_neighbor_list(
        input_dict["coord"],
        input_dict["atype"],
        des.get_rcut(),
        des.get_sel(),
        mixed_types=des.mixed_types(),
        box=input_dict["box"],
    )

    descriptor, rot_mat, g2, h2, sw = des(
        extended_coord,
        extended_atype,
        nlist,
        mapping=mapping,
    )

    model_pred, loss, more_loss = trainer.wrapper(
                    **input_dict, cur_lr=0.0, label=label_dict, task_key="Default")
    
    data = torch.cat([descriptor, model_pred["atom_energy"]], dim=-1)
    label = label_dict["property"]
    return data, label


class PostLoss(torch.nn.Module):
    def forward(self, pred, label):
        pred = torch.mean(pred, dim=1)
        loss = F.l1_loss(pred, label)
        return loss


if __name__ == "__main__":
    input_json = str(Path(__file__).parent / "post/input_finetune_2.json")
    # file_model_param = Path(CUR_DIR) / "post" / "OpenLAM_2.2.0_27heads_beta3.pt"
    file_model_param = Path(CUR_DIR) / "post" / "DPA2_medium_28_10M_rc0.pt"
    with open(input_json, "r") as f:
        input_dict = json.load(f)
    # model_params = input_dict["model"]
    # dparams = model_params["descriptor"]
    # ntypes = len(model_params["type_map"])
    # dparams["ntypes"] = ntypes
    # des = DescrptDPA2(**dparams)
    # state_dict = torch.load(file_model_param)
    # state_dict = state_dict["model"]


    input_dict["model"], finetune_links = get_finetune_rules(
            file_model_param,
            input_dict["model"],
            model_branch="MP_traj_v024_alldata_mixu",
            change_model_params=False,
    )

    trainer = get_trainer(
        input_dict,
        finetune_model=file_model_param,
        finetune_links=finetune_links,
    )
    model = trainer.model

    des = model.get_descriptor()
    fit = model.get_fitting_net()
    property_fit = FittingNet(in_dim=des.get_dim_out()+1, out_dim=1, neuron=[40, 40, 40])

    model = FittingNet(in_dim=des.get_dim_out()+1, out_dim=1, neuron=[40, 40, 40])
    loss = PostLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer.lr_exp.start_lr)

    for i in range(trainer.num_steps):
        data, label = get_data(trainer)
        pred = model(data)
        loss_value = loss(pred, label)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        print(f"Step {i}, Loss: {loss_value.item()}")

    # data = get_data(trainer)
    # print(data.shape)

    # result = property_fit(data)
    # _lr = trainer.lr_exp
    # print(_lr.value(1000))




