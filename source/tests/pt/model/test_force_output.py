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
import matplotlib.pyplot as plt


CUR_DIR = os.path.dirname(__file__)



def get_data(trainer):
    input_dict, label_dict, _ = trainer.get_data(is_train=True, task_key="Default")

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

    model_pred, _, _ = trainer.wrapper(
                    **input_dict, cur_lr=0.0, label=label_dict, task_key="Default")
    
    force = model_pred["debug"]
    atom_energy = model_pred["atom_energy"]
    coord = input_dict["coord"]
    print(coord.shape, atom_energy.shape)


    return extended_coord, extended_atype, force, coord, atom_energy


class PostLoss(torch.nn.Module):
    def forward(self, pred, label):
        pred = torch.mean(pred, dim=1)
        loss = F.l1_loss(pred, label)
        return loss


if __name__ == "__main__":
    input_json = str(Path(__file__).parent / "post/input_finetune_3.json")
    # file_model_param = Path(CUR_DIR) / "post" / "OpenLAM_2.2.0_27heads_beta3.pt"
    file_model_param = Path(CUR_DIR) / "post" / "DPA2_medium_28_10M_rc0.pt"
    with open(input_json, "r") as f:
        input_dict = json.load(f)


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

    # model = FittingNet(in_dim=des.get_dim_out()+1, out_dim=1, neuron=[40, 40, 40])
    # loss = PostLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=trainer.lr_exp.start_lr)

    for s in range(1):
        ext_coord, ext_atype, force, coord, ae = get_data(trainer)


    # test the rf, first batch
    ae = ae[0]
    coord = coord[0]

    out = torch.cat((coord, ae), dim=1)

    np.savetxt("out_ae.xyz", out.detach().cpu().numpy())

    ext_coord = ext_coord[0]
    force = force[0]


    # for i in range(num_atoms):
    #     rf_ = rf[0][i]

    #     # print(rf_.shape, ext_coord.shape)

    #     # combine the ext_coord with rf
    #     out = torch.cat((ext_coord, ext_atype.unsqueeze(-1), rf_), dim=1)
    #     # print(out.shape)


    #     # save the rf and ext_coord to txt
    #     print("start save {} to txt".format(i))
    #     file_name = "out_{}.xyz".format(i)
    #     np.savetxt(file_name, out.cpu().numpy())

    #     # add a first column to the txt file
    #     with open(file_name, "r") as f:
    #         lines = f.readlines()
    #     with open(file_name, "w") as f:
    #         f.write(f"{num_ext_atoms}\n\n")
    #         f.write("".join(lines))







