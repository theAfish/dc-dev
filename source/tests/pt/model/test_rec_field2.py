import torch
from deepmd.pt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
    RESERVED_PRECISON_DICT,
)
from deepmd.infer.receptive_field import ReceptiveField
import ase
import ase.io.extxyz
import numpy as np

from pathlib import (
    Path,
)


model_path = "source/tests/pt/model/post/DPA2_medium_28_10M_rc0.pt"
head = "MP_traj_v024_alldata_mixu"


dp = ReceptiveField(
    str(Path(model_path).resolve()),
    neighbor_list=None,
    head=head,
)
type_dict = dict(zip(dp.get_type_map(), range(dp.get_ntypes())))

for p in range(100):
    # load some data using ase poscar
    atoms = ase.io.read(f"/mnt/d/Codes/mat-data/out/diamond-sc-{p}.poscar")
    # atoms = ase.io.read(f"/mnt/d/data/rec_field/bccTi-sc.poscar")
    coord = torch.tensor(atoms.get_positions()).unsqueeze(0)
    # atype = torch.tensor(atoms.get_atomic_numbers()).unsqueeze(0)
    cell = torch.tensor(np.array(atoms.get_cell())).unsqueeze(0)
    symbols = atoms.get_chemical_symbols()
    atype = [type_dict[k] for k in symbols]

    num_atoms = coord.shape[1]

    results = dp.eval(coords=coord, cells=cell, atom_types=atype)

    for i in range(num_atoms):
        atoms.arrays["force"] = results[0][i, :]
        ase.io.extxyz.write_extxyz("rec_f-{}.xyz".format(p), atoms, write_info=True, append=True, comment="Time={}".format(i))
    
    print(f"Done with {p}")

print("Done")