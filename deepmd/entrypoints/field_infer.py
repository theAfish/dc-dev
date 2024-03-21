import logging
from pathlib import Path
from typing import Tuple, List, Optional

from deepmd.infer.deep_eval import DeepEval
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.infer.deep_pot import (
    DeepPot,
)

import numpy as np

__all__ = ["field_infer"]

log = logging.getLogger(__name__)

def field_infer(
    *,
    model: str,
    structure: str,
    detail_file: str,
    head: Optional[str] = None,
    **kwargs,
):
    dp = DeepEval(model, head=head)

    tmap = dp.get_type_map() if isinstance(dp, DeepPot) else None
    data = DeepmdData(
        structure,
        "set",
        shuffle_test=False,
        type_map=tmap,
        sort_atoms=False,
    )

    if isinstance(dp, DeepPot):
        err = test_ener(
            dp,
            data,
            structure,
            0,
            detail_file
        )


def test_ener(
    dp: "DeepPot",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: Optional[str],
    append_detail: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data : DeepmdData
        data container object
    system : str
        system directory
    numb_test : int
        number of tests to do
    detail_file : Optional[str]
        file where test details will be output
    append_detail : bool, optional
        if true append output detail file, by default False

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add("rho", 1, atomic=True, must=False, high_prec=True)
    if dp.has_efield:
        data.add("efield", 3, atomic=True, must=True, high_prec=False)
    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)
    if dp.has_spin:
        data.add("spin", 3, atomic=True, must=True, high_prec=False)

    test_data = data.get_test()
    mixed_type = data.mixed_type
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    if dp.has_efield:
        efield = test_data["efield"][:numb_test].reshape([numb_test, -1])
    else:
        efield = None
    if dp.has_spin:
        spin = test_data["spin"][:numb_test].reshape([numb_test, -1])
    else:
        spin = None
    if not data.pbc:
        box = None
    if mixed_type:
        atype = test_data["type"][:numb_test].reshape([numb_test, -1])
    else:
        atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test]
    else:
        aparam = None

    ret = dp.eval(
        coord,
        box,
        atype,
        fparam=fparam,
        aparam=aparam,
        efield=efield,
        mixed_type=mixed_type,
        spin=spin,
    )
    energy = ret[0]
    energy = energy.reshape([numb_test, 1])

    out_put_spin = dp.get_ntypes_spin() != 0 or dp.has_spin

    diff_e = energy - test_data["energy"][:numb_test].reshape([-1, 1])

    log.info(f"# number of test data : {numb_test:d} ")
    log.info(f"Energy MAE         : {diff_e:e} eV")

    if detail_file is not None:
        detail_path = Path(detail_file)

        pe = np.concatenate(
            (
                np.reshape(test_data["energy"][:numb_test], [-1, 1]),
                np.reshape(energy, [-1, 1]),
            ),
            axis=1,
        )
        save_txt_file(
            detail_path.with_suffix(".e.out"),
            pe,
            header="%s: data_e pred_e" % system,
            append=append_detail,
        )
        pe_atom = pe / natoms
        save_txt_file(
            detail_path.with_suffix(".e_peratom.out"),
            pe_atom,
            header="%s: data_e pred_e" % system,
            append=append_detail,
        )

    mae_e = None
    mae_ea = None
    return {
        "mae_e": (mae_e, energy.size),
        "mae_ea": (mae_ea, energy.size),
    }


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
):
    """Save numpy array to test file.

    Parameters
    ----------
    fname : str
        filename
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended insted of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)