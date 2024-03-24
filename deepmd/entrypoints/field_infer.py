import logging
from pathlib import Path
from typing import Tuple, List, Optional

from deepmd.infer.deep_eval import DeepEval
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.infer.deep_charge import (
    DeepCharge,
)

import numpy as np

__all__ = ["field_infer"]

log = logging.getLogger(__name__)


def field_infer(
    *,
    model: str,
    structure: str,
    output: str,
    batch_size: int,
    x_size: int,
    y_size: int,
    z_size: int,
    head: Optional[str] = None,
    **kwargs,
):
    dp = DeepEval(model, head=head)

    if batch_size == 0:
        batch_size = int(x_size * y_size)

    tmap = dp.get_type_map() if isinstance(dp, DeepCharge) else None
    data = DeepmdData(
        structure,
        "set",
        shuffle_test=False,
        type_map=tmap,
        sort_atoms=False,
    )



    if isinstance(dp, DeepCharge):
        err = pred_rho(
            dp,
            data,
            structure,
            output,
            batch_size,
            x_size,
            y_size,
            z_size,
        )


def pred_rho(
    dp: "DeepPot",
    data: DeepmdData,
    system: str,
    output: Optional[str],
    batch_size: int,
    x: int,
    y: int,
    z: int,
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

    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)


    test_data = data.get_test()
    mixed_type = data.mixed_type
    natoms = len(test_data["type"][0])

    coord = test_data["coord"]
    box = test_data["box"]

    if not data.pbc:
        box = None
    if mixed_type:
        atype = test_data["type"]
    else:
        atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"]
    else:
        aparam = None

    raw_grid = create_grid_points([x, y, z])  # create grid points
    raw_grid = raw_grid @ box.reshape(3, 3)
    raw_grid = raw_grid.reshape(-1, batch_size, 3)

    _atype = np.concatenate((atype, np.zeros([1, batch_size])), axis=1)
    results = []

    for i in range(raw_grid.shape[0]):
        grid = raw_grid[i]
        _coord = np.concatenate((coord, grid.reshape(-1)[None, :]), axis=1)

        ret = dp.eval(
            coord,
            box,
            atype,
            fparam=fparam,
            aparam=aparam,
        )
        rho = ret[0][0, -batch_size:]
        result = np.concatenate((grid, rho), axis=1)
        results.append(result)

    results = np.concatenate(results, axis=0)
    np.savetxt(output, results, header="x y z rho", comments="")

    diff_e = rho - test_data["rho"][:numb_test].reshape([-1, 1])

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


def create_grid_points(grid_size):
    zeros = np.zeros(grid_size)
    label = np.where(zeros == 0)
    points = np.transpose(label) / grid_size
    return points.astype(np.float64)