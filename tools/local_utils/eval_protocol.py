import numpy as np
from .mos_utils import fit_logistic, plcc, srcc, krcc

def eval_with_mapping(pred, mos, use_5p=True):
    pred = np.asarray(pred); mos = np.asarray(mos)
    mapped, beta = fit_logistic(pred, mos, use_5p=use_5p)
    rmse = float(np.sqrt(np.mean((mapped - mos) ** 2)))
    return {"PLCC": plcc(mapped, mos), "SRCC": srcc(pred, mos), "KRCC": krcc(pred, mos), "RMSE": rmse, "beta": beta}
