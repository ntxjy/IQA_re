import argparse, csv, numpy as np
from tools.local_utils.eval_protocol import eval_with_mapping

def read_csv(path, key, val):
    M = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            M[r[key]] = float(r[val])
    return M

def merge_pred_gt(pred_csv, gt_csv):
    P, G = read_csv(pred_csv, "id", "pred"), read_csv(gt_csv, "id", "mos")
    ids = sorted(set(P) & set(G))
    pred = np.array([P[i] for i in ids], float)
    gt   = np.array([G[i] for i in ids], float)
    return ids, pred, gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--use_5p", action="store_true")
    args = ap.parse_args()
    ids, pred, gt = merge_pred_gt(args.pred, args.gt)
    res = eval_with_mapping(pred, gt, use_5p=args.use_5p)
    print(res)

if __name__ == "__main__":
    main()
