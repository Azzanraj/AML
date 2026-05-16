import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBClassifier

# ---------- PATHS ----------
ART_DIR = "artifacts"

PREPROC_JSON = f"{ART_DIR}/ae_preproc_config_0p2.json"
SCALER_PKL   = f"{ART_DIR}/ae_num_scaler_0p2.pkl"
AE_MODEL_PTH = f"{ART_DIR}/autoencoder_v2_0p2.pth"

MV_EMB_NPY   = f"{ART_DIR}/node_embeddings_mv_sage_0p2.npy"
ACC_MAP_PKL  = f"{ART_DIR}/account_index_map_0p2.pkl"

XGB_PKL      = f"{ART_DIR}/xgboost_v4_ae_mvgnn_0p2.pkl"

# ---------- LOAD CONFIG ----------
with open(PREPROC_JSON) as f:
    cfg = json.load(f)

NUM_COLS     = cfg["num_cols"]
CAT_COLS     = cfg["cat_onehot_base_cols"]
ONEHOT_COLS  = cfg["onehot_cols"]
ACCOUNT_COL  = cfg["account_col"]

scaler = joblib.load(SCALER_PKL)
xgb_model: XGBClassifier = joblib.load(XGB_PKL)

embeddings = np.load(MV_EMB_NPY)
acc_map = joblib.load(ACC_MAP_PKL)

# ---------- AUTOENCODER ----------
class StrongAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


input_dim = len(NUM_COLS) + len(ONEHOT_COLS)
ae = StrongAutoencoder(input_dim, latent_dim=64)
ae.load_state_dict(torch.load(AE_MODEL_PTH, map_location="cpu"))
ae.eval()

# ---------- FEATURE BUILD ----------
def build_features(df: pd.DataFrame):
    df = df.copy()

    if "Amount_Diff" not in df:
        df["Amount_Diff"] = df["Amount Paid"] - df["Amount Received"]
    if "Amount_Ratio" not in df:
        df["Amount_Ratio"] = df["Amount Paid"] / (df["Amount Received"] + 1e-6)

    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_num = scaler.transform(df[NUM_COLS])

    df_cat = pd.get_dummies(df[CAT_COLS].astype(str))
    df_cat = df_cat.reindex(columns=ONEHOT_COLS, fill_value=0)

    X_ae = np.hstack([X_num, df_cat.values]).astype("float32")
    X_ae_t = torch.tensor(X_ae, dtype=torch.float32)

    with torch.no_grad():
        x_hat, Z = ae(X_ae_t)

    recon_error = ((x_hat - X_ae_t) ** 2).mean(dim=1, keepdim=True)

    Z = Z.cpu().numpy()
    recon_error = recon_error.cpu().numpy()

    mv = np.zeros((len(df), embeddings.shape[1]), dtype=np.float32)
    for i, acc in enumerate(df[ACCOUNT_COL].astype(str)):
        if acc in acc_map:
            mv[i] = embeddings[acc_map[acc]]

    X_final = np.hstack([Z, recon_error, mv])

    if X_final.shape[1] != 193:
        raise ValueError(f"Feature mismatch: got {X_final.shape[1]}, expected 193")

    return X_final


# ---------- PREDICT ----------
def predict(df: pd.DataFrame):
    X = build_features(df)
    probs = xgb_model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs
