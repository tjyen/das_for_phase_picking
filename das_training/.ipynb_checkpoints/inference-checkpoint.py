# das_training/inference.py
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from .myutils import postprocess, normalize, detect_peaks, extract_picks

def pred_phasenet_das(model, data_loader, pick_path, figure_path):
    with torch.inference_mode():
        for meta in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            output = model(meta)
            meta, output = postprocess(meta, output)
            scores = torch.softmax(output["phase"], dim=1)
            topk_scores, topk_inds = detect_peaks(scores, vmin=0.3, kernel=21)
            picks_ = extract_picks(
                topk_inds, topk_scores, file_name=meta["file_name"], begin_time=meta.get("begin_time"),
                begin_time_index=meta.get("begin_time_index"), begin_channel_index=meta.get("begin_channel_index"),
                dt=meta.get("dt_s", 0.01), vmin=0.3, phases=['P', 'S']
            )
            for i in range(len(meta["file_name"])):
                tmp = meta["file_name"][i].split("/")
                parent_dir = "/".join(tmp[:-1])
                filename = tmp[-1].replace("*", "").replace(".h5", "")
                if not os.path.exists(os.path.join(pick_path, parent_dir)):
                    os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                if len(picks_[i]) == 0:
                    with open(os.path.join(pick_path, parent_dir, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(picks_[i])
                picks_df["channel_index"] = picks_df["station_id"].apply(lambda x: int(x))
                picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(pick_path, parent_dir, filename + ".csv"),
                    columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type"],
                    index=False
                )
    return meta, output

def pred_phasenet_das_new(model, data_loader, pick_path, figure_path, vmin):
    with torch.inference_mode():
        for meta in tqdm(data_loader, desc="Predicting", total=len(data_loader)):
            output = model(meta)
            meta, output = postprocess(meta, output)
            scores = torch.softmax(output["phase"], dim=1)
            topk_scores, topk_inds = detect_peaks(scores, vmin=vmin, kernel=21)
            picks_ = extract_picks(
                topk_inds, topk_scores, file_name=meta["file_name"], begin_time=meta.get("begin_time"),
                begin_time_index=meta.get("begin_time_index"), begin_channel_index=meta.get("begin_channel_index"),
                dt=meta.get("dt_s", 0.01), vmin=vmin, phases=['P_u', 'P_d', 'S_u', 'S_d']
            )
            for i in range(len(meta["file_name"])):
                tmp = meta["file_name"][i].split("/")
                parent_dir = "/".join(tmp[:-1])
                filename = tmp[-1].replace("*", "").replace(".h5", "")
                if not os.path.exists(os.path.join(pick_path, parent_dir)):
                    os.makedirs(os.path.join(pick_path, parent_dir), exist_ok=True)
                if len(picks_[i]) == 0:
                    with open(os.path.join(pick_path, parent_dir, filename + ".csv"), "a"):
                        pass
                    continue
                picks_df = pd.DataFrame(picks_[i])
                picks_df["channel_index"] = picks_df["station_id"].apply(lambda x: int(x))
                picks_df.sort_values(by=["channel_index", "phase_index"], inplace=True)
                picks_df.to_csv(
                    os.path.join(pick_path, parent_dir, filename + ".csv"),
                    columns=["channel_index", "phase_index", "phase_time", "phase_score", "phase_type"],
                    index=False
                )
    return meta, output