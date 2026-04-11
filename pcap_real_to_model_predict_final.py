#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chuẩn hóa traffic.pcap thực -> build flow feature -> ép đúng feature_cols.pkl -> predict bằng model.pkl

Điểm mạnh của bản này:
- Đọc PCAP thô
- Parse Ethernet + IPv4 + TCP/UDP
- Gom flow 2 chiều
- Build các flow feature phổ biến
- Kiểm tra feature traffic so với feature AI
- Tự thêm cột thiếu = 0.0
- Tự bỏ cột dư khi chọn theo feature_cols.pkl
- Ép đúng thứ tự cột tuyệt đối
- Làm sạch NaN / inf / sai kiểu
- Cảnh báo các cột toàn 0
- Predict Label + Risk Score + Risk Level

Ví dụ chạy:
python pcap_real_to_model_predict_final.py \
    --pcap /kaggle/input/yourdata/traffic.pcap \
    --model /kaggle/working/model.pkl \
    --feature-cols /kaggle/working/feature_cols.pkl \
    --out /kaggle/working/pcap_predict_result.csv
"""

import argparse
import pickle
import struct
import socket
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 1) LOAD MODEL / FEATURE COLS
# =========================================================
def load_pickle_or_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# =========================================================
# 2) ĐỌC PCAP THÔ
# =========================================================
def parse_pcap_packets(path):
    packets = []

    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            raise ValueError("File pcap không hợp lệ hoặc quá ngắn.")

        magic = gh[:4]

        if magic in (b'\xd4\xc3\xb2\xa1', b'M<\xb2\xa1'):
            endian = "<"
        elif magic in (b'\xa1\xb2\xc3\xd4', b'\xa1\xb2<M'):
            endian = ">"
        else:
            raise ValueError("Không nhận diện được PCAP magic number.")

        while True:
            ph = f.read(16)
            if len(ph) < 16:
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", ph)
            data = f.read(incl_len)
            ts = ts_sec + ts_usec / 1_000_000.0
            packets.append((ts, data))

    return packets


# =========================================================
# 3) PARSE ETHERNET + IPv4 + TCP/UDP
# =========================================================
def parse_ipv4_tcp_udp(frame):
    if len(frame) < 14:
        return None

    eth_type = struct.unpack("!H", frame[12:14])[0]
    if eth_type != 0x0800:
        return None

    ip = frame[14:]
    if len(ip) < 20:
        return None

    version = ip[0] >> 4
    if version != 4:
        return None

    ihl = (ip[0] & 0x0F) * 4
    if len(ip) < ihl:
        return None

    proto = ip[9]
    src_ip = socket.inet_ntoa(ip[12:16])
    dst_ip = socket.inet_ntoa(ip[16:20])
    total_len = struct.unpack("!H", ip[2:4])[0]

    # TCP
    if proto == 6:
        if len(ip) < ihl + 20:
            return None

        tcp = ip[ihl:]
        src_port, dst_port = struct.unpack("!HH", tcp[:4])

        data_offset = (tcp[12] >> 4) * 4
        flags_byte = tcp[13]

        syn = 1 if (flags_byte & 0x02) else 0
        ack = 1 if (flags_byte & 0x10) else 0
        fin = 1 if (flags_byte & 0x01) else 0
        rst = 1 if (flags_byte & 0x04) else 0
        psh = 1 if (flags_byte & 0x08) else 0
        urg = 1 if (flags_byte & 0x20) else 0

        payload_len = max(total_len - ihl - data_offset, 0)

        return {
            "proto_num": 6,
            "proto_name": "TCP",
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(src_port),
            "dst_port": int(dst_port),
            "packet_len": int(total_len),
            "payload_len": int(payload_len),
            "SYN": syn,
            "ACK": ack,
            "FIN": fin,
            "RST": rst,
            "PSH": psh,
            "URG": urg,
        }

    # UDP
    if proto == 17:
        if len(ip) < ihl + 8:
            return None

        udp = ip[ihl:]
        src_port, dst_port, udp_len, _ = struct.unpack("!HHHH", udp[:8])
        payload_len = max(udp_len - 8, 0)

        return {
            "proto_num": 17,
            "proto_name": "UDP",
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": int(src_port),
            "dst_port": int(dst_port),
            "packet_len": int(total_len),
            "payload_len": int(payload_len),
            "SYN": 0,
            "ACK": 0,
            "FIN": 0,
            "RST": 0,
            "PSH": 0,
            "URG": 0,
        }

    return None


# =========================================================
# 4) GOM FLOW 2 CHIỀU
# =========================================================
def make_bidirectional_key(pkt):
    a = (pkt["src_ip"], pkt["src_port"])
    b = (pkt["dst_ip"], pkt["dst_port"])

    if a <= b:
        return (pkt["proto_num"], a[0], a[1], b[0], b[1])
    return (pkt["proto_num"], b[0], b[1], a[0], a[1])


def build_flow_table_from_pcap(pcap_path):
    flows = {}

    for ts, raw in parse_pcap_packets(pcap_path):
        pkt = parse_ipv4_tcp_udp(raw)
        if pkt is None:
            continue

        key = make_bidirectional_key(pkt)
        real_dir = (pkt["src_ip"], pkt["src_port"], pkt["dst_ip"], pkt["dst_port"])

        if key not in flows:
            flows[key] = {
                "proto_num": pkt["proto_num"],
                "proto_name": pkt["proto_name"],
                "first_dir": real_dir,
                "src_ip": pkt["src_ip"],
                "src_port": pkt["src_port"],
                "dst_ip": pkt["dst_ip"],
                "dst_port": pkt["dst_port"],
                "times": [],
                "pkt_lens": [],
                "payload_lens": [],
                "fwd_count": 0,
                "bwd_count": 0,
                "fwd_bytes": 0,
                "bwd_bytes": 0,
                "syn_count": 0,
                "ack_count": 0,
                "fin_count": 0,
                "rst_count": 0,
                "psh_count": 0,
                "urg_count": 0,
            }

        flow = flows[key]
        flow["times"].append(ts)
        flow["pkt_lens"].append(pkt["packet_len"])
        flow["payload_lens"].append(pkt["payload_len"])

        if real_dir == flow["first_dir"]:
            flow["fwd_count"] += 1
            flow["fwd_bytes"] += pkt["packet_len"]
        else:
            flow["bwd_count"] += 1
            flow["bwd_bytes"] += pkt["packet_len"]

        flow["syn_count"] += pkt["SYN"]
        flow["ack_count"] += pkt["ACK"]
        flow["fin_count"] += pkt["FIN"]
        flow["rst_count"] += pkt["RST"]
        flow["psh_count"] += pkt["PSH"]
        flow["urg_count"] += pkt["URG"]

    rows = []

    for _, flow in flows.items():
        times = sorted(flow["times"])
        total_packets = len(times)
        total_bytes = sum(flow["pkt_lens"])
        duration = (max(times) - min(times)) if total_packets > 1 else 0.0

        if total_packets > 1:
            iats = np.diff(times)
            flow_iat_mean = float(np.mean(iats))
            flow_iat_std = float(np.std(iats))
            flow_iat_max = float(np.max(iats))
            flow_iat_min = float(np.min(iats))
        else:
            flow_iat_mean = 0.0
            flow_iat_std = 0.0
            flow_iat_max = 0.0
            flow_iat_min = 0.0

        flow_packets_s = (total_packets / duration) if duration > 0 else 0.0
        flow_bytes_s = (total_bytes / duration) if duration > 0 else 0.0

        total_fwd = int(flow["fwd_count"])
        total_bwd = int(flow["bwd_count"])

        down_up_ratio = (total_bwd / total_fwd) if total_fwd > 0 else 0.0
        avg_pkt_len = float(np.mean(flow["pkt_lens"])) if flow["pkt_lens"] else 0.0
        pkt_len_mean = avg_pkt_len
        pkt_len_std = float(np.std(flow["pkt_lens"])) if flow["pkt_lens"] else 0.0
        pkt_len_max = float(np.max(flow["pkt_lens"])) if flow["pkt_lens"] else 0.0
        pkt_len_min = float(np.min(flow["pkt_lens"])) if flow["pkt_lens"] else 0.0

        syn_ack_ratio = (
            flow["syn_count"] / (flow["ack_count"] + 1e-9)
            if flow["ack_count"] > 0 else float(flow["syn_count"])
        )
        rst_ratio = (flow["rst_count"] / total_packets) if total_packets > 0 else 0.0

        row = {
            # metadata
            "Proto": flow["proto_name"],
            "Protocol": flow["proto_num"],
            "Src IP": flow["src_ip"],
            "Src Port": flow["src_port"],
            "Dst IP": flow["dst_ip"],
            "Dst Port": flow["dst_port"],

            # feature phổ biến
            "Flow Duration": float(duration),
            "Flow Packets/s": float(flow_packets_s),
            "Flow Bytes/s": float(flow_bytes_s),
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            "Total Length of Fwd Packets": float(flow["fwd_bytes"]),
            "Total Length of Bwd Packets": float(flow["bwd_bytes"]),
            "Average Packet Size": float(avg_pkt_len),
            "Packet Length Mean": float(pkt_len_mean),
            "Packet Length Std": float(pkt_len_std),
            "Packet Length Max": float(pkt_len_max),
            "Packet Length Min": float(pkt_len_min),
            "Flow IAT Mean": float(flow_iat_mean),
            "Flow IAT Std": float(flow_iat_std),
            "Flow IAT Max": float(flow_iat_max),
            "Flow IAT Min": float(flow_iat_min),
            "Down/Up Ratio": float(down_up_ratio),

            # feature tự thiết kế thường gặp
            "SYN Count": int(flow["syn_count"]),
            "ACK Count": int(flow["ack_count"]),
            "FIN Count": int(flow["fin_count"]),
            "RST Count": int(flow["rst_count"]),
            "PSH Count": int(flow["psh_count"]),
            "URG Count": int(flow["urg_count"]),
            "SYN/ACK Ratio": float(syn_ack_ratio),
            "RST Ratio": float(rst_ratio),
            "Packet Count": int(total_packets),
            "Byte Count": float(total_bytes),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Không build được flow nào từ pcap.")

    return df


# =========================================================
# 5) CHUẨN HÓA THEO feature_cols.pkl
# =========================================================
def standardize_to_model_features(flow_df, feature_cols):
    df = flow_df.copy()

    # alias tên cột phổ biến
    alias_map = {
        "Avg Packet Size": "Average Packet Size",
        "Pkt Len Mean": "Packet Length Mean",
        "Pkt Len Std": "Packet Length Std",
        "Pkt Len Max": "Packet Length Max",
        "Pkt Len Min": "Packet Length Min",
        "Fwd Pkt Count": "Total Fwd Packets",
        "Bwd Pkt Count": "Total Backward Packets",
        "Fwd Packet Count": "Total Fwd Packets",
        "Bwd Packet Count": "Total Backward Packets",
    }

    for wrong_name, right_name in alias_map.items():
        if wrong_name in df.columns and right_name not in df.columns:
            df[right_name] = df[wrong_name]

    df_cols_before = list(df.columns)
    df_col_set_before = set(df_cols_before)
    model_col_set = set(feature_cols)

    # cột thiếu / cột dư trước khi fill
    missing_cols = [col for col in feature_cols if col not in df_col_set_before]
    extra_cols = [col for col in df_cols_before if col not in model_col_set]

    print("\n===== CHECK FEATURE MISMATCH =====")
    print("Số cột model cần:", len(feature_cols))
    print("Số cột traffic đang có:", len(df_cols_before))
    print("Thiếu cột:", missing_cols if missing_cols else "Không")
    print("Dư cột:", extra_cols if extra_cols else "Không")

    # thêm cột thiếu = 0.0
    for col in missing_cols:
        df[col] = 0.0

    # chỉ lấy đúng cột model cần, đúng thứ tự
    X = df[feature_cols].copy()

    # ép numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # làm sạch
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # ép lại thứ tự lần cuối
    X = X[feature_cols]

    # kiểm tra cột toàn 0
    zero_only_cols = [col for col in X.columns if (X[col] == 0).all()]

    print("\n===== FINAL FEATURE CHECK =====")
    print("Đúng thứ tự cột model:", list(X.columns) == list(feature_cols))
    print("Shape X:", X.shape)
    print("Cột toàn 0:", zero_only_cols if zero_only_cols else "Không")

    if len(zero_only_cols) > 0:
        ratio = len(zero_only_cols) / max(len(feature_cols), 1)
        print("Tỷ lệ cột toàn 0:", round(ratio * 100, 2), "%")

    return X, df, missing_cols, extra_cols, zero_only_cols


# =========================================================
# 6) PREDICT
# =========================================================
def predict_with_model(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            attack_prob = probs[:, 1]
        else:
            attack_prob = np.asarray(y_pred, dtype=float)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        raw = np.asarray(raw, dtype=float)
        mn, mx = raw.min(), raw.max()
        attack_prob = (raw - mn) / (mx - mn + 1e-9)
    else:
        attack_prob = np.asarray(y_pred, dtype=float)

    return np.asarray(y_pred).astype(int), np.asarray(attack_prob).astype(float)


def map_risk(score_0_100):
    if score_0_100 < 30:
        return "Low"
    if score_0_100 < 70:
        return "Medium"
    return "High"


# =========================================================
# 7) MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", required=True, help="Đường dẫn file pcap thực")
    parser.add_argument("--model", default="/kaggle/working/model.pkl", help="Đường dẫn model.pkl")
    parser.add_argument("--feature-cols", default="/kaggle/working/feature_cols.pkl", help="Đường dẫn feature_cols.pkl")
    parser.add_argument("--out", default="/kaggle/working/pcap_predict_result.csv", help="File CSV output")
    args = parser.parse_args()

    print("===== 1. LOAD MODEL + FEATURE COLS =====")
    model = load_pickle_or_joblib(args.model)
    feature_cols = load_pickle_or_joblib(args.feature_cols)

    if not isinstance(feature_cols, (list, tuple)):
        raise ValueError("feature_cols.pkl phải là list hoặc tuple tên cột.")

    feature_cols = list(feature_cols)

    print("Model:", type(model))
    print("Số feature model cần:", len(feature_cols))
    print("Feature cols:", feature_cols)

    print("\n===== 2. BUILD FLOW FEATURE TỪ PCAP =====")
    flow_df = build_flow_table_from_pcap(args.pcap)
    print("Số flow trích được:", len(flow_df))
    print("Các cột flow build được:")
    print(list(flow_df.columns))

    print("\n===== 3. CHUẨN HÓA THEO feature_cols.pkl =====")
    X, meta_df, missing_cols, extra_cols, zero_only_cols = standardize_to_model_features(
        flow_df, feature_cols
    )
    print(X.head())

    print("\n===== 4. PREDICT LABEL + RISK =====")
    y_pred, attack_prob = predict_with_model(model, X)

    result_df = meta_df.copy()
    result_df["Label"] = y_pred
    result_df["Attack_Probability"] = attack_prob
    result_df["Risk_Score"] = np.round(attack_prob * 100, 2)
    result_df["Risk_Level"] = result_df["Risk_Score"].apply(map_risk)

    print(result_df[[
        "Src IP", "Src Port", "Dst IP", "Dst Port",
        "Label", "Attack_Probability", "Risk_Score", "Risk_Level"
    ]].head(20))

    result_df.to_csv(args.out, index=False)

    print("\n===== 5. THỐNG KÊ =====")
    print("Label distribution:")
    print(result_df["Label"].value_counts(dropna=False))

    print("\nRisk distribution:")
    print(result_df["Risk_Level"].value_counts(dropna=False))

    print("\nTop flow nguy hiểm nhất:")
    top_cols = [c for c in [
        "Proto", "Src IP", "Src Port", "Dst IP", "Dst Port",
        "Flow Packets/s", "Total Fwd Packets", "Flow IAT Mean",
        "Total Backward Packets", "Down/Up Ratio",
        "Label", "Attack_Probability", "Risk_Score", "Risk_Level"
    ] if c in result_df.columns]
    print(result_df.sort_values("Risk_Score", ascending=False)[top_cols].head(10))

    print("\n===== 6. TÓM TẮT CHECK FEATURE =====")
    print("Missing columns:", missing_cols if missing_cols else "Không")
    print("Extra columns:", extra_cols if extra_cols else "Không")
    print("Zero-only columns:", zero_only_cols if zero_only_cols else "Không")
    print("Output file:", args.out)


if __name__ == "__main__":
    main()
