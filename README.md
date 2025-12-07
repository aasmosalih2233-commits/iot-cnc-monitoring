# iot-cnc-monitoring
CNC vibration monitoring project (IoT &amp; Data Science)
project-root/
│
├── data/
│   ├── raw/                # raw downloaded dataset (do not commit large files)
│   └── processed/          # processed / downsampled files
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_preprocessing_feature_engineering.ipynb
│
├── src/
│   ├── io.py               # dataset loader & save/load helpers
│   ├── preprocess.py       # cleaning & windowing
│   ├── features.py         # feature extraction
│   └── model.py            # train / inference / save / load
│
├── dashboards/
│   └── app.py              # Streamlit dashboard
│
├── scripts/
│   ├── train.py            # CLI to train baseline model
│   └── simulate_stream.py  # simulate streaming to measure latency
│
├── models/
│   └── baseline_model.pkl
│
├── docs/
│   └── report_template.md
│
├── requirements.txt
└── README.md

# src/io.py
import os
import pandas as pd
import numpy as np

def load_csv(filepath):
    """Load a CSV; returns DataFrame."""
    return pd.read_csv(filepath)

def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path):
    return pd.read_parquet(path)

# src/preprocess.py
import numpy as np
import pandas as pd
from scipy import signal

def downsample_df(df, factor=10, axis_cols=['acc_x','acc_y','acc_z']):
    """Simple downsample by averaging `factor` samples to reduce 2kHz -> lower rate."""
    n = len(df)
    k = factor
    # group into non-overlapping windows
    trimmed = (n // k) * k
    arr = df[axis_cols].values[:trimmed].reshape(-1, k, len(axis_cols)).mean(axis=1)
    idx = np.arange(trimmed//k)
    out = pd.DataFrame(arr, columns=axis_cols)
    # preserve timestamp if exists, else create pseudo timestamps
    if 'timestamp' in df.columns:
        out['timestamp'] = df['timestamp'].values[:trimmed].reshape(-1, k).mean(axis=1)
    else:
        out['timestamp'] = idx
    # carry label if exists (majority in window)
    if 'label' in df.columns:
        labels = df['label'].values[:trimmed].reshape(-1, k)
        # majority vote
        out['label'] = [np.bincount(win.astype(int)).argmax() for win in labels]
    return out

def sliding_windows(df, window_size:int, step:int, axis_cols=['acc_x','acc_y','acc_z']):
    """Return a generator of windows (DataFrame slices) with given size and step (in rows)."""
    n = len(df)
    for start in range(0, n - window_size + 1, step):
        yield df.iloc[start:start+window_size]

# src/features.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_basic_time_features(window_df, axis_cols=['acc_x','acc_y','acc_z']):
    feats = {}
    for col in axis_cols:
        arr = window_df[col].values
        feats[f'{col}_mean'] = arr.mean()
        feats[f'{col}_std'] = arr.std()
        feats[f'{col}_max'] = arr.max()
        feats[f'{col}_min'] = arr.min()
        feats[f'{col}_rms'] = np.sqrt((arr**2).mean())
        feats[f'{col}_skew'] = skew(arr)
        feats[f'{col}_kurt'] = kurtosis(arr)
    return feats

def extract_frequency_band_energy(window_df, fs=2000, axis='acc_x'):
    # compute PSD using Welch and return band energies
    f, Pxx = welch(window_df[axis].values, fs=fs, nperseg=256)
    bands = {'0-50': (0,50),'50-150':(50,150),'150-500':(150,500),'500-1000':(500,1000)}
    feats = {}
    for k,v in bands.items():
        idx = (f>=v[0]) & (f<v[1])
        feats[f'{axis}_band_{k}_energy'] = Pxx[idx].sum()
    return feats

def extract_features_for_window(window_df, fs=2000, axis_cols=['acc_x','acc_y','acc_z']):
    feat = extract_basic_time_features(window_df, axis_cols)
    # add freq features for each axis
    for axis in axis_cols:
        freq_feats = extract_frequency_band_energy(window_df, fs=fs, axis=axis)
        feat.update(freq_feats)
    return feat

# src/model.py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

def train_baseline(X_train, y_train, n_estimators=100):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))
    return report

def save_model(clf, path='models/baseline_model.pkl'):
    joblib.dump(clf, path)

def load_model(path='models/baseline_model.pkl'):
    return joblib.load(path)

# scripts/train.py
import os
import pandas as pd
from src.preprocess import downsample_df, sliding_windows
from src.features import extract_features_for_window
from src.model import train_baseline, save_model, evaluate
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

DATA_PATH = "data/processed/downsampled.parquet"

def main():
    df = pd.read_parquet(DATA_PATH)
    # window params (rows after downsampling): 1s windows at new rate
    window_size = 200  # adjust depending on downsample factor
    step = 100
    axis_cols = ['acc_x','acc_y','acc_z']
    X = []
    y = []
    for win in tqdm(list(sliding_windows(df, window_size, step, axis_cols))):
        feats = extract_features_for_window(win, fs=200)  # fs after downsample
        X.append(feats)
        if 'label' in win.columns:
            y.append(win['label'].iloc[-1])
        else:
            y.append(0)  # placeholder
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = train_baseline(X_train, y_train)
    report = evaluate(clf, X_test, y_test)
    save_model(clf)
    pd.DataFrame(report).T.to_csv('models/eval_report.csv')
    print("Model saved to models/baseline_model.pkl")

if __name__ == "__main__":
    main()

# scripts/simulate_stream.py
import time
import numpy as np
import pandas as pd
from src.model import load_model
from src.preprocess import sliding_windows
from src.features import extract_features_for_window
import timeit

MODEL_PATH = "models/baseline_model.pkl"
DATA_PATH = "data/processed/downsampled.parquet"

def main():
    model = load_model(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)
    window_size = 200
    step = 100
    axis_cols = ['acc_x','acc_y','acc_z']
    latencies = []
    labels = []
    for win in sliding_windows(df, window_size, step, axis_cols):
        t0 = timeit.default_timer()
        feat = extract_features_for_window(win, fs=200, axis_cols=axis_cols)
        X = pd.DataFrame([feat])
        pred = model.predict(X)[0]
        t1 = timeit.default_timer()
        latencies.append((t1 - t0))
        labels.append(pred)
        # simulate wait as if streaming: here no wait to measure pure inference time
    print("Avg inference latency (s):", sum(latencies)/len(latencies))
    print("Max inference latency (s):", max(latencies))

if __name__ == "__main__":
    main()

# dashboards/app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.model import load_model
from src.features import extract_features_for_window
from src.preprocess import sliding_windows
import time

st.set_page_config(layout="wide", page_title="CNC Vibration Monitor")

st.title("CNC Vibration Monitoring Dashboard — Demo")
st.sidebar.header("Controls")
data_path = st.sidebar.text_input("Path to processed data", "data/processed/downsampled.parquet")
model_path = st.sidebar.text_input("Model path", "models/baseline_model.pkl")
run_stream = st.sidebar.button("Simulate Stream (Run once)")

st.markdown("**Load summary**")
if st.button("Load data and show preview"):
    df = pd.read_parquet(data_path)
    st.write("Data preview (first 200 rows):")
    st.dataframe(df.head(200))
    st.line_chart(df[['acc_x','acc_y','acc_z']].head(1000))

if run_stream:
    model = load_model(model_path)
    df = pd.read_parquet(data_path)
    window_size = 200
    step = 100
    status_text = st.empty()
    chart = st.empty()
    anomalies = 0
    t_start = time.time()
    for i, win in enumerate(sliding_windows(df, window_size, step, ['acc_x','acc_y','acc_z'])):
        feat = extract_features_for_window(win, fs=200)
        X = pd.DataFrame([feat])
        pred = model.predict(X)[0]
        if pred == 1:
            anomalies += 1
            st.warning(f"Anomaly detected at window {i}")
        if i % 10 == 0:
            chart.line_chart(win[['acc_x','acc_y','acc_z']].reset_index(drop=True).iloc[:200])
        status_text.text(f"Processed windows: {i+1} — anomalies: {anomalies}")
        # small sleep to let UI update - in real system this is triggered by incoming data
        time.sleep(0.1)
    t_end = time.time()
    st.success(f"Stream simulation complete. Processed windows: {i+1}. Time: {t_end-t_start:.2f}s")

# scripts/prepare_data.py
import pandas as pd
from src.preprocess import downsample_df
df = pd.read_csv("data/raw/your_vibration_file.csv")  # adjust filename
df_d = downsample_df(df, factor=10, axis_cols=['acc_x','acc_y','acc_z'])
df_d.to_parquet("data/processed/downsampled.parquet", index=False)
