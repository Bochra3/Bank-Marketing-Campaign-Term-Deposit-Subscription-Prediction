from pathlib import Path

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from xgboost import XGBClassifier

DEFAULT_SEED = 42
TOP_INPUT_FEATURES = ["contact", "housing", "duration", "poutcome", "previous"]

st.set_page_config(page_title="Term Deposit Prediction", page_icon=":sparkles:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #eef7ff 0%, #f8fbff 45%, #ffffff 100%);
    }
    .hero {
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: linear-gradient(120deg, #0f172a, #1d4ed8);
        color: white;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "deposit" in data.columns and "term_deposit" not in data.columns:
        data = data.rename(columns={"deposit": "term_deposit"})
    if "term_deposit" not in data.columns:
        raise ValueError("Dataset must include 'deposit' or 'term_deposit'.")
    return data


def split_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "term_deposit" in cat_cols:
        cat_cols.remove("term_deposit")
    return num_cols, cat_cols


@st.cache_data(show_spinner=False)
def read_data(path: str) -> pd.DataFrame:
    return normalize_dataset(pd.read_csv(path))


@st.cache_resource(show_spinner=True)
def train_xgboost_model(
    df: pd.DataFrame,
    random_state: int,
    test_size: float,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_min_child_weight: int,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
) -> dict:
    num_cols, cat_cols = split_features(df)
    y = (df["term_deposit"].astype(str).str.lower() == "yes").astype(int).values

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_cat = pd.DataFrame(oe.fit_transform(df[cat_cols]), columns=cat_cols)
    x_raw = pd.concat([df[num_cols].reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_res, y_res = SMOTE(random_state=random_state).fit_resample(x_train_scaled, y_train)

    model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        min_child_weight=xgb_min_child_weight,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
    )
    model.fit(x_res, y_res)

    y_prob = model.predict_proba(x_test_scaled)[:, 1]
    importance = (
        pd.DataFrame({"Feature": x_raw.columns, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "scaler": scaler,
        "encoder": oe,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "x_columns": list(x_raw.columns),
        "y_test": y_test,
        "y_prob": y_prob,
        "importance": importance,
    }


def build_model_input(
    df: pd.DataFrame,
    user_numeric: dict[str, float],
    user_categorical: dict[str, str],
    trained: dict,
) -> pd.DataFrame:
    num_cols: list[str] = trained["num_cols"]
    cat_cols: list[str] = trained["cat_cols"]
    oe: OrdinalEncoder = trained["encoder"]

    num_defaults = {col: float(df[col].median()) for col in num_cols}
    cat_defaults = {
        col: str(df[col].mode(dropna=True).iloc[0]) if not df[col].mode(dropna=True).empty else str(df[col].astype(str).iloc[0])
        for col in cat_cols
    }

    num_defaults.update({k: float(v) for k, v in user_numeric.items()})
    cat_defaults.update({k: str(v) for k, v in user_categorical.items()})

    num_row = pd.DataFrame([num_defaults])
    cat_row = pd.DataFrame([cat_defaults])
    cat_encoded = pd.DataFrame(oe.transform(cat_row), columns=cat_cols)
    x_input = pd.concat([num_row, cat_encoded], axis=1)[trained["x_columns"]]
    return x_input


def compute_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">Term Deposit Prediction</h2>
        <p style="margin:0.25rem 0 0 0; opacity:0.9;">
            Predict whether a client will subscribe to a term deposit (Yes/No) after a marketing campaign.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

csv_path = Path("bank.csv")
if not csv_path.exists():
    st.error("Missing `bank.csv` beside `app.py`.")
    st.stop()

df = read_data(str(csv_path))
num_cols, cat_cols = split_features(df)

with st.sidebar:
    st.header("Controls")
    test_size = st.slider("Train/Test sample fraction (test size)", min_value=0.10, max_value=0.40, value=0.10, step=0.05)
    threshold = st.slider("Decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    random_state = st.number_input("Random state", min_value=1, max_value=9999, value=DEFAULT_SEED, step=1)

    st.subheader("XGBoost")
    xgb_n_estimators = st.slider("xgb n_estimators", min_value=50, max_value=600, value=250, step=25)
    xgb_max_depth = st.slider("xgb max_depth", min_value=2, max_value=12, value=5, step=1)
    xgb_learning_rate = st.slider("xgb learning_rate", min_value=0.01, max_value=0.40, value=0.08, step=0.01)
    xgb_min_child_weight = st.slider("xgb min_child_weight", min_value=1, max_value=15, value=1, step=1)
    xgb_subsample = st.slider("xgb subsample", min_value=0.50, max_value=1.00, value=0.90, step=0.05)
    xgb_colsample_bytree = st.slider("xgb colsample_bytree", min_value=0.50, max_value=1.00, value=0.90, step=0.05)

try:
    trained = train_xgboost_model(
        df=df,
        random_state=int(random_state),
        test_size=float(test_size),
        xgb_n_estimators=int(xgb_n_estimators),
        xgb_max_depth=int(xgb_max_depth),
        xgb_learning_rate=float(xgb_learning_rate),
        xgb_min_child_weight=int(xgb_min_child_weight),
        xgb_subsample=float(xgb_subsample),
        xgb_colsample_bytree=float(xgb_colsample_bytree),
    )
except Exception as exc:
    st.error(f"Training failed: {exc}")
    st.stop()

metrics = compute_metrics(trained["y_test"], trained["y_prob"], float(threshold))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
col2.metric("Recall", f"{metrics['Recall']:.3f}")
col3.metric("Precision", f"{metrics['Precision']:.3f}")
col4.metric("F1", f"{metrics['F1']:.3f}")
st.caption(f"Metrics are on test split at threshold = {threshold:.2f}")

with st.expander("Top 5 Most Important Features - XGBoost", expanded=True):
    rank_df = trained["importance"].head(5).copy()
    rank_df.index = rank_df.index + 1
    st.dataframe(rank_df, width="stretch")

st.subheader("Client Inputs")
st.caption("Input only: contact, housing, duration, previous outcome (poutcome), and previous.")

numeric_inputs: dict[str, float] = {}
categorical_inputs: dict[str, str] = {}

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        st.markdown("**Top Numeric Features**")
        for col in ["duration", "previous"]:
            if col in num_cols:
                series = df[col]
                min_val = float(series.min())
                max_val = float(series.max())
                med_val = float(series.median())
                step = 10.0 if col == "duration" else 1.0
                numeric_inputs[col] = float(
                    st.number_input(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=med_val,
                        step=step,
                    )
                )

    with right:
        st.markdown("**Top Categorical Features**")
        for col in ["contact", "housing", "poutcome"]:
            if col in cat_cols:
                options = sorted(df[col].astype(str).dropna().unique().tolist())
                mode_val = str(df[col].mode(dropna=True).iloc[0]) if not df[col].mode(dropna=True).empty else options[0]
                default_idx = options.index(mode_val) if mode_val in options else 0
                label = "previous outcome (poutcome)" if col == "poutcome" else col
                categorical_inputs[col] = st.selectbox(label, options=options, index=default_idx)

    missing_requested = [feature for feature in TOP_INPUT_FEATURES if feature not in num_cols + cat_cols]
    if missing_requested:
        st.warning(f"Some requested top features are missing from this dataset: {', '.join(missing_requested)}")

    predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    x_input = build_model_input(df=df, user_numeric=numeric_inputs, user_categorical=categorical_inputs, trained=trained)
    x_scaled = trained["scaler"].transform(x_input)
    prob_yes = float(trained["model"].predict_proba(x_scaled)[0, 1])
    pred_yes = int(prob_yes >= float(threshold))

    st.metric("Subscription Probability (Yes)", f"{prob_yes * 100:.1f}%")
    st.caption(f"Decision threshold: {threshold:.2f}")

    if pred_yes == 1:
        st.success("Prediction: YES - likely to subscribe")
    else:
        st.warning("Prediction: NO - less likely to subscribe")
