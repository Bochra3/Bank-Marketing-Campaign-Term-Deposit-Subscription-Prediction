from pathlib import Path

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder

DEFAULT_SEED = 42

st.set_page_config(page_title="Bank Marketing Predictor", page_icon=":sparkles:", layout="centered")

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
    div[data-testid="stMetric"] {
        border: 1px solid #d0d5dd;
        border-radius: 12px;
        background: white;
        padding: 0.45rem 0.2rem;
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


def get_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = df.select_dtypes(exclude="object").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if "term_deposit" in cat_cols:
        cat_cols.remove("term_deposit")
    if "pdays" in num_cols:
        num_cols.remove("pdays")

    return num_cols, cat_cols


@st.cache_data(show_spinner=False)
def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=True)
def prepare_top5(df: pd.DataFrame, random_state: int) -> dict:
    data = normalize_dataset(df)
    num_cols, cat_cols = get_feature_lists(data)

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(data["term_deposit"].astype(str))

    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_cat = pd.DataFrame(cat_encoder.fit_transform(data[cat_cols]), columns=cat_cols, index=data.index)
    x_num = data[num_cols].copy()
    x_all = pd.concat([x_num, x_cat], axis=1)

    scaler_all = MinMaxScaler()
    x_all_scaled = scaler_all.fit_transform(x_all)
    x_res, y_res = SMOTE(random_state=random_state).fit_resample(x_all_scaled, y)

    rank_model = RandomForestClassifier(random_state=random_state, n_estimators=300)
    rank_model.fit(x_res, y_res)

    importance = pd.DataFrame(
        {
            "feature": x_all.columns,
            "importance": rank_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top5_features = importance.head(5)["feature"].tolist()

    return {
        "data": data,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_encoder": cat_encoder,
        "x_all": x_all,
        "y": y,
        "importance": importance,
        "top5": top5_features,
    }


def build_model(model_name: str, random_state: int, params: dict):
    return RandomForestClassifier(
        random_state=random_state,
        n_estimators=params["rf_n_estimators"],
        max_depth=params["rf_max_depth"],
        min_samples_split=params["rf_min_samples_split"],
    )


@st.cache_resource(show_spinner=True)
def train_top5_model(
    x_all: pd.DataFrame,
    y,
    top5: list[str],
    model_name: str,
    test_size: float,
    sample_frac: float,
    random_state: int,
    params: dict,
) -> dict:
    if sample_frac < 1.0:
        x_sample, _, y_sample, _ = train_test_split(
            x_all,
            y,
            train_size=sample_frac,
            stratify=y,
            random_state=random_state,
        )
    else:
        x_sample, y_sample = x_all, y

    x_top5 = x_sample[top5].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_top5,
        y_sample,
        test_size=test_size,
        stratify=y_sample,
        random_state=random_state,
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    x_res, y_res = SMOTE(random_state=random_state).fit_resample(x_train_scaled, y_train)

    model = build_model(model_name, random_state, params)
    model.fit(x_res, y_res)

    return {
        "model": model,
        "scaler": scaler,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "pos_rate": float(pd.Series(y_sample).mean()),
    }


def encode_categorical_value(feature: str, value: str, cat_cols: list[str], cat_encoder: OrdinalEncoder) -> float:
    idx = cat_cols.index(feature)
    categories = [str(x) for x in cat_encoder.categories_[idx]]
    try:
        return float(categories.index(str(value)))
    except ValueError:
        return -1.0


def build_top5_input_df(user_values: dict, bundle: dict) -> pd.DataFrame:
    row = {}
    for feature in bundle["top5"]:
        if feature in bundle["num_cols"]:
            row[feature] = float(user_values[feature])
        else:
            row[feature] = encode_categorical_value(
                feature,
                str(user_values[feature]),
                bundle["cat_cols"],
                bundle["cat_encoder"],
            )
    return pd.DataFrame([row])[bundle["top5"]]


st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">Bank Marketing Predictor</h2>
        <p style="margin:0.25rem 0 0 0; opacity:0.9;">Beautiful, simple UI with only your 5 most important features.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

csv_path = Path("bank.csv")
if not csv_path.exists():
    st.error("Missing `bank.csv` beside `app.py`.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    sample_frac = st.slider("Train/Test sample fraction", min_value=0.10, max_value=1.00, value=1.00, step=0.05)
    test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    threshold = st.slider("Decision threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    random_state = st.number_input("Random state", min_value=1, max_value=9999, value=DEFAULT_SEED, step=1)

    st.markdown("---")
    st.caption("Random Forest (Best for 80/20 split)")
    rf_n_estimators = st.slider("n_estimators", 100, 600, 300, 20)
    rf_max_depth = st.slider("max_depth", 2, 20, 10, 1)
    rf_min_samples_split = st.slider("min_samples_split", 2, 20, 2, 1)

params = {
    "rf_n_estimators": rf_n_estimators,
    "rf_max_depth": rf_max_depth,
    "rf_min_samples_split": rf_min_samples_split,
}

try:
    df = read_data(str(csv_path))
    prep = prepare_top5(df, int(random_state))
    trained = train_top5_model(
        prep["x_all"],
        prep["y"],
        prep["top5"],
        "RandomForest",
        float(test_size),
        float(sample_frac),
        int(random_state),
        params,
    )
except Exception as exc:
    st.error(f"Setup failed: {exc}")
    st.stop()

st.caption("Model uses only the 5 most important features discovered from your dataset.")

with st.expander("Top 5 Features", expanded=True):
    rank_df = prep["importance"].head(5).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    st.dataframe(rank_df.rename(columns={"feature": "Feature", "importance": "Importance"}), use_container_width=True)

st.subheader("Client Inputs")

user_values: dict[str, float | str] = {}
cols = st.columns(2)

for i, feature in enumerate(prep["top5"]):
    with cols[i % 2]:
        if feature in prep["num_cols"]:
            series = prep["data"][feature]
            min_val = float(series.min())
            max_val = float(series.max())
            med_val = float(series.median())

            if feature == "duration":
                user_values[feature] = st.slider(
                    feature,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(med_val),
                    step=10,
                )
            else:
                user_values[feature] = st.number_input(
                    feature,
                    value=med_val,
                    min_value=min_val,
                    max_value=max_val,
                )
        else:
            options = sorted(prep["data"][feature].astype(str).dropna().unique().tolist())
            user_values[feature] = st.selectbox(feature, options=options, index=0)

if st.button("Predict", type="primary", use_container_width=True):
    x_input = build_top5_input_df(user_values, prep)
    x_scaled = trained["scaler"].transform(x_input)

    prob_yes = float(trained["model"].predict_proba(x_scaled)[0, 1])
    pred_yes = int(prob_yes >= threshold)

    st.metric("Subscription Probability (Yes)", f"{prob_yes * 100:.1f}%")
    st.caption(f"Threshold: {threshold:.2f}")

    if pred_yes == 1:
        st.success("Prediction: YES - likely to subscribe")
    else:
        st.warning("Prediction: NO - less likely to subscribe")
