from pathlib import Path
import warnings

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Airbnb Room Type Predictor | Sagar Kapoor",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.pkl"

# These are the exact input columns found in your saved pipeline.
FEATURE_COLUMNS = [
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

# ---------------------------------------------------------
# BLACK & WHITE PROFESSIONAL THEME
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #090909;
        --panel: #111111;
        --panel-2: #171717;
        --line: #303030;
        --muted: #a1a1a1;
        --white: #f5f5f5;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: var(--bg);
        color: var(--white);
    }
    [data-testid="stHeader"] {
        background: rgba(9,9,9,0.92);
    }
    [data-testid="stSidebar"] {
        background: #101010;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: #eeeeee;
    }
    .block-container {
        padding-top: 2.1rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    h1, h2, h3, .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.7px;
    }
    .hero {
        padding: 30px 32px;
        border: 1px solid #333333;
        border-radius: 18px;
        background: linear-gradient(135deg, #171717 0%, #0c0c0c 70%);
        margin-bottom: 25px;
    }
    .eyebrow {
        color: #bdbdbd;
        text-transform: uppercase;
        letter-spacing: 2.2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: clamp(30px, 4vw, 48px);
        line-height: 1.08;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 12px 0;
    }
    .hero-subtitle {
        color: #b5b5b5;
        font-size: 15px;
        line-height: 1.7;
        max-width: 760px;
    }
    .section-label {
        color: #f1f1f1;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 600;
        margin: 12px 0 5px 0;
    }
    .section-help {
        color: #999999;
        font-size: 13px;
        margin-bottom: 18px;
    }
    .info-card {
        background: #111111;
        border: 1px solid #303030;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 112px;
    }
    .info-label {
        color: #a5a5a5;
        font-size: 11px;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }
    .info-value {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 23px;
        font-weight: 700;
    }
    .result-card {
        background: #f3f3f3;
        color: #101010;
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0 18px 0;
    }
    .result-card .small {
        color: #555555;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 11px;
        font-weight: 700;
    }
    .result-card .prediction {
        color: #080808;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 700;
        margin-top: 7px;
    }
    .result-card .confidence {
        color: #333333;
        font-size: 14px;
        margin-top: 8px;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #f4f4f4 !important;
        color: #080808 !important;
        border: 1px solid #f4f4f4 !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        min-height: 46px;
        transition: all .2s ease;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #cfcfcf !important;
        border-color: #cfcfcf !important;
        color: #000000 !important;
    }
    [data-testid="stMetric"] {
        background: #111111;
        border: 1px solid #303030;
        padding: 15px 17px;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: #aaaaaa; }
    [data-testid="stMetricValue"] { color: #ffffff; }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #151515;
        border-color: #3a3a3a;
    }
    input, textarea {
        color: #ffffff !important;
    }
    hr {
        border-color: #303030;
    }
    .footer {
        border-top: 1px solid #303030;
        margin-top: 42px;
        padding-top: 18px;
        color: #8e8e8e;
        font-size: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD THE REAL TRAINED PIPELINE
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading your trained Airbnb model...")
def load_model(model_file: str):
    # The model was serialized with scikit-learn 1.9.0.
    # Install the matching version from requirements.txt.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(model_file)
    return pipeline


def get_model_details(pipeline):
    """Read valid dropdown values directly from the fitted OneHotEncoder."""
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encode"]
    categories = encoder.categories_

    category_map = {
        "neighbourhood_group": sorted(str(value) for value in categories[0]),
        "neighbourhood": sorted(str(value) for value in categories[1]),
    }
    return category_map


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⌂ Airbnb Analytics")
    st.caption("MACHINE LEARNING PROJECT")
    st.divider()
    st.markdown("**Project navigation**")
    page = st.radio(
        "Go to",
        ["Predict Room Type", "About the Project", "Model Information"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Developer**")
    st.markdown("### Sagar Kapoor")
    st.caption("Data Analytics & Machine Learning")
    st.divider()
    st.caption("Model: Random Forest Classifier")
    st.caption("Dataset: Airbnb NYC 2019")


# ---------------------------------------------------------
# LOAD MODEL WITH A CLEAR ERROR MESSAGE
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Keep **app.py** and **model_pipeline.pkl** "
        "inside the same project folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    category_values = get_model_details(model)
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Use the package versions in requirements.txt. Your uploaded model "
        "was saved with scikit-learn 1.9.0."
    )
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine Learning · Classification</div>
        <div class="hero-title">Airbnb Room Type<br>Prediction</div>
        <div class="hero-subtitle">
            Explore an ML-powered prediction experience for Airbnb NYC 2019.
            Enter listing and location details to estimate the room type using
            the trained Random Forest pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Predict Room Type":
    st.markdown('<div class="section-label">Listing details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-help">Enter the property information below. '
        'The application uses the same feature names and preprocessing pipeline '
        'as your saved model.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("#### Location")
        loc1, loc2 = st.columns(2)
        with loc1:
            neighbourhood_group = st.selectbox(
                "Neighbourhood group",
                options=category_values["neighbourhood_group"],
                help="Choose one of the five NYC borough groups used during training.",
            )
            latitude = st.number_input(
                "Latitude",
                min_value=40.40,
                max_value=41.00,
                value=40.7128,
                step=0.0001,
                format="%.4f",
                help="NYC latitude coordinate.",
            )
        with loc2:
            neighbourhood = st.selectbox(
                "Neighbourhood",
                options=category_values["neighbourhood"],
                help="Neighbourhood values are loaded from the trained model's encoder.",
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-74.30,
                max_value=-73.60,
                value=-74.0060,
                step=0.0001,
                format="%.4f",
                help="NYC longitude coordinate.",
            )

        st.markdown("#### Property information")
        p1, p2, p3 = st.columns(3)
        with p1:
            price = st.number_input(
                "Price per night ($)",
                min_value=0.0,
                max_value=100000.0,
                value=150.0,
                step=10.0,
            )
            minimum_nights = st.number_input(
                "Minimum nights",
                min_value=1,
                max_value=3650,
                value=2,
                step=1,
            )
        with p2:
            number_of_reviews = st.number_input(
                "Number of reviews",
                min_value=0,
                max_value=100000,
                value=10,
                step=1,
            )
            reviews_per_month = st.number_input(
                "Reviews per month",
                min_value=0.0,
                max_value=1000.0,
                value=1.0,
                step=0.1,
            )
        with p3:
            calculated_host_listings_count = st.number_input(
                "Host listings count",
                min_value=1,
                max_value=10000,
                value=1,
                step=1,
            )
            availability_365 = st.number_input(
                "Availability (days/year)",
                min_value=0,
                max_value=365,
                value=180,
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict Room Type  →",
            use_container_width=True,
        )

    if submitted:
        input_data = {
            "neighbourhood_group": neighbourhood_group,
            "neighbourhood": neighbourhood,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "price": float(price),
            "minimum_nights": int(minimum_nights),
            "number_of_reviews": int(number_of_reviews),
            "reviews_per_month": float(reviews_per_month),
            "calculated_host_listings_count": int(calculated_host_listings_count),
            "availability_365": int(availability_365),
        }

        # Preserve the exact feature order expected by the fitted pipeline.
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = [str(label) for label in model.classes_]
            probability_df = pd.DataFrame(
                {
                    "Room type": classes,
                    "Probability": probabilities,
                }
            ).sort_values("Probability", ascending=False)

            confidence = float(max(probabilities)) * 100

            st.divider()
            st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="small">Predicted room type</div>
                    <div class="prediction">{str(prediction)}</div>
                    <div class="confidence">Model confidence: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", str(prediction))
            m2.metric("Confidence", f"{confidence:.2f}%")
            m3.metric("Classes evaluated", len(classes))

            st.markdown("#### Class probability")
            fig = px.bar(
                probability_df,
                x="Probability",
                y="Room type",
                orientation="h",
                text=probability_df["Probability"].map(lambda x: f"{x:.1%}"),
                template="plotly_dark",
            )
            fig.update_traces(
                marker_color="#e8e8e8",
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_layout(
                paper_bgcolor="#090909",
                plot_bgcolor="#090909",
                font_color="#f5f5f5",
                xaxis=dict(
                    title="Predicted probability",
                    tickformat=".0%",
                    range=[0, 1.15],
                    gridcolor="#292929",
                ),
                yaxis=dict(title="", autorange="reversed", gridcolor="#292929"),
                margin=dict(l=10, r=35, t=20, b=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View submitted input values"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)

            st.caption(
                "Note: The displayed confidence is the model's predicted class "
                "probability, not a guarantee that the prediction is correct."
            )

        except Exception as exc:
            st.error("Prediction could not be completed. Please check the inputs.")
            st.code(f"{type(exc).__name__}: {exc}")

elif page == "About the Project":
    st.markdown('<div class="section-label">About this project</div>', unsafe_allow_html=True)
    st.write(
        "This project uses the Airbnb NYC 2019 dataset to classify a listing "
        "into one of three room types: Entire home/apt, Private room, or Shared room. "
        "The notebook explores the dataset, performs data cleaning and feature "
        "engineering, compares classification algorithms, and tunes a Random Forest model."
    )

    a, b = st.columns(2)
    with a:
        st.markdown("#### Workflow")
        st.markdown(
            """
            1. Load and inspect Airbnb NYC 2019 data.
            2. Explore missing values and feature distributions.
            3. Remove selected identifier/text columns.
            4. Fill missing review frequency values with zero.
            5. Cap extreme price and minimum-night values.
            6. Preprocess numerical and categorical features.
            7. Train and evaluate classification models.
            8. Tune and save the Random Forest pipeline.
            """
        )
    with b:
        st.markdown("#### Technologies used")
        st.markdown(
            """
            - Python
            - Pandas
            - Matplotlib and Seaborn
            - Scikit-learn
            - Random Forest Classifier
            - Joblib
            - Streamlit
            - Plotly
            """
        )

    st.info(
        "The website runs inference using your uploaded model_pipeline.pkl. "
        "It does not retrain the model or claim a new test accuracy."
    )

elif page == "Model Information":
    st.markdown('<div class="section-label">Model information</div>', unsafe_allow_html=True)
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", type(classifier).__name__)
    c2.metric("Input features", len(FEATURE_COLUMNS))
    c3.metric("Output classes", len(model.classes_))

    st.markdown("#### Target classes")
    st.write(", ".join(str(value) for value in model.classes_))

    st.markdown("#### Model input features")
    st.dataframe(
        pd.DataFrame({"Feature": FEATURE_COLUMNS}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Preprocessing pipeline")
    st.markdown(
        """
        **Numerical features**
        - Median imputation for missing values
        - StandardScaler

        **Categorical features**
        - Most-frequent imputation
        - OneHotEncoder with unknown-category handling
        """
    )
    st.caption(
        "The details above are read from the fitted pipeline file. "
        "The saved pipeline contains the preprocessing and classifier steps."
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Designed & Developed by <strong style="color:#eeeeee;">Sagar Kapoor</strong>
        &nbsp;·&nbsp; Airbnb NYC 2019 Room Type Classification
    </div>
    """,
    unsafe_allow_html=True,
)
