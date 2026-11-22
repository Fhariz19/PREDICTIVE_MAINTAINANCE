from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Design tokens: one neutral base + one alert colour ----------
NEUTRAL = "#4C78A8"   # healthy / general data
ALERT = "#E4572E"     # failure / risk ONLY
GREY = "#9AA5B1"
COLORS = {"No failure": NEUTRAL, "Failure": ALERT}

SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_MODES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# Final test results copied from README.md
MODEL_METRICS = {
    "Accuracy": 98.80,
    "Precision": 80.00,
    "Recall": 86.27,
    "F1-score": 83.02,
    "ROC-AUC": 98.65,
    "PR-AUC": 90.30,
}


@st.cache_data
def load_data() -> pd.DataFrame:
    folder = Path(__file__).parent / "dataset"
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        st.error("No CSV file found in the dataset/ folder.")
        st.stop()
    df = pd.read_csv(csv_files[0])
    df["Status"] = df["Machine failure"].map({0: "No failure", 1: "Failure"})
    df["Wear band"] = pd.cut(
        df["Tool wear [min]"],
        bins=[-1, 50, 100, 150, 200, 300],
        labels=["0-50", "51-100", "101-150", "151-200", "201+"],
    )
    return df


def _style(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.2),
        font=dict(size=13),
    )
    return fig


def render_dashboard(threshold: float) -> None:
    df_all = load_data()

    # ---------- Sidebar filters ----------
    st.sidebar.subheader("Dashboard filters")
    types = st.sidebar.multiselect(
        "Machine type (L = Low, M = Medium, H = High quality)",
        options=["L", "M", "H"],
        default=["L", "M", "H"],
    )
    df = df_all[df_all["Type"].isin(types)]
    if df.empty:
        st.warning("Select at least one machine type.")
        return

    total = len(df)
    failures = int(df["Machine failure"].sum())
    rate = failures / total

    # ---------- Header ----------
    st.header("Dashboard Overview")
    st.caption(
        "Which machines fail, under what conditions, and how well the model "
        "catches them. Source: AI4I 2020 Predictive Maintenance Dataset."
    )

    # ---------- KPI row ----------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Machines (records)", f"{total:,}")
    k2.metric("Failures", f"{failures:,}")
    k3.metric("Failure rate", f"{rate:.2%}")
    k4.metric("Model recall", f"{MODEL_METRICS['Recall']:.1f}%")
    k5.metric("Decision threshold", f"{threshold:.4f}")

    st.divider()

    # ---------- Row 1: what fails and how often ----------
    c1, c2, c3 = st.columns([1, 1.3, 1.3])

    with c1:
        st.subheader("Failure share")
        counts = df["Status"].value_counts().reindex(["No failure", "Failure"]).fillna(0)
        fig = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.6,
                marker_colors=[COLORS[s] for s in counts.index],
                sort=False,
            )
        )
        st.plotly_chart(_style(fig), use_container_width=True)

    with c2:
        st.subheader("Most common failure modes")
        modes = (
            df[list(FAILURE_MODES)].sum().rename(index=FAILURE_MODES).sort_values()
        )
        fig = px.bar(
            x=modes.values,
            y=modes.index,
            orientation="h",
            labels={"x": "Number of failures", "y": ""},
            color_discrete_sequence=[ALERT],
        )
        st.plotly_chart(_style(fig), use_container_width=True)

    with c3:
        st.subheader("Failure rate by tool wear")
        wear = (
            df.groupby("Wear band", observed=True)["Machine failure"]
            .mean()
            .mul(100)
            .reset_index()
        )
        fig = px.bar(
            wear,
            x="Wear band",
            y="Machine failure",
            labels={"Machine failure": "Failure rate (%)", "Wear band": "Tool wear [min]"},
            color_discrete_sequence=[NEUTRAL],
        )
        # highlight the riskiest band with the alert colour
        peak = wear["Machine failure"].idxmax()
        fig.update_traces(
            marker_color=[ALERT if i == peak else NEUTRAL for i in wear.index]
        )
        st.plotly_chart(_style(fig), use_container_width=True)

    # ---------- Row 2: operating conditions ----------
    c4, c5 = st.columns(2)

    with c4:
        st.subheader("Sensor readings: healthy vs failed")
        sensor = st.selectbox("Choose a sensor", SENSORS, index=3)
        fig = px.box(
            df,
            x="Status",
            y=sensor,
            color="Status",
            color_discrete_map=COLORS,
            category_orders={"Status": ["No failure", "Failure"]},
            labels={"Status": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_style(fig), use_container_width=True)

    with c5:
        st.subheader("Torque vs rotational speed")
        # plot healthy first so failures stay visible on top
        plot_df = pd.concat(
            [df[df["Machine failure"] == 0].sample(min(2500, int((df["Machine failure"] == 0).sum())), random_state=1),
             df[df["Machine failure"] == 1]]
        )
        fig = px.scatter(
            plot_df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Status",
            color_discrete_map=COLORS,
            opacity=0.6,
            category_orders={"Status": ["No failure", "Failure"]},
        )
        st.plotly_chart(_style(fig), use_container_width=True)

    # ---------- Row 3: type, correlation, model ----------
    c6, c7, c8 = st.columns([1, 1.3, 1.2])

    with c6:
        st.subheader("Failure rate by machine type")
        by_type = (
            df.groupby("Type")["Machine failure"].mean().mul(100).reset_index()
        )
        fig = px.bar(
            by_type,
            x="Type",
            y="Machine failure",
            labels={"Machine failure": "Failure rate (%)", "Type": "Machine type"},
            color_discrete_sequence=[NEUTRAL],
        )
        st.plotly_chart(_style(fig), use_container_width=True)

    with c7:
        st.subheader("Correlation heatmap")
        corr = df[SENSORS + ["Machine failure"]].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(_style(fig, 340), use_container_width=True)

    with c8:
        st.subheader("Model test performance")
        m = pd.Series(MODEL_METRICS).sort_values()
        fig = px.bar(
            x=m.values,
            y=m.index,
            orientation="h",
            labels={"x": "Score (%)", "y": ""},
            color_discrete_sequence=[NEUTRAL],
            range_x=[0, 100],
        )
        fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
        st.plotly_chart(_style(fig), use_container_width=True)

    # ---------- Key insights (text that states the finding) ----------
    st.divider()
    st.subheader("Key insights")
    worst_wear = wear.loc[wear["Machine failure"].idxmax()]
    worst_type = by_type.loc[by_type["Machine failure"].idxmax()]
    st.markdown(
        f"- Only **{rate:.2%}** of records are failures, so the data is highly imbalanced; "
        "accuracy alone is misleading, which is why recall and F1 are reported.\n"
        f"- Failure rate is highest at tool wear **{worst_wear['Wear band']} min** "
        f"(**{worst_wear['Machine failure']:.1f}%**).\n"
        f"- Machine type **{worst_type['Type']}** has the highest failure rate "
        f"(**{worst_type['Machine failure']:.2f}%**).\n"
        f"- The model catches **{MODEL_METRICS['Recall']:.1f}%** of failures "
        f"with a decision threshold of **{threshold:.4f}**."
    )
