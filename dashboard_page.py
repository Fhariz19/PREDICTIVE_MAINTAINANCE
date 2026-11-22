
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# DESIGN SETTINGS
# ============================================================

NEUTRAL = "#4C78A8"
ALERT = "#E4572E"
GREY = "#9AA5B1"

COLORS = {
    "No failure": NEUTRAL,
    "Failure": ALERT,
}

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

# Final test-set results: unchanged
MODEL_METRICS = {
    "Accuracy": 98.80,
    "Precision": 80.00,
    "Recall": 86.27,
    "F1-score": 83.02,
    "ROC-AUC": 98.65,
    "PR-AUC": 90.30,
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:

    folder = Path(__file__).parent / "dataset"
    csv_path = folder / "ai4i2020.csv"

    if not csv_path.exists():
        st.error("AI4I dataset not found in the dataset folder.")
        st.stop()

    df = pd.read_csv(csv_path)

    required_columns = (
        SENSORS
        + ["Type", "Machine failure"]
        + list(FAILURE_MODES.keys())
    )

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Dataset is missing columns: {missing}")
        st.stop()

    df["Status"] = df["Machine failure"].map({
        0: "No failure",
        1: "Failure"
    })

    df["Wear band"] = pd.cut(
        df["Tool wear [min]"],
        bins=[-1, 50, 100, 150, 200, 300],
        labels=["0-50", "51-100", "101-150", "151-200", "201+"]
    )

    return df


# ============================================================
# CHART STYLING
# ============================================================

def style_chart(fig, height=340):

    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=-0.2
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")

    return fig


# ============================================================
# MAINTENANCE INSIGHTS
# ============================================================


def generate_recommendations(worst_wear, worst_type, modes):

    recommendations = []

    if worst_wear is not None:
        recommendations.append({
            "title": "Monitor tool wear",
            "description": (
                f"The {worst_wear['Wear band']} minute tool-wear group "
                f"has the highest observed failure rate "
                f"({worst_wear['Failure rate (%)']:.2f}%). "
                "Consider regular tool-condition inspections."
            )
        })

    if worst_type is not None:
        recommendations.append({
            "title": "Prioritise machine inspection",
            "description": (
                f"Machine type {worst_type['Type']} has the highest "
                f"observed failure rate "
                f"({worst_type['Failure rate (%)']:.2f}%). "
                "Review its operating conditions and maintenance records."
            )
        })

    if not modes.empty:
        top_mode = modes.index[-1]

        recommendations.append({
            "title": "Review common failure patterns",
            "description": (
                f"{top_mode} is the most frequently recorded failure mode. "
                "Review the relevant machine components during inspection."
            )
        })

    recommendations.append({
        "title": "Maintain continuous monitoring",
        "description": (
            "Monitor torque, rotational speed, temperature "
            "and tool wear to identify changing machine conditions."
        )
    })

    return recommendations


# ============================================================
# MAIN DASHBOARD
# ============================================================

def render_dashboard(threshold: float):

    df_all = load_data()

    # --------------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------------

    st.sidebar.subheader("Dashboard filters")

    types = st.sidebar.multiselect(
        "Machine type",
        options=["L", "M", "H"],
        default=["L", "M", "H"],
        help="L = Low, M = Medium, H = High product quality."
    )

    df = df_all[df_all["Type"].isin(types)].copy()

    if df.empty:
        st.warning("Please select at least one machine type.")
        return

    total = len(df)
    failures = int(df["Machine failure"].sum())
    healthy = total - failures
    rate = failures / total

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.header("Predictive Maintenance Dashboard")

    st.caption(
        "Monitor machine failure patterns, operating conditions "
        "and machine learning performance."
    )

    st.caption("Dataset: AI4I 2020 Predictive Maintenance Dataset")

    st.divider()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    st.subheader("Performance Overview")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Total Machines",
        f"{total:,}"
    )

    k2.metric(
        "Machine Failures",
        f"{failures:,}"
    )

    k3.metric(
        "Healthy Machines",
        f"{healthy:,}"
    )

    k4.metric(
        "Failure Rate",
        f"{rate:.2%}"
    )

    k5.metric(
        "Model Recall",
        f"{MODEL_METRICS['Recall']:.2f}%"
    )

    st.caption(
        f"Model decision threshold: {threshold:.4f} "
        f"({threshold:.2%})"
    )

    st.divider()

    # --------------------------------------------------------
    # FAILURE ANALYSIS
    # --------------------------------------------------------

    st.subheader("Machine Failure Analysis")

    c1, c2, c3 = st.columns([1, 1.3, 1.3])

    with c1:

        st.markdown("**Failure Distribution**")

        counts = (
            df["Status"]
            .value_counts()
            .reindex(["No failure", "Failure"])
            .fillna(0)
        )

        fig = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.62,
                marker_colors=[
                    COLORS[s] for s in counts.index
                ],
                textinfo="percent+label",
                sort=False
            )
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with c2:

        st.markdown("**Most Common Failure Modes**")

        modes = (
            df[list(FAILURE_MODES)]
            .sum()
            .rename(index=FAILURE_MODES)
            .sort_values()
        )

        fig = px.bar(
            x=modes.values,
            y=modes.index,
            orientation="h",
            labels={
                "x": "Recorded failure occurrences",
                "y": ""
            },
            color_discrete_sequence=[ALERT]
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

        st.caption(
            "Failure modes may overlap; occurrences do not necessarily "
            "represent unique failed machines."
        )

    with c3:

        st.markdown("**Failure Rate by Tool Wear**")

        wear = (
            df.groupby("Wear band", observed=True)["Machine failure"]
            .agg(["mean", "count"])
            .reset_index()
        )

        wear["Failure rate (%)"] = wear["mean"] * 100

        fig = px.bar(
            wear,
            x="Wear band",
            y="Failure rate (%)",
            text="Failure rate (%)",
            labels={
                "Wear band": "Tool wear (minutes)"
            },
            color_discrete_sequence=[NEUTRAL]
        )

        if not wear.empty:
            peak = wear["Failure rate (%)"].idxmax()

            fig.update_traces(
                marker_color=[
                    ALERT if i == peak else NEUTRAL
                    for i in range(len(wear))
                ],
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # OPERATING CONDITION ANALYSIS
    # --------------------------------------------------------

    st.subheader("Machine Operating Conditions")

    c4, c5 = st.columns(2)

    with c4:

        st.markdown("**Sensor Readings: Healthy vs Failed**")

        sensor = st.selectbox(
            "Select sensor",
            SENSORS,
            index=3
        )

        fig = px.box(
            df,
            x="Status",
            y=sensor,
            color="Status",
            color_discrete_map=COLORS,
            category_orders={
                "Status": ["No failure", "Failure"]
            }
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with c5:

        st.markdown("**Torque vs Rotational Speed**")

        healthy_df = df[df["Machine failure"] == 0]

        healthy_sample = healthy_df.sample(
            n=min(2500, len(healthy_df)),
            random_state=1
        )

        failed_df = df[df["Machine failure"] == 1]

        plot_df = pd.concat([
            healthy_sample,
            failed_df
        ])

        fig = px.scatter(
            plot_df,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Status",
            color_discrete_map=COLORS,
            opacity=0.6,
            category_orders={
                "Status": ["No failure", "Failure"]
            }
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # MACHINE TYPE, CORRELATION AND MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Advanced Analytics")

    c6, c7, c8 = st.columns([1, 1.3, 1.2])

    with c6:

        st.markdown("**Failure Rate by Machine Type**")

        by_type = (
            df.groupby("Type")["Machine failure"]
            .agg(["mean", "count"])
            .reset_index()
        )

        by_type["Failure rate (%)"] = by_type["mean"] * 100

        fig = px.bar(
            by_type,
            x="Type",
            y="Failure rate (%)",
            text="Failure rate (%)",
            labels={
                "Type": "Machine type"
            },
            color_discrete_sequence=[NEUTRAL]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with c7:

        st.markdown("**Sensor Correlation Heatmap**")

        corr = df[
            SENSORS + ["Machine failure"]
        ].corr().round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )

        fig.update_xaxes(tickangle=-35)

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

    with c8:

        st.markdown("**Model Test Performance**")

        metrics = pd.Series(MODEL_METRICS).sort_values()

        fig = px.bar(
            x=metrics.values,
            y=metrics.index,
            orientation="h",
            labels={
                "x": "Score (%)",
                "y": ""
            },
            color_discrete_sequence=[NEUTRAL],
            range_x=[0, 105]
        )

        fig.update_traces(
            texttemplate="%{x:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            style_chart(fig),
            use_container_width=True
        )

        st.caption(
            "Performance measured on the held-out test dataset."
        )

    st.divider()

    # --------------------------------------------------------
    # KEY INSIGHTS
    # --------------------------------------------------------

    st.subheader("Key Insights")

    worst_wear = (
        wear.loc[wear["Failure rate (%)"].idxmax()]
        if not wear.empty else None
    )

    worst_type = (
        by_type.loc[by_type["Failure rate (%)"].idxmax()]
        if not by_type.empty else None
    )

    i1, i2, i3 = st.columns(3)

    with i1:

        st.metric(
            "Failure Rate",
            f"{rate:.2%}"
        )

        st.caption(
            "Proportion of records with machine failure "
            "in the selected dataset."
        )

    with i2:

        if worst_wear is not None:

            st.metric(
                "Highest Wear-Band Failure Rate",
                f"{worst_wear['Failure rate (%)']:.2f}%"
            )

            st.caption(
                f"Tool wear group: {worst_wear['Wear band']} minutes"
            )

    with i3:

        if worst_type is not None:

            st.metric(
                "Highest-Risk Machine Type",
                str(worst_type["Type"])
            )

            st.caption(
                f"Observed failure rate: "
                f"{worst_type['Failure rate (%)']:.2f}%"
            )

    st.info(
        f"The model achieved {MODEL_METRICS['Recall']:.2f}% recall "
        f"on the held-out test dataset. This means it identified "
        f"approximately {MODEL_METRICS['Recall']:.2f}% of actual "
        f"failures in that evaluation."
    )

    st.divider()

    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("Maintenance Recommendations")

    st.caption(
        "Suggested actions based on observed dataset patterns. "
        "These are recommendations for inspection, not confirmed diagnoses."
    )

    recommendations = generate_recommendations(
        worst_wear,
        worst_type,
        modes
    )

    for i, recommendation in enumerate(recommendations, start=1):

        with st.expander(
            f"{i}. {recommendation['title']}",
            expanded=(i == 1)
        ):

            st.write(recommendation["description"])

    st.divider()

    # --------------------------------------------------------
    # DATA EXPORT
    # --------------------------------------------------------

    st.subheader("Download Dataset")

    st.write(
        "Download the dataset after applying your selected machine type filters."
    )

    csv_data = df.drop(
        columns=["Status", "Wear band"],
        errors="ignore"
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="predictive_maintenance_filtered_data.csv",
        mime="text/csv"
    )

    st.caption(
        f"Showing {total:,} records, including {failures:,} failure records."
    )
