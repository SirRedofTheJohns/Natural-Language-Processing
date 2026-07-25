from pathlib import Path

import streamlit as st

from financial_complaint_triage import (
    FinancialComplaintTriageService,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "transformer"
    / "modernbert_base_512"
)

SYNTHETIC_EXAMPLES = {
    "Card charge": (
        "A synthetic cardholder noticed the same online purchase "
        "listed twice and could not resolve the duplicate charge."
    ),
    "Mortgage payment": (
        "A synthetic borrower made an on-time mortgage payment, but "
        "the account still shows the installment as overdue."
    ),
    "Credit report": (
        "A synthetic consumer found an unfamiliar account on a credit "
        "report and the dispute did not remove the incorrect entry."
    ),
}


@st.cache_resource
def get_service() -> FinancialComplaintTriageService:
    """Load one shared local model for the Streamlit process."""

    return FinancialComplaintTriageService(
        model_path=MODEL_PATH,
        confidence_threshold=0.90,
        max_length=512,
    )


def select_example() -> None:
    """Copy the selected synthetic example into the narrative field."""

    selected_example = st.session_state["selected_example"]
    st.session_state["complaint_narrative"] = (
        SYNTHETIC_EXAMPLES[selected_example]
    )


def main() -> None:
    """Render the focused single-complaint demo."""

    st.set_page_config(
        page_title="Financial Complaint Triage",
        page_icon="📨",
    )
    st.title("Financial Complaint Triage")
    st.caption(
        "Classify a complaint narrative and decide whether it can be "
        "routed automatically or needs human review."
    )

    first_example = next(iter(SYNTHETIC_EXAMPLES))
    if "selected_example" not in st.session_state:
        st.session_state["selected_example"] = first_example
    if "complaint_narrative" not in st.session_state:
        st.session_state["complaint_narrative"] = (
            SYNTHETIC_EXAMPLES[first_example]
        )

    st.selectbox(
        "Synthetic example",
        options=list(SYNTHETIC_EXAMPLES),
        key="selected_example",
        on_change=select_example,
    )
    narrative = st.text_area(
        "Complaint narrative",
        key="complaint_narrative",
        height=180,
        max_chars=10_000,
    )

    if st.button(
        "Classify complaint",
        type="primary",
        use_container_width=True,
    ):
        if not narrative.strip():
            st.warning("Enter a complaint narrative before classifying.")
        else:
            try:
                prediction = get_service().predict_one(narrative)
            except Exception:
                st.error(
                    "The local inference service is unavailable. "
                    "Check that the model artifacts are configured."
                )
            else:
                left, right = st.columns(2)
                left.metric(
                    "Predicted product",
                    prediction.predicted_product,
                )
                right.metric(
                    "Confidence score",
                    f"{prediction.confidence_score:.1%}",
                )
                st.write(
                    f"**Routing decision:** "
                    f"`{prediction.routing_decision}`"
                )
                st.write(
                    f"**Decision reason:** "
                    f"`{prediction.decision_reason}`"
                )
                st.write(
                    f"**Active threshold:** "
                    f"{prediction.threshold:.0%}"
                )
                st.subheader("Top candidates")
                st.table(
                    [
                        {
                            "Product": candidate.product,
                            "Confidence score": (
                                f"{candidate.confidence_score:.1%}"
                            ),
                        }
                        for candidate in prediction.top_candidates
                    ]
                )

    st.info(
        "Portfolio prototype: confidence scores are not calibrated "
        "probabilities. Narratives are truncated at 512 tokens, and "
        "low-confidence or insufficient-text cases require human review. "
        "This tool supports routing; it does not make financial decisions."
    )


if __name__ == "__main__":
    main()
