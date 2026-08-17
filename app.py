"""
CreditLens MY, loan application review system.

A credit officer opens a pending application, sees the applicant's profile and a
risk score, and records a decision. Declines produce an adverse action notice, which
is the letter the applicant is legally entitled to receive.

The scoring path here is identical to CreditLens.ipynb. The model, the calibrator and
every scorecard setting are loaded from artifacts/ rather than rewritten, so the app
and the notebook can never quietly disagree.

Run:  streamlit run app.py
"""

import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="CreditLens MY", page_icon=None, layout="wide")


# ---------------------------------------------------------------------------
# Load everything the notebook produced
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    needed = ["creditlens_lgbm.joblib", "creditlens_calibrator.joblib",
              "creditlens_scorecard.json", "sample_applicants.parquet"]
    missing = [f for f in needed if not os.path.exists(os.path.join(ART, f))]
    if missing:
        return None, missing

    model = joblib.load(os.path.join(ART, "creditlens_lgbm.joblib"))
    calibrator = joblib.load(os.path.join(ART, "creditlens_calibrator.joblib"))
    with open(os.path.join(ART, "creditlens_scorecard.json")) as f:
        meta = json.load(f)
    applicants = pd.read_parquet(os.path.join(ART, "sample_applicants.parquet"))

    outcomes_path = os.path.join(ART, "sample_actuals.parquet")
    outcomes = (pd.read_parquet(outcomes_path)["TARGET"]
                if os.path.exists(outcomes_path) else None)
    return (model, calibrator, meta, applicants, outcomes), []


loaded, missing = load_artifacts()
if loaded is None:
    st.title("CreditLens MY")
    st.error("The system cannot start because the model files are missing.")
    st.write("Run `CreditLens.ipynb` from top to bottom, then reload this page.")
    st.write("Not found in `artifacts/`:")
    for m in missing:
        st.write(f"- `{m}`")
    st.stop()

model, calibrator, meta, applicants, outcomes = loaded

CUTOFF = int(meta["cutoff"])
FLOOR, CAP = meta["score_floor"], meta["score_cap"]
REASON_TEXT = meta.get("adverse_action", {})
NOT_A_REASON = set(meta.get("non_actionable_features", []))


# ---------------------------------------------------------------------------
# Scoring, identical to the notebook
# ---------------------------------------------------------------------------
def align(frame):
    """Restore the exact dtypes and column order the model was trained with."""
    frame = frame.copy()
    for col, cats in meta.get("categorical_levels", {}).items():
        if col in frame.columns:
            frame[col] = pd.Categorical(frame[col].astype("object"), categories=cats)
    if meta.get("column_order"):
        frame = frame[meta["column_order"]]
    return frame


def probability_to_score(p_default):
    p = np.clip(p_default, 1e-6, 1 - 1e-6)
    odds = (1 - p) / p
    factor = meta["pdo"] / np.log(2)
    offset = meta["base_score"] - factor * np.log(meta["base_odds"])
    return np.clip(offset + factor * np.log(odds), FLOOR, CAP).round().astype(int)


@st.cache_data
def score_everyone():
    frame = align(applicants)
    p = calibrator.predict(model.predict_proba(frame)[:, 1])
    return probability_to_score(p), p


ALL_SCORES, ALL_PROBA = score_everyone()


# ---------------------------------------------------------------------------
# Demo identities
#
# The Home Credit dataset is anonymous, so the queue would otherwise be a list of
# numbers. Names are generated from the applicant id so they stay consistent between
# reloads. They are invented for the demo and belong to nobody.
# ---------------------------------------------------------------------------
FIRST = ["Ahmad", "Nurul", "Siti", "Muhammad", "Aina", "Faizal", "Hafiz", "Zulkifli",
         "Lim", "Tan", "Wong", "Chong", "Rajesh", "Kavitha", "Suresh", "Priya",
         "Aisyah", "Danial", "Farah", "Iskandar", "Mei Ling", "Wei Jie", "Anand"]
LAST = ["bin Abdullah", "binti Hassan", "bin Ibrahim", "binti Osman", "Wei Ming",
        "Siew Ling", "Chee Keong", "Hui Min", "a/l Muthu", "a/p Krishnan",
        "bin Rahman", "binti Yusof", "Kok Wah", "a/l Subramaniam"]
STATES = ["Selangor", "Kuala Lumpur", "Johor", "Pulau Pinang", "Perak",
          "Negeri Sembilan", "Melaka", "Kedah", "Sabah", "Sarawak"]


def identity(applicant_id):
    n = int(applicant_id)
    return {
        "name": f"{FIRST[n % len(FIRST)]} {LAST[(n // 7) % len(LAST)]}",
        "state": STATES[(n // 3) % len(STATES)],
        "ref": f"CL-{n:07d}",
    }


def money(value):
    if pd.isna(value):
        return "not provided"
    return f"RM {value:,.0f}"


def get(row, column, default=np.nan):
    return row[column] if column in row.index else default


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
@st.cache_data
def build_queue():
    rows = []
    for pos, applicant_id in enumerate(applicants.index):
        person = applicants.loc[applicant_id]
        who = identity(applicant_id)
        score = int(ALL_SCORES[pos])
        rows.append({
            "Reference": who["ref"],
            "Applicant": who["name"],
            "State": who["state"],
            "Amount requested": get(person, "AMT_CREDIT"),
            "Monthly income": get(person, "AMT_INCOME_TOTAL") / 12
            if not pd.isna(get(person, "AMT_INCOME_TOTAL")) else np.nan,
            "Score": score,
            "Recommendation": "Approve" if score >= CUTOFF else "Decline",
            "_id": applicant_id,
            "_pos": pos,
        })
    return pd.DataFrame(rows)


QUEUE = build_queue()

st.title("CreditLens MY")
st.caption("Loan application review system")

with st.sidebar:
    st.subheader("Credit policy")
    st.metric("Approval cutoff", CUTOFF)
    st.write(f"Score range {FLOOR} to {CAP}")
    st.write(f"Target approval rate {meta['target_approval_rate']:.0%}")
    st.divider()
    st.subheader("Queue")
    st.metric("Pending applications", len(QUEUE))
    st.metric("Recommended approve",
              int((QUEUE["Recommendation"] == "Approve").sum()))
    st.divider()
    st.caption("Demo system. Applicant names are invented and the underlying "
               "records come from the public Home Credit dataset. Amounts are "
               "shown as RM for illustration.")

tab_queue, tab_review, tab_portfolio = st.tabs(
    ["Application queue", "Review application", "Portfolio"])


# ---------------------------------------------------------------------------
# Tab 1, the queue
# ---------------------------------------------------------------------------
with tab_queue:
    st.subheader("Pending applications")

    left, right = st.columns([3, 1])
    with left:
        search = st.text_input("Search by name or reference", "")
    with right:
        show = st.selectbox("Show", ["All", "Recommended approve", "Recommended decline"])

    view = QUEUE
    if search:
        mask = (view["Applicant"].str.contains(search, case=False, na=False)
                | view["Reference"].str.contains(search, case=False, na=False))
        view = view[mask]
    if show == "Recommended approve":
        view = view[view["Recommendation"] == "Approve"]
    elif show == "Recommended decline":
        view = view[view["Recommendation"] == "Decline"]

    st.caption(f"{len(view)} applications")
    st.dataframe(
        view.drop(columns=["_id", "_pos"]),
        hide_index=True,
        use_container_width=True,
        height=460,
        column_config={
            "Amount requested": st.column_config.NumberColumn(format="RM %d"),
            "Monthly income": st.column_config.NumberColumn(format="RM %d"),
            "Score": st.column_config.ProgressColumn(
                min_value=FLOOR, max_value=CAP, format="%d"),
        },
    )
    st.info("Open the Review application tab to work a file.")


# ---------------------------------------------------------------------------
# Tab 2, the case file
# ---------------------------------------------------------------------------
with tab_review:
    options = QUEUE["Reference"] + "  |  " + QUEUE["Applicant"]
    choice = st.selectbox("Application", options, index=0)
    record = QUEUE.iloc[int(options[options == choice].index[0])]

    applicant_id, pos = record["_id"], int(record["_pos"])
    person = applicants.loc[applicant_id]
    who = identity(applicant_id)
    score, p_default = int(ALL_SCORES[pos]), float(ALL_PROBA[pos])
    approve = score >= CUTOFF

    st.divider()
    head_left, head_right = st.columns([2, 1])

    with head_left:
        st.subheader(who["name"])
        st.write(f"Reference {who['ref']}  |  {who['state']}")
        st.write(f"Application date {date.today().isoformat()}")

    with head_right:
        st.metric("CreditLens score", score, delta=f"{score - CUTOFF} vs cutoff")
        if approve:
            st.success("Recommendation: APPROVE")
        else:
            st.error("Recommendation: DECLINE")

    st.progress(float(np.clip((score - FLOOR) / (CAP - FLOOR), 0, 1)))
    st.caption(f"{FLOOR}   .....   cutoff {CUTOFF}   .....   {CAP}")

    st.divider()
    st.subheader("Applicant profile")

    income = get(person, "AMT_INCOME_TOTAL")
    credit = get(person, "AMT_CREDIT")
    annuity = get(person, "AMT_ANNUITY")

    a, b, c, d = st.columns(4)
    a.metric("Annual income", money(income))
    b.metric("Amount requested", money(credit))
    c.metric("Monthly instalment", money(annuity))
    d.metric("Estimated risk of default", f"{p_default:.1%}")

    e, f, g, h = st.columns(4)
    employed = get(person, "employed_years")
    age = get(person, "age_years")
    e.metric("Years employed",
             "not employed" if pd.isna(employed) else f"{employed:.1f}")
    f.metric("Age", "unknown" if pd.isna(age) else f"{age:.0f}")
    g.metric("Active credit lines",
             "0" if pd.isna(get(person, "bureau_active_loans"))
             else f"{get(person, 'bureau_active_loans'):.0f}")
    utilisation = get(person, "credit_utilization")
    h.metric("Credit utilisation",
             "unknown" if pd.isna(utilisation) else f"{utilisation:.0%}")

    with st.expander("Employment and background"):
        for label, column in [("Loan type", "NAME_CONTRACT_TYPE"),
                              ("Income type", "NAME_INCOME_TYPE"),
                              ("Education", "NAME_EDUCATION_TYPE"),
                              ("Employer sector", "ORGANIZATION_TYPE"),
                              ("Owns a car", "FLAG_OWN_CAR"),
                              ("Owns property", "FLAG_OWN_REALTY")]:
            value = get(person, column)
            if not pd.isna(value):
                st.write(f"**{label}:** {value}")

    st.divider()
    st.subheader("Assessment")

    @st.cache_resource
    def get_explainer():
        import shap
        return shap.TreeExplainer(model)

    def drivers(applicant_id):
        frame = align(applicants.loc[[applicant_id]])
        values = get_explainer().shap_values(frame)
        if isinstance(values, list):
            values = values[1]
        values = np.array(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values[0], frame

    shap_row, aligned = drivers(applicant_id)
    names = aligned.columns.tolist()
    order = np.argsort(shap_row)[::-1]

    reasons = []
    for j in order:
        if shap_row[j] <= 0 or len(reasons) == 4:
            break
        if names[j] in NOT_A_REASON:
            continue
        reasons.append((REASON_TEXT.get(names[j],
                                        names[j].replace("_", " ").capitalize()),
                        aligned.iloc[0, j]))

    if approve:
        st.write("The application meets the credit policy. Points of note for the file:")
    else:
        st.write("The application falls below the approval cutoff. Principal reasons:")

    if reasons:
        for n, (text, value) in enumerate(reasons, start=1):
            shown = f"{value:,.2f}" if isinstance(value, (int, float, np.floating)) \
                and not pd.isna(value) else value
            st.write(f"{n}. {text}  *(recorded value {shown})*")
    else:
        st.write("No adverse factors of significance were identified.")

    st.divider()
    st.subheader("Record decision")

    d1, d2 = st.columns(2)
    with d1:
        officer = st.text_input("Reviewing officer", "H. Hazim")
    with d2:
        decision = st.radio("Decision", ["Follow recommendation", "Approve", "Decline"],
                            horizontal=True)

    final = ("Approve" if approve else "Decline") if decision == "Follow recommendation" \
        else decision

    if final == "Decline":
        letter = [
            f"{date.today().strftime('%d %B %Y')}",
            "",
            f"{who['name']}",
            f"{who['state']}",
            "",
            f"Reference: {who['ref']}",
            "",
            "NOTICE OF ACTION TAKEN ON YOUR CREDIT APPLICATION",
            "",
            "Thank you for your recent application. After careful assessment we are",
            "unable to approve your request for credit at this time.",
            "",
            "The principal reasons for this decision were:",
            "",
        ]
        for n, (text, _) in enumerate(reasons, start=1):
            letter.append(f"    {n}. {text}")
        if not reasons:
            letter.append("    1. Overall credit assessment did not meet our criteria")
        letter += [
            "",
            "This decision was reached with the assistance of an automated scoring",
            "system. You have the right to request a review of this decision by a",
            "member of our staff, and to ask for a copy of the information used.",
            "",
            "You may reapply once your circumstances change.",
            "",
            "Yours sincerely,",
            "",
            f"{officer}",
            "Credit Assessment, CreditLens MY",
        ]
        text = "\n".join(letter)

        st.write("**Adverse action notice**")
        st.caption("Generated automatically from the reasons above. This is the "
                   "document the applicant receives.")
        st.code(text, language=None)
        st.download_button("Download notice", text,
                           file_name=f"{who['ref']}_notice.txt")
    else:
        st.success(f"Application {who['ref']} approved by {officer}. "
                   f"Offer of {money(credit)} to be issued.")

    if outcomes is not None and applicant_id in outcomes.index:
        with st.expander("Reveal what actually happened (demo only)"):
            defaulted = outcomes.loc[applicant_id] == 1
            st.write("This applicant **defaulted**." if defaulted
                     else "This applicant **repaid** the loan.")
            correct = (defaulted and not approve) or (not defaulted and approve)
            st.write("The recommendation was correct."
                     if correct else "The recommendation was wrong on this case.")
            st.caption("A model at this accuracy level gets individual cases wrong. "
                       "It is judged on the whole portfolio, not one file.")


# ---------------------------------------------------------------------------
# Tab 3, management view
# ---------------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Portfolio at the current cutoff")

    if outcomes is None:
        st.info("Outcome data is not available.")
    else:
        aligned_outcomes = outcomes.loc[applicants.index].values
        frame = pd.DataFrame({"score": ALL_SCORES, "defaulted": aligned_outcomes})
        approved_mask = frame["score"] >= CUTOFF

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Applications", len(frame))
        m2.metric("Approved", f"{approved_mask.mean():.1%}")
        m3.metric("Default rate if approved",
                  f"{frame.loc[approved_mask, 'defaulted'].mean():.2%}")
        m4.metric("Default rate of those declined",
                  f"{frame.loc[~approved_mask, 'defaulted'].mean():.2%}")

        st.caption("The gap between the last two numbers is the value of the system. "
                   "A wide gap means it is sorting good from bad.")

        st.divider()
        st.write("**Risk by score band**")
        bands = pd.cut(frame["score"], bins=[300, 500, 550, 600, 650, 700, 850],
                       include_lowest=True)
        table = frame.groupby(bands, observed=True).agg(
            applications=("defaulted", "size"),
            default_rate=("defaulted", "mean"),
        )
        table["default_rate"] = (table["default_rate"] * 100).round(2)
        table.index = table.index.astype(str)
        st.dataframe(table, use_container_width=True)
        st.caption("Default rate should fall steadily as the score rises. "
                   "If it does not, the score is not doing its job.")

        st.divider()
        st.write("**Where applicants sit**")
        st.bar_chart(frame["score"].value_counts(bins=30).sort_index().rename("applicants"))

    with st.expander("Model details, for technical review"):
        st.write(f"Gradient boosted trees (LightGBM). Test AUC "
                 f"{meta['test_auc']:.4f}, Gini {meta['test_gini']:.4f}. "
                 f"Logistic regression baseline scored {meta['baseline_auc']:.4f}, "
                 f"so the added complexity is buying "
                 f"{meta['test_auc'] - meta['baseline_auc']:.4f} AUC.")
        st.write(f"{meta['n_features']} features. Probabilities are "
                 f"{meta.get('calibration', 'calibrated')} before scoring, without "
                 f"which the class weighting of {meta['scale_pos_weight']:.1f} would "
                 f"push every score roughly 200 points too low.")
        removed = meta.get("removed_protected") or []
        if removed:
            st.write("Excluded from the model on legal grounds: "
                     + ", ".join(removed) + ". Credit decisions may not be based on "
                     "these characteristics.")
        st.caption("Accuracy is deliberately not reported. At an 8 percent default "
                   "rate, predicting that nobody defaults scores 92 percent while "
                   "catching no defaulters at all.")
