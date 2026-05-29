"""
glossary.py — Shared metric and concept definitions for all Streamlit pages.

Renders a collapsible expander containing plain-English definitions of every
metric, technique, and abbreviation used in the dissertation prototype.
Import and call render_glossary() at the top of any page where it is needed.
"""

from __future__ import annotations
import streamlit as st


# ── Definition dictionaries ───────────────────────────────────────────────────

IDS_METRICS: dict[str, str] = {
    "Accuracy": (
        "Percentage of all records (benign + attacks) classified correctly. "
        "**Not reliable as a headline metric for IDS**: because the dataset is heavily "
        "imbalanced (far more benign traffic than attacks), a model that flags "
        "everything as benign can still achieve ~95% accuracy while missing every attack."
    ),
    "Precision (macro)": (
        "Of all records the model predicted as a given class (e.g., attack), "
        "what fraction actually belonged to that class? "
        "High precision means **few false alarms**. "
        "Macro averaging treats all classes equally regardless of how many samples each has."
    ),
    "Recall (macro) / Detection Rate": (
        "Of all records that actually belong to a given class (e.g., attacks), "
        "what fraction did the model correctly detect? "
        "High recall means **few missed attacks**. "
        "Also called True Positive Rate (TPR) in binary classification. "
        "Macro averaging treats all classes equally."
    ),
    "F1 Score (macro)": (
        "The harmonic mean of Precision and Recall, averaged equally across all classes. "
        "Preferred over accuracy for imbalanced IDS data because it penalises "
        "models that sacrifice recall to gain precision (or vice versa). "
        "Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall). "
        "Range: 0 (worst) to 1 (best)."
    ),
    "ROC-AUC": (
        "Area Under the Receiver Operating Characteristic Curve. "
        "Measures the model's ability to discriminate between classes across **all possible "
        "decision thresholds** — not just the default 0.5 cut-off. "
        "0.5 = random guessing. 1.0 = perfect separation. "
        "For multiclass problems, One-vs-Rest (OvR) macro averaging is used."
    ),
    "MCC (Matthews Correlation Coefficient)": (
        "A single balanced metric for binary classification that accounts for all four "
        "confusion matrix cells (TN, FP, FN, TP). "
        "Unlike F1, MCC considers true negatives — important when benign traffic far "
        "outnumbers attacks. "
        "Range: −1 (perfectly wrong) to +1 (perfect). 0 = random. "
        "Mathematically equivalent to the Phi correlation coefficient."
    ),
    "False Positive Rate (FPR)": (
        "FP / (FP + TN). "
        "Fraction of **benign traffic incorrectly flagged as attacks**. "
        "In an operational IDS, a high FPR causes alert fatigue — security analysts "
        "are overwhelmed by false alarms and start ignoring alerts. "
        "Lower is better. Ideal = 0."
    ),
    "Detection Rate (DR) / True Positive Rate": (
        "TP / (TP + FN). "
        "Fraction of **actual attacks that were correctly detected**. "
        "Also called Sensitivity or Recall in binary classification. "
        "A low DR means attacks slip through undetected — the primary failure mode for IDS. "
        "Higher is better. Ideal = 1."
    ),
    "False Negative Rate (FNR)": (
        "FN / (FN + TP) = 1 − Detection Rate. "
        "Fraction of actual attacks the model **missed** (classified as benign). "
        "The inverse of Detection Rate. Lower is better."
    ),
    "Confusion Matrix": (
        "A table summarising all four prediction outcomes for binary classification:\n\n"
        "| | Predicted Benign | Predicted Attack |\n"
        "|---|---|---|\n"
        "| **Actual Benign** | True Negative (TN) ✅ | False Positive (FP) ❌ |\n"
        "| **Actual Attack** | False Negative (FN) ❌ | True Positive (TP) ✅ |\n\n"
        "FP = false alarm (benign flagged as attack). FN = missed attack (attack classified as benign)."
    ),
}

DP_CONCEPTS: dict[str, str] = {
    "Differential Privacy (DP)": (
        "A mathematical privacy framework that provides a **provable guarantee**: "
        "the output distribution of a trained model changes by at most a bounded amount "
        "when any single training record is added or removed. "
        "This limits how much any individual's data influences the model, "
        "making it harder for an adversary to infer membership."
    ),
    "ε (epsilon) — Privacy Budget": (
        "Controls the strength of the privacy guarantee. "
        "**Lower ε = stronger privacy = more noise added to the model = lower accuracy.** "
        "Higher ε = weaker privacy = less noise = better accuracy. "
        "The dissertation evaluates ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} to show the full tradeoff curve. "
        "ε = 0 is impossible in practice; ε ≤ 1 is considered strong; ε ≤ 10 is acceptable for many applications."
    ),
    "δ (delta)": (
        "The probability that the DP guarantee fails. "
        "Set to 1×10⁻⁵ (0.001%) in this experiment, which is standard practice. "
        "Together (ε, δ) define an (ε, δ)-DP guarantee. "
        "δ should be much smaller than 1/N where N is the training set size."
    ),
    "DP-LR (DP Logistic Regression)": (
        "Logistic Regression trained with differential privacy using IBM's **diffprivlib** library. "
        "Noise is added directly to the objective function or gradient during optimisation. "
        "A simple linear model — easier to reason about theoretically but limited in expressiveness "
        "compared to neural networks."
    ),
    "DP-SGD (Differentially Private Stochastic Gradient Descent)": (
        "A neural network (2-hidden-layer MLP: input → 128 → 64 → output) trained with DP-SGD "
        "via **PyTorch Opacus**. "
        "The mechanism: (1) compute per-sample gradients, (2) **clip** each gradient to bound "
        "sensitivity (max_grad_norm = 1.0), (3) **add Gaussian noise** scaled to the clipping norm. "
        "This prevents any single record from dominating the gradient update."
    ),
    "RDP Accounting (Rényi DP)": (
        "Opacus tracks privacy cost across training epochs using the Rényi Differential Privacy "
        "accountant, which composes privacy losses across iterations more tightly than "
        "naïve composition. The **actual ε** spent is reported at the end of training "
        "and may be ≤ the target ε (if training stopped early)."
    ),
    "Privacy–Utility Tradeoff": (
        "The central result of the dissertation. "
        "As ε decreases (stronger privacy), model utility (F1, AUC) typically decreases because "
        "more noise is injected into the gradients. "
        "The goal is to find the lowest ε where utility remains within an acceptable range "
        "(e.g., ≤ 5% F1 loss vs non-private baseline)."
    ),
    "Gradient Clipping (max_grad_norm)": (
        "Before adding noise, each individual training sample's gradient is clipped to a "
        "maximum L2 norm (1.0 in this experiment). "
        "This bounds the **sensitivity** of the gradient — how much a single sample can "
        "change the update — so the noise added afterward is calibrated and meaningful."
    ),
    "Noise Multiplier": (
        "The ratio of the standard deviation of the Gaussian noise added to each gradient "
        "update relative to the clipping norm. "
        "A higher noise multiplier gives stronger privacy (smaller ε) but hurts convergence."
    ),
}

MIA_CONCEPTS: dict[str, str] = {
    "Membership Inference Attack (MIA)": (
        "An adversary queries the trained model with `predict_proba()` for both training records "
        "(members) and held-out records (non-members), then tries to distinguish the two groups "
        "based on output behaviour. "
        "The intuition: models tend to be **more confident and accurate on records they were "
        "trained on** — if this signal is detectable, membership is inferrable."
    ),
    "Black-box Attack": (
        "The attacker only observes model **outputs** (probability scores) — not internal weights, "
        "architecture, or training data. "
        "This is the most realistic and hardest-to-defend-against setting, and the most "
        "defensible choice for a dissertation because it reflects realistic adversarial access."
    ),
    "Attack Features": (
        "Features computed from model output for each record:\n\n"
        "- **max_conf**: Maximum predicted probability across all classes. High for members (model is confident).\n"
        "- **entropy**: Prediction entropy. Low for members (model is less uncertain).\n"
        "- **loss**: Cross-entropy loss for the true label. Low for members.\n"
        "- **correct**: 1 if the model's top prediction matches the true label. Higher for members.\n\n"
        "A logistic regression meta-classifier is trained on these features to predict membership."
    ),
    "Attack ROC-AUC": (
        "Measures the attacker's ability to rank members above non-members. "
        "**0.5 = random guessing** — the attacker has no advantage, strong privacy protection. "
        "**1.0 = perfect membership inference** — the model completely reveals training membership. "
        "Expected result: DP models (lower ε) should trend toward 0.5."
    ),
    "Attack Advantage": (
        "Attack ROC-AUC − 0.5. "
        "**0 = no information leaked** above random guessing. "
        "Positive values indicate the attacker has a measurable advantage. "
        "This is the primary metric for comparing privacy protection across models and ε values."
    ),
    "Member vs Non-member": (
        "**Members** are records from the model's actual training set. "
        "**Non-members** are records from the held-out test set that the model never saw during training. "
        "Both sets are sampled equally (balanced) to prevent the attack classifier from exploiting "
        "class imbalance. The same sampling seed is used for all target models so results are comparable."
    ),
    "TPR @ low FPR": (
        "True Positive Rate (correctly identified members) when the False Positive Rate is fixed at "
        "a low threshold (1% or 10%). "
        "This is the operationally relevant metric for **targeted attacks** — an adversary who wants "
        "to confirm membership for a specific individual while keeping false accusations low."
    ),
}

DATASET_CONCEPTS: dict[str, str] = {
    "CIC-IDS2018 (Primary Dataset)": (
        "Canadian Institute for Cybersecurity Intrusion Detection System 2018. "
        "Captured over 5 days in a realistic enterprise network environment. "
        "~16 GB of raw network flow CSV files containing ~80 features per flow "
        "extracted by CICFlowMeter. "
        "Attack types: DoS, DDoS, Brute Force, Botnet, Infiltration, Web Attacks. "
        "Used as the **primary training and evaluation dataset** in this dissertation."
    ),
    "UNSW-NB15 (Validation Dataset)": (
        "UNSW Canberra Network Intrusion dataset (2015). "
        "Captured in a different network environment using different tooling. "
        "~49 features per flow. "
        "Attack types: DoS, Fuzzing, Reconnaissance, Exploits, Backdoors, Worms. "
        "Used as the **cross-dataset validation set** to test whether findings generalise "
        "beyond the training dataset."
    ),
    "Binary vs Multiclass Labels": (
        "**Binary**: Each flow is classified as either Benign (0) or Attack (1). "
        "Simple, high-recall goal — detect any attack. "
        "Metrics include FPR and Detection Rate.\n\n"
        "**Multiclass**: Each flow is classified into its specific attack category "
        "(DoS, DDoS, BruteForce, etc.) or Benign. "
        "More challenging. F1 macro averages equally across all categories."
    ),
    "Train / Val / Test Splits (60/20/20)": (
        "The working subset (~300k rows) is stratified-split once and fixed for all experiments:\n\n"
        "- **Train (60%)**: Used to fit all models.\n"
        "- **Validation (20%)**: Used for hyperparameter tuning and early stopping.\n"
        "- **Test (20%)**: Held out until final evaluation. Never seen during training.\n\n"
        "Using the **same splits for all models** (baseline and DP) ensures that differences in "
        "performance are due to the model/privacy mechanism, not data variation."
    ),
    "Stratified Sampling": (
        "When drawing the working subset from the full dataset, each attack category is sampled "
        "proportionally so rare attack types are not dropped. "
        "The same stratification is applied to train/val/test splits. "
        "This ensures the class distribution in every split reflects the full dataset."
    ),
    "Feature Scaling (StandardScaler)": (
        "All numeric features are normalised to zero mean and unit variance using StandardScaler. "
        "The scaler is **fitted on the training split only** and then applied to validation and test. "
        "This prevents data leakage — the model never sees test-split statistics during training."
    ),
}


# ── Render functions ──────────────────────────────────────────────────────────

def _render_section(title: str, definitions: dict[str, str]) -> None:
    """Render one section of the glossary as a markdown table inside an expander."""
    for term, definition in definitions.items():
        st.markdown(f"**{term}**")
        st.markdown(definition)
        st.markdown("---")


def render_glossary(
    sections: list[str] | None = None,
    expanded: bool = False,
) -> None:
    """
    Render a collapsible metric glossary expander.

    Args:
        sections: Which sections to include. Options: "ids", "dp", "mia", "data".
                  Defaults to all sections.
        expanded: Whether the expander is open by default.
    """
    if sections is None:
        sections = ["ids", "dp", "mia", "data"]

    section_map = {
        "ids":  ("IDS Performance Metrics", IDS_METRICS),
        "dp":   ("Differential Privacy Concepts", DP_CONCEPTS),
        "mia":  ("Membership Inference Attack (MIA)", MIA_CONCEPTS),
        "data": ("Datasets and Experimental Setup", DATASET_CONCEPTS),
    }

    with st.expander("Definitions & Glossary — click to expand", expanded=expanded):
        for key in sections:
            if key in section_map:
                title, defs = section_map[key]
                st.subheader(title)
                _render_section(title, defs)


def render_metric_tooltip(metric: str) -> str:
    """
    Return a short one-line tooltip string for a given metric name.
    Suitable for use as the `help=` argument in st.metric().
    """
    tooltips = {
        "f1_macro": "Harmonic mean of precision and recall, averaged equally across classes. Preferred over accuracy for imbalanced IDS data.",
        "f1_weighted": "F1 score averaged weighted by class support. Higher weight on frequent classes.",
        "roc_auc": "Area Under ROC Curve. 0.5 = random, 1.0 = perfect. Threshold-independent discrimination measure.",
        "mcc": "Matthews Correlation Coefficient. Balanced metric for binary classification using all 4 confusion matrix cells. 0 = random, 1 = perfect.",
        "false_positive_rate": "FP / (FP + TN). Fraction of benign traffic incorrectly flagged as attacks. High FPR = alert fatigue.",
        "detection_rate": "TP / (TP + FN). Fraction of actual attacks correctly detected. Low DR = missed attacks.",
        "false_negative_rate": "FN / (FN + TP). Fraction of actual attacks missed. Lower is better.",
        "accuracy": "% of all records correctly classified. Not reliable for imbalanced IDS data — use F1 or MCC instead.",
        "precision_macro": "Of all records predicted as a given class, what fraction actually belonged to it. High precision = few false alarms.",
        "recall_macro": "Of all records of a given class, what fraction were detected. Same as Detection Rate for binary classification.",
        "attack_roc_auc": "Attacker's discrimination ability. 0.5 = random guessing (ideal for privacy). 1.0 = perfect membership inference.",
        "attack_advantage": "Attack ROC-AUC − 0.5. 0 = no information leaked. Positive = attacker has measurable advantage.",
        "attack_accuracy": "Fraction of records correctly classified as member/non-member by the attack meta-classifier.",
        "attack_f1": "F1 score of the membership inference attack classifier.",
        "tpr_at_fpr_0_01": "True positive rate (correctly identified members) when false positive rate is capped at 1%. Relevant for targeted membership inference.",
    }
    return tooltips.get(metric, "")
