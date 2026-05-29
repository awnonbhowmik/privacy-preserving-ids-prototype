"""
model_inversion.py — Model-inversion evaluation (deferred).

Developer note:
    Model inversion for tabular network-flow data is not implemented in this
    phase. Membership inference was prioritised because it is the stronger and
    more directly testable privacy claim for IDS classifiers.

    For image or text models, model inversion is well-studied. For tabular
    flow-feature data (CIC-IDS2018, UNSW-NB15) the threat model is weaker:
    individual flow features have no natural human-interpretable "image",
    making reconstruction attacks harder to define, evaluate, and explain in
    a dissertation defence.

    If implemented in a future phase, a defensible tabular approach would be:
        1. Gradient-based feature reconstruction: for a differentiable target
           model (e.g. DP-SGD MLP), minimise the difference between model
           output and a target class by optimising input features via backprop.
        2. Reconstruction quality metric: mean absolute error or cosine
           similarity between reconstructed and true feature vectors.
        3. Comparison: reconstruction error should be higher (worse for
           attacker) under tighter DP (lower ε).

    This is left as future work and does not affect the dissertation's primary
    privacy-utility evaluation or the membership-inference experiments.
"""
