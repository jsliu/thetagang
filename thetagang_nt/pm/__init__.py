"""The Portfolio Manager: decision support per option strategy, implementing the
PM Knowledge Base vault (~/Documents/project/PM_Knowledge_Base). One module per
vault note; pure functions only (no I/O), so the same code runs live and in
replay. Every threshold is a hypothesis carrying its vault register ID.

    quality     data_quality_controls   flags, executable vs mid value
    classify    convexity_classification  long / short / mixed convexity
    efficiency  gamma_theta_efficiency  GT, GT_h, breakeven vol, TG
    profit      profit_efficiency       return on debit, capture, remaining reward/risk
    signals     exit_signals            P0 / P1 / P2 review signals and the outcome
    strategy_id data_model              stable strategy IDs across rolls
"""
