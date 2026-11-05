"""The Portfolio Manager: decision support per option strategy, implementing the
PM Knowledge Base vault (~/Documents/project/PM_Knowledge_Base). Pure functions
only (no I/O), so the same code runs live and in replay. Every threshold is a
hypothesis carrying its vault register ID.

One Portfolio Manager, two management modules, one shared core. A strategy's
manager is its management category, how it was opened (core.review.manager:
debit -> long_options, credit -> short_options); its convexity class comes
from net greeks (core.classify) and both managers read it, never assume it.
long_options and short_options import core, never each other.

    core/            shared infrastructure (vault: shared_components)
      quality        data_quality_controls     flags, executable vs mid value
      classify       convexity_classification  long / short / mixed convexity
      efficiency     gamma_theta_efficiency    GT, GT_h, breakeven vol (long
                                               convexity), TG (short convexity)
      profit         profit_efficiency         return on debit, capture,
                                               remaining reward/risk
      assess         position_quality          one strategy's assessment
      review         hold_rules, close_rules   signal record, strategy state,
                                               P0 and class-change signals,
                                               the Phase 1 outcome, manager
      strategy_id    data_model                stable strategy IDs across rolls
    long_options/    Long Options Manager (vault: long_options_manager)
      signals        exit_signals              T1, T3-T8 review signals, decide
    short_options/   Short Options Manager (vault: short_options_manager)
      analysis       premium_capture_model,    premium to risk, loss multiple,
                     loss_control              distance to a short strike
      signals        short_exit_signals        S1-S5 review signals, decide
"""
