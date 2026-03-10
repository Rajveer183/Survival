import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def clinical_risk_stratification(score: float, mean_score: float) -> str:
    """
    Strata based on risk score relative to average.
    """
    ratio = score / mean_score if mean_score != 0 else 1.0
    if ratio < 0.8:
        return "Low Risk"
    elif ratio < 1.2:
        return "Medium Risk"
    else:
        return "High Risk"
