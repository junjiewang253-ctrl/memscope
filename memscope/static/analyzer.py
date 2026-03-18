from memscope.schemas.report import StaticReport
from memscope.static.estimator import estimate_static_summary

def analyze_static(cfg):
    summary, ops = estimate_static_summary(cfg)
    return StaticReport(
        summary=summary, 
        operators=ops, 
        metadata={
            "mode": "static", 
            "model_type": cfg.model.model_type, 
            "dtype": cfg.train.dtype, 
        }
    )