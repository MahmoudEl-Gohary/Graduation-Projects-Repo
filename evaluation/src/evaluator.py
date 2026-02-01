import os
from pathlib import Path

# --- CONFIG: Set paths for helper libraries ---
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / ".resources"

os.environ["NLTK_DATA"] = str(RESOURCES_DIR / "nltk_data")
os.environ["STANZA_RESOURCES_DIR"] = str(RESOURCES_DIR / "stanza_resources")

from RadEval import RadEval


class RadiologyEvaluator:
    """
    Wrapper for RadEval to evaluate radiology report generation.
    
    Supported metrics:
        - radcliq: RadCliQ-v1
        - bleu: BLEU score
        - bertscore: BertScore
        - semb: SembScore
        - radgraph: RadGraph F1
        - ratescore: RaTEScore
        - green: GREEN score
    """
    
    AVAILABLE_METRICS = [
        "radcliq", "bleu", "bertscore", "semb", "radgraph", "ratescore", "green"
    ]
    
    def __init__(
        self,
        metrics: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of metrics to compute. If None, computes all metrics.
                     Options: radcliq, bleu, bertscore, semb, radgraph, ratescore, green
            **kwargs: Additional arguments passed to RadEval
        """
        if metrics is None:
            metrics = self.AVAILABLE_METRICS
        
        # Validate metrics
        invalid = set(metrics) - set(self.AVAILABLE_METRICS)
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. Available: {self.AVAILABLE_METRICS}")
        
        self.metrics = metrics
        
        # Build RadEval config
        eval_config = {
            "do_radcliq": "radcliq" in metrics,
            "do_bleu": "bleu" in metrics,
            "do_bertscore": "bertscore" in metrics,
            "do_chexbert": "semb" in metrics,
            "do_radgraph": "radgraph" in metrics,
            "do_ratescore": "ratescore" in metrics,
            "do_green": "green" in metrics,
        }
        eval_config.update(kwargs)
        
        self._evaluator = RadEval(**eval_config)
    
    def evaluate(
        self,
        references: list[str],
        predictions: list[str]
    ) -> dict:
        """
        Evaluate predictions against references.
        
        Args:
            references: List of ground truth reports
            predictions: List of predicted reports
            
        Returns:
            Dictionary containing metric scores
        """
        if len(references) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(references)} references vs {len(predictions)} predictions"
            )
        
        results = self._evaluator(refs=references, hyps=predictions)
        return results
    
    def __call__(self, references: list[str], predictions: list[str]) -> dict:
        return self.evaluate(references, predictions)