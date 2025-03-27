import json
from evaluation.eval_utils import *

if __name__ == "__main__":

    
    eval_collection = MetricCollection({
        "recall@5": RecallAtK(k=5),
        "recall@10": RecallAtK(k=10),
        "precision@5": PrecisionAtK(k=5),
        "precision@10": PrecisionAtK(k=10),
        "mrr": MRR(),
        "ndcg": nDCG(),
        "bleu": BLEU(),
        "rouge": ROUGE(),
        "bertscore": BERTScore()
    })
