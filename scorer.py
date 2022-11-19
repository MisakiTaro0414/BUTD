from bleu.bleu import Bleu
from cider.cider import Cider
from meteor.meteor import Meteor
from rouge.rouge import Rouge

def _strip(s):
    return s.strip()

def compute_metrics(groundtruths, predictions):
    gts = {idx: strippedlines for (idx, strippedlines) in enumerate(groundtruths)}
    preds = {idx: [lines.strip()] for (idx, lines) in enumerate(predictions)}
    eval_scores = {}
    metrics = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for metric, method in metrics:
        score, scores = metric.compute_score(gts, preds)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                eval_scores[m] = sc
        else:
            eval_scores[method] = score
    return eval_scores