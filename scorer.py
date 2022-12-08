from evaluation_metrics.bleu.bleu import Bleu
from evaluation_metrics.cider.cider import Cider
from evaluation_metrics.meteor.meteor import Meteor
from evaluation_metrics.rouge.rouge import Rouge

# Calculates the evaluation scores and return them in form of dictionary
def compute_metrics(groundtruths, predictions):
    gts = {index: lines for (index, lines) in enumerate(groundtruths)}
    preds = {index: [lines.strip()] for (index, lines) in enumerate(predictions)}
    eval_scores = {}
    metrics = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    for metric, method in metrics:
        score, scores = metric.compute_score(gts, preds)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                eval_scores[m] = sc
        else:
            eval_scores[method] = score
    return eval_scores