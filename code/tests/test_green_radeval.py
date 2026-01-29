from RadEval import RadEval
import json

refs = [
    "Mild cardiomegaly with small bilateral pleural effusions and basilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]
hyps = [
    "Mildly enlarged cardiac silhouette with small pleural effusions and dependent bibasilar atelectasis.",
    "No pleural effusions or pneumothoraces.",
]

medical_evaluator = RadEval(
    do_radgraph=True,
    do_chexbert=True,
    do_green=True
)

results = medical_evaluator(refs=refs, hyps=hyps)
print(json.dumps(results, indent=2))