import opik
opik.configure(use_local=False)
#opik key sXwCOv34TR6gH668iw0gz63UY

from opik import Opik

client = Opik()
dataset = client.get_or_create_dataset(name="Test dataset")

from opik import track

@track
def my_llm_application(input: str) -> str:
    response = query_engine.query(input)
    return str(response)

def evaluation_task(x):
    return {
        "output": my_llm_application(x['input'])
    }

from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall
)

# Define the metrics
hallucination_metric = Hallucination()
answer_relevance_metric = AnswerRelevance()
context_precision_metric = ContextPrecision()
context_recall_metric = ContextRecall() 

from opik.evaluation import evaluate

evaluation = evaluate(
    dataset=dataset,
    task=evaluation_task,
    experiment_name = model_name,
    scoring_metrics=[hallucination_metric, answer_relevance_metric, context_precision_metric, context_recall_metric],
    experiment_config={
        "model": "gpt-3.5-turbo"
    }
)