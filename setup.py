from deepeval.models import OllamaModel
from llm_client import LLMClient
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.summarization.summarization import SummarizationMetric
from deepeval.metrics.toxicity.toxicity import ToxicityMetric
from deepeval.metrics.hallucination.hallucination import HallucinationMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.bias.bias import BiasMetric

# Desired local model to be run 
model = OllamaModel(
    model="gemma3:1b",
    base_url="http://127.0.0.1:11434",
    temperature=0
)

# Setup Ollama client
ollama_client = LLMClient(model='gemma3:1b')

# Setup metrics for Ollama model
answer_relevancy = AnswerRelevancyMetric(model=model)
bias = BiasMetric(model=model)
toxicity = ToxicityMetric(model=model)
hallucination = HallucinationMetric(model=model)
faithfulness = FaithfulnessMetric(model=model)
summarization = SummarizationMetric(model=model)