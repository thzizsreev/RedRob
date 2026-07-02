"""JD config and TRACER hybrid scorer weights."""

import re

BGE_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_FALLBACKS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
]
ARTIFACTS_DIR = "artifacts"

# TRACER structured weights (sum = 1.0) — semantic is pre-fused offline
WEIGHTS = {
    "semantic": 0.22,
    "title": 0.22,
    "career": 0.22,
    "skill": 0.12,
    "experience": 0.07,
    "location": 0.05,
    "assessment": 0.07,
}

# Offline semantic fusion: hybrid + multi-query + rerank boost
SEMANTIC_HYBRID_W = 0.55
SEMANTIC_MQ_W = 0.25
SEMANTIC_RERANK_W = 0.20

HYBRID_DENSE_WEIGHT = 0.65
HYBRID_SPARSE_WEIGHT = 0.35

HEAP_BUFFER = 150  # rank top-N then filter to 100

JD_FACETS = [
    "Production retrieval ranking embeddings hybrid search NDCG MAP FAISS vector DB Pinecone Milvus",
    "Senior AI ML engineer shipped ranking models sentence-transformers product company at scale",
    "LLM fine-tuning LoRA QLoRA learning-to-rank LightGBM XGBoost neural rankers PEFT",
    "India 5 to 9 years experience open to work recruiter response notice period Pune Noida",
]

JD_TEXT = """
Senior AI Engineer founding team role at Redrob AI talent intelligence platform.
Own ranking retrieval and matching systems for recruiter candidate search.
Production embeddings-based retrieval sentence-transformers BGE E5 deployed to real users.
Vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS.
Strong Python code quality. Evaluation frameworks NDCG MRR MAP offline-to-online A/B testing.
Ship end-to-end ranking search recommendation systems at product companies at scale.
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost LightGBM neural rankers.
Applied ML in production not pure research not LangChain-only demos.
5 to 9 years experience India Pune Noida Hyderabad Mumbai Delhi willing to relocate.
Not consulting-only not keyword stuffing not unrelated job titles.
Behavioral hireability recruiter response open to work.
""".strip()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TITLE_TIER1 = re.compile(
    r"\b("
    r"(Senior|Staff|Lead|Applied)\s+(AI|ML|Machine Learning|NLP)\s+Engineer|"
    r"(Senior|Staff|Lead)\s+(Machine Learning|NLP)\s+Engineer|"
    r"Recommendation Systems Engineer|"
    r"(Senior\s+)?(AI|ML|Machine Learning|NLP|Retrieval|Recommendation|Search)\s+Engineer"
    r")\b",
    re.I,
)
TITLE_TIER2 = re.compile(
    r"\b(AI|ML|Machine Learning|Data Scientist|Applied ML|Recommendation Systems|"
    r"Retrieval|Search Engineer|NLP Engineer|AI Specialist|Staff ML)\b",
    re.I,
)
TITLE_NEGATIVE = re.compile(
    r"\b(Marketing Manager|HR Manager|Accountant|Graphic Designer|Content Writer|"
    r"Customer Support|Sales Executive|Civil Engineer|Mechanical Engineer|"
    r"Project Manager|Operations Manager|Business Analyst)\b",
    re.I,
)

MUST_HAVE_SKILLS = [
    "embedding", "sentence-transformer", "sentence transformer", "retrieval", "ranking",
    "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "ndcg", "mrr", "map", "python", "hybrid search", "information retrieval",
]

NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "lightgbm", "xgboost",
    "learning to rank", "recsys", "recommendation", "rag", "llm", "transformer",
    "a/b test", "pgvector",
]

CAREER_POSITIVE = [
    "ranking model", "ranking system", "ranking models", "retrieval", "embedding",
    "embeddings", "vector search", "recommendation system", "recsys", "production ml",
    "shipped", "deployed", "ndcg", "mrr", "offline-to-online", "a/b test", "hybrid search",
    "sentence-transformer", "sentence transformer", "faiss", "pinecone", "milvus",
    "learning to rank", "lightgbm", "xgboost", "dense retrieval", "re-ranking", "reranking",
    "information retrieval", "offline-to-online", "cross-encoder", "bi-encoder",
]

CAREER_NEGATIVE = [
    "langchain tutorial", "only langchain", "pure research", "academic lab",
    "research-only", "no production",
]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "mindtree",
    "hcl", "tech mahindra", "ltimindtree", "lti", "mphasis",
}

PRODUCT_INDICATORS = [
    "startup", "saas", "product", "series a", "series b", "unicorn", "platform", "marketplace",
]

PREFERRED_CITIES = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "chennai", "ncr",
]

AI_SKILL_PATTERN = re.compile(
    r"embed|retriev|rank|vector|llm|lora|ndcg|mrr|faiss|pinecone|milvus|"
    r"sentence.?transform|rag|nlp|transformer|lightgbm|xgboost|weaviate|qdrant",
    re.I,
)

PROFICIENCY_WEIGHT = {
    "beginner": 0.25, "intermediate": 0.55, "advanced": 0.85, "expert": 1.0,
}

IDEAL_YOE_MIN = 5.0
IDEAL_YOE_MAX = 9.0
SOFT_YOE_MIN = 4.0
SOFT_YOE_MAX = 12.0

STUFFER_SKILL_THRESHOLD = 0.65
STUFFER_TITLE_THRESHOLD = 0.20
STUFFER_SEMANTIC_FLOOR = 0.52
