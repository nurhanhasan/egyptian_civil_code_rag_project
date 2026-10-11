import os

import bentoml
from FlagEmbedding import BGEM3FlagModel

from ragapp.modules.rag.domain.entities import EmbeddingResult

EMBEDDER_MODEL_ID = "BAAI/bge-m3"
EMBEDDER_MODEL_REVISION_ID = "5617a9f61b028005a4858fdac845db406aefb181"


class _FakeEmbedder:
    """Deterministic fake embedder for tests."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.revision_id = "fake-embedder"

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        dense = [
            [1.0 if j == i % self.dim else 0.05 for j in range(self.dim)]
            for i in range(len(texts))
        ]
        sparse = [{100 + i: 1.0, 1000: 0.2} for i in range(len(texts))]
        return EmbeddingResult(dense=dense, sparse=sparse, revision_id=self.revision_id)

    def close(self) -> None:
        pass


@bentoml.service()
class BentoMLBGEEmbedder:
    """
    Embedder implementation that leverages BentoML SequenceTransformer model.
    """

    # model_path = HuggingFaceModel(MODEL_ID)

    def __init__(self):
        self.model_path = os.getenv("BGE_M3_MODEL_PATH", EMBEDDER_MODEL_ID)
        self.model = BGEM3FlagModel(self.model_path, use_fp16=True)
        self.revision_id = os.getenv(
            "EMBEDDER_MODEL_REVISION_ID", EMBEDDER_MODEL_REVISION_ID
        )

    def _to_embedding_result(self, raw: dict, revision_id: str) -> EmbeddingResult:
        dense = raw["dense_vecs"].tolist()  # ndarray[n,1024] -> list[list[float]]
        sparse = [
            {
                int(tok): float(w) for tok, w in lw.items()
            }  # keys are str(token_id) in FlagEmbedding 1.4.2
            for lw in raw["lexical_weights"]
        ]
        return EmbeddingResult(dense=dense, sparse=sparse, revision_id=revision_id)

    @bentoml.api(batchable=True, max_batch_size=32)
    async def embed(self, question: list[str]) -> EmbeddingResult:
        if not question:  # encode([]) crashes in np.concatenate
            return EmbeddingResult(dense=[], sparse=[], revision_id=self.revision_id)
        raw = self.model.encode(
            question, return_dense=True, return_sparse=True, return_colbert_vecs=False
        )
        return self._to_embedding_result(raw, self.revision_id)

    def close(self) -> None:
        self.model = None
