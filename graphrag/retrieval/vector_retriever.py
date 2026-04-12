from typing import List, Dict, Any
import importlib
from ..graph.neo4j_manager import Neo4jManager
from ..utils.embeddings import EmbeddingGenerator
from ..config import get_settings


class VectorRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.settings = get_settings()
        self.reranker_model = "BAAI/bge-reranker-v2-m3"
        self.reranker_batch_size = 8
        self.reranker_max_length = 512
        self._reranker_tokenizer = None
        self._reranker_model = None
        self._reranker_device = "cpu"
        try:
            self.neo4j.create_vector_index(
                index_name="question_embeddings",
                label="Question",
                property_name="question_embedding"
            )
        except Exception as e:
            print(f"Error al crear el índice vectorial: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks relevantes en 2 etapas:
        1) vector search sobre preguntas hipotéticas,
        2) reranking con bge-reranker-v2-m3 vía Ollama.
        """
        top_k = top_k or self.settings.top_k_results
        top_k_candidates = 20

        # Generar embedding de la query
        query_embedding = self.embedding_gen.embed_text(query)

        # Recuperar candidatos por vector search de preguntas
        candidates = self._retrieve_question_candidates(query_embedding, top_k_candidates)
        if not candidates:
            return []

        # Rerank por relevancia query-document con bge-reranker-v2-m3
        reranked = self._rerank_candidates(query, candidates)
        if not reranked:
            # Fallback si el reranker no está disponible.
            return candidates[:top_k]

        return reranked[:top_k]

    def _retrieve_question_candidates(
            self,
            query_embedding: List[float],
            top_k_candidates: int
    ) -> List[Dict[str, Any]]:
        """Recupera candidatos desde el índice vectorial de preguntas."""
        cypher_query = """
        CALL db.index.vector.queryNodes('question_embeddings', $top_k_candidates, $query_embedding)
        YIELD node AS question, score
        MATCH (chunk:Chunk)-[:HAS_QUESTION]->(question)
        WITH chunk, max(score) AS score, collect(DISTINCT question.text) AS matched_questions
        RETURN chunk.text AS text,
               chunk.id AS chunk_id,
               score,
               matched_questions
        ORDER BY score DESC
        """

        return self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "top_k_candidates": top_k_candidates
        })

    def _rerank_candidates(
            self,
            query: str,
            candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerankea candidatos con Transformers (BAAI/bge-reranker-v2-m3)."""
        documents = [candidate.get("text", "") for candidate in candidates]
        if not documents:
            return []

        if not self._ensure_transformers_reranker_loaded():
            return []

        try:
            scores = self._score_with_transformers(query, documents)
        except Exception as e:
            print(f"Reranker no disponible, usando ranking vectorial original: {e}")
            return []

        rerank_results = sorted(
            [{"index": idx, "relevance_score": score} for idx, score in enumerate(scores)],
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        output: List[Dict[str, Any]] = []
        for item in rerank_results:
            idx = item.get("index")
            if idx is None or idx < 0 or idx >= len(candidates):
                continue

            ranked_item = dict(candidates[idx])
            ranked_item["rerank_score"] = item.get("relevance_score", 0.0)
            output.append(ranked_item)

        return output

    def _ensure_transformers_reranker_loaded(self) -> bool:
        """Carga perezosa del modelo de reranking."""
        if self._reranker_model is not None and self._reranker_tokenizer is not None:
            return True

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
            AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")
        except Exception as e:
            print(f"No se pudo importar transformers/torch: {e}")
            return False

        self._reranker_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model)
        self._reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model)
        self._reranker_model.to(self._reranker_device)
        self._reranker_model.eval()
        return True

    def _score_with_transformers(self, query: str, documents: List[str]) -> List[float]:
        """Calcula scores de relevancia para pares (query, document)."""
        torch = importlib.import_module("torch")

        scores: List[float] = []
        total = len(documents)

        for start in range(0, total, self.reranker_batch_size):
            batch_docs = documents[start:start + self.reranker_batch_size]
            batch_pairs = [[query, doc] for doc in batch_docs]

            encoded = self._reranker_tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.reranker_max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self._reranker_device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self._reranker_model(**encoded).logits

            if logits.dim() == 2:
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits

            scores.extend(batch_scores.detach().cpu().tolist())

        return [float(score) for score in scores]

    def retrieve_with_entities(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks relevantes junto con sus entidades.
        """
        top_k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_gen.embed_text(query)

        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:Entity)
        WITH chunk, score, collect(DISTINCT {
            name: e.name, 
            type: e.type, 
            summary: e.summary
        }) AS entities
        RETURN chunk.text AS text,
               chunk.id AS chunk_id,
               score,
               entities
        ORDER BY score DESC
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "top_k": top_k
        })

        return results


class HybridRetriever:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.settings = get_settings()
        self._create_fulltext_index()

    def _create_fulltext_index(self):
        """Crea un índice de texto completo."""
        query = """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
        FOR (c:Chunk)
        ON EACH [c.text]
        """
        try:
            self.neo4j.execute_query(query)
        except Exception as e:
            print(f"Índice fulltext ya existe o error: {e}")

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera chunks usando búsqueda híbrida (vectorial + texto completo).
        """
        top_k = top_k or self.settings.top_k_results
        query_embedding = self.embedding_gen.embed_text(query)

        cypher_query = """
        CALL {
            // Vector search
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
            YIELD node, score
            WITH collect({node: node, score: score}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            RETURN n.node AS node, (n.score / max) AS score
            UNION
            // Fulltext search
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query, {limit: $top_k})
            YIELD node, score
            WITH collect({node: node, score: score}) AS nodes, max(score) AS max
            UNWIND nodes AS n
            RETURN n.node AS node, (n.score / max) AS score
        }
        WITH node, max(score) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN node.text AS text,
               node.id AS chunk_id,
               score
        """

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "query": query,
            "top_k": top_k
        })

        return results