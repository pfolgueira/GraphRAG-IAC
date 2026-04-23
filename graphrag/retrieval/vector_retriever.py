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
        top_k = top_k or self.settings.top_k_candidates

        # Generar embedding de la query
        query_embedding = self.embedding_gen.embed_text(query)

        # Recuperar candidatos por vector search de preguntas
        candidates = self._retrieve_question_candidates(query_embedding, top_k)
        
        return candidates

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

        results = self.neo4j.execute_query(cypher_query, {
            "query_embedding": query_embedding,
            "top_k_candidates": top_k_candidates
        })

        return results

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
