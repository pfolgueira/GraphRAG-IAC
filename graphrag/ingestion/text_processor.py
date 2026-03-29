import uuid
from typing import List, Dict, Any
from tqdm import tqdm
from ..graph.neo4j_manager import Neo4jManager
from ..utils.chunking import chunk_text
from ..utils.embeddings import EmbeddingGenerator
from .entity_extractor import EntityExtractor


class TextProcessor:
    def __init__(
            self,
            neo4j_manager: Neo4jManager,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
            entity_types: List[str] = None,
    ):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.entity_extractor = EntityExtractor(entity_types=entity_types)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(
            self,
            text: str,
            document_id: str = None,
            metadata: Dict[str, Any] = None
    ):
        """
        Procesa un documento: lo divide en chunks, extrae entidades y relaciones,
        y lo almacena en Neo4j.
        """
        document_id = document_id or str(uuid.uuid4())
        metadata = metadata or {}

        # 1. Crear nodo de documento
        self._create_document_node(document_id, metadata)

        # 2. Dividir en chunks
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        print(f"Documento dividido en {len(chunks)} chunks")

        # 3. Procesar cada chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Procesando chunks")):
            chunk_id = f"{document_id}_chunk_{i}"
            self._process_chunk(chunk_id, chunk.page_content, chunk.metadata, document_id, i)

        # 4. Consolidar entidades y relaciones
        self._consolidate_entities()
        self._consolidate_relationships()

        print(f"Documento {document_id} procesado exitosamente")

    def _create_document_node(self, document_id: str, metadata: Dict[str, Any]):
        """Crea un nodo de documento en Neo4j."""
        query = """
        MERGE (d:Document {id: $document_id})
        SET d += $metadata
        """
        self.neo4j.execute_query(query, {
            "document_id": document_id,
            "metadata": metadata
        })

    def _process_chunk(
            self,
            chunk_id: str,
            text: str,
            metadata: Dict[str, Any],
            document_id: str,
            index: int
    ):
        """Procesa un chunk individual."""
        # Generar embedding
        embedding = self.embedding_gen.embed_text(text)

        # Crear nodo de chunk
        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.embedding = $embedding,
            c.index = $index
            c.metadata = $metadata
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
            "index": index,
            "document_id": document_id
        })

        # Extraer entidades y relaciones
        entities, relationships = self.entity_extractor.extract_entities_and_relationships(text)

        # Almacenar entidades
        for label, entities_list in entities.items():
            for entity_data in entities_list:
                self._store_entity(label, entity_data, chunk_id)

        # Almacenar relaciones
        for rel_type, rels_list in relationships.items():
            for rel_data in rels_list:                
                self._store_relationship(rel_type, rel_data, chunk_id)

    def _store_entity(self, label: str, entity_data: Dict[str, Any], chunk_id: str):
        """Almacena una entidad y la conecta al chunk, manejando descripciones faltantes."""
        # Detecta si la clave principal es 'name' (para Species) o 'type' (para el resto)
        primary_value = entity_data.get("name") or entity_data.get("type")
        
        if not primary_value:
            return

        # Extrae el resto de propiedades ignorando las claves principales
        properties = {
            k: v for k, v in entity_data.items() 
            if k not in ["name", "type"]
        }

        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (e:{label} {{id: $primary_value}})
        SET e.name = $primary_value
        SET e += $properties
        MERGE (c)-[:HAS_ENTITY]->(e)
        """
        
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "primary_value": primary_value,
            "properties": properties
        })

    def _store_relationship(self, rel_type: str, rel_data: Dict[str, Any], chunk_id: str):
        """
        Almacena una relación semántica asegurando primero que el chunk de origen existe.
        """
        
        # Entidades implicadas en la relación
        source_id = rel_data.get("source")
        target_id = rel_data.get("target")

        if hasattr(source_id, 'value'): source_id = source_id.value
        if hasattr(target_id, 'value'): target_id = target_id.value

        if not source_id or not target_id:
            print(f"Advertencia: Relación [{rel_type}] incompleta u omitida en chunk {chunk_id}")
            return

        # Propiedades extra de la relación
        properties = {
            k: (v.value if hasattr(v, 'value') else v) 
            for k, v in rel_data.items() 
            if v is not None and k not in ["source", "target"]
        }

        # Consulta Cypher
        query = f"""
        // Validamos que el Chunk existe (tu petición original)
        MATCH (c:Chunk {{id: $chunk_id}})
        
        // Buscamos las entidades de origen y destino por su ID universal
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        
        // Creamos la relación dinámica entre las entidades
        MERGE (source)-[r:{rel_type}]->(target)
        
        // Inyectamos las propiedades extra
        SET r += $properties
        """
        
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties
        })

    def _consolidate_entities(self):
        """Consolida las descripciones de las entidades."""
        query = """
        MATCH (e:Entity)
        WHERE size(e.description) > 1
        RETURN e.name AS name, e.description AS descriptions
        """
        entities = self.neo4j.execute_query(query)

        for entity in tqdm(entities, desc="Consolidando entidades"):
            summary = self.entity_extractor.summarize_entity(
                entity["name"],
                entity["descriptions"]
            )

            update_query = """
            MATCH (e:Entity {name: $name})
            SET e.summary = $summary
            """
            self.neo4j.execute_query(update_query, {
                "name": entity["name"],
                "summary": summary
            })

    def _consolidate_relationships(self):
        """Consolida las descripciones de las relaciones."""
        query = """
        MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity)
        WHERE size(r.description) > 1
        RETURN s.name AS source, t.name AS target, r.description AS descriptions, r.strength AS strengths
        """
        relationships = self.neo4j.execute_query(query)

        for rel in tqdm(relationships, desc="Consolidando relaciones"):
            summary = self.entity_extractor.summarize_relationship(
                rel["source"],
                rel["target"],
                rel["descriptions"]
            )

            avg_strength = sum(rel["strengths"]) / len(rel["strengths"])

            update_query = """
            MATCH (s:Entity {name: $source})-[r:RELATIONSHIP]->(t:Entity {name: $target})
            SET r.summary = $summary,
                r.avg_strength = $avg_strength
            """
            self.neo4j.execute_query(update_query, {
                "source": rel["source"],
                "target": rel["target"],
                "summary": summary,
                "avg_strength": avg_strength
            })