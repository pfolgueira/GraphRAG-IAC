import uuid
from typing import List, Dict, Any
from tqdm import tqdm
from ..graph.neo4j_manager import Neo4jManager
from ..utils.chunking import chunk_text
from ..utils.embeddings import EmbeddingGenerator
from .entity_extractor import EntityExtractor
from typing import Literal
from pydantic import BaseModel, Field
from ..llm.ollama_client import OllamaClient

class SpeciesResolution(BaseModel):
        status: Literal["MATCH", "NEW"] = Field(..., description="MUST BE 'MATCH' if the species matches one in the canonical list of species names, or 'NEW' if it is a completely different species not in the list.")
        resolved_name: str = Field(..., description="If status is 'MATCH', this MUST be the exact name from the canonical list. If status is 'NEW', this MUST be the most standard, common English name for the extracted species.")


class TextProcessor:
    def __init__(
            self,
            neo4j_manager: Neo4jManager,
            chunk_size: int = 500,
            chunk_overlap: int = 50,
    ):
        self.neo4j = neo4j_manager
        self.embedding_gen = EmbeddingGenerator()
        self.entity_extractor = EntityExtractor()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = OllamaClient()
        self.species_names = self._load_species_names()


    def _load_species_names(self):
        try:
            with open('../scraping/species.txt', 'r', encoding='utf-8') as fichero:
                lista = [linea.strip() for linea in fichero.readlines() if linea.strip()]
            return lista
        except FileNotFoundError:
            print(f"El fichero no existe.")
            return []


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
        #self._consolidate_entities()
        #self._consolidate_relationships()

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
        SET c += $metadata
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
        """Almacena una entidad y la conecta al chunk, 
           aplicando Entity Resolution a los nombres de las especies."""
        # Detecta si la clave principal es 'name' (para Species) o 'type' (para el resto)
        primary_value = entity_data.get("name") or entity_data.get("type")
        
        if not primary_value:
            return
        
        # Entitie resolution para las Species
        if label == "Species":
            primary_value = self._resolve_species_name(primary_value)

        # Extrae el resto de propiedades ignorando las claves principales
        properties = {
            k: v for k, v in entity_data.items() 
            if k not in ["name", "type"] and v is not None
        }

        primary_key = "name" if label == "Species" else "type"

        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (e:{label} {{{primary_key}: $primary_value}})
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
        Almacena una relación semántica asegurando la compatibilidad con el esquema
        y aplicando Entity Resolution a las especies implicadas.
        """
        
        # Entidades implicadas en la relación
        source_val = rel_data.get("source")
        target_val = rel_data.get("target")

        if hasattr(source_val, 'value'): source_val = source_val.value
        if hasattr(target_val, 'value'): target_val = target_val.value

        if not source_val or not target_val:
            print(f"Relación [{rel_type}] omitida: Faltan origen o destino en el chunk {chunk_id}")
            return

        # Aplicaa Entity Resolution a la especie de origen
        source_val = self._resolve_species_name(source_val)

        # Si la relación es de depredación, el destino también es una especie y requiere resolución
        if rel_type == "PREYS_ON":
            target_val = self._resolve_species_name(target_val)

        # 3. Mapeo de etiquetas y propiedades basado estrictamente en tu diseño
        source_label = "Species"
        source_prop = "name"

        # Mapeamos el tipo de relación con la etiqueta del nodo de destino y su clave
        target_mapping = {
            "MEMBER_OF_FAMILY": ("Family", "type"),
            "BELONGS_TO_CLASS": ("AnimalClass", "type"),
            "HAS_SKELETAL_STRUCTURE": ("SkeletalStructure", "type"),
            "REPRODUCES_VIA": ("ReproductionMethod", "type"),
            "LIVES_IN_ENVIRONMENT": ("EnvironmentType", "type"),
            "INHABITS": ("Habitat", "type"),
            "FOUND_IN": ("Location", "type"),
            "MIGRATES_TO": ("Location", "type"),
            "HAS_ACTIVITY_CYCLE": ("ActivityCycle", "type"),
            "ORGANIZED_IN": ("SocialStructure", "type"),
            "HAS_DIET_TYPE": ("DietType", "type"),
            "PREYS_ON": ("Species", "name"),
            "FEEDS_ON": ("FoodSource", "type"),
            "HAS_CONSERVATION_STATUS": ("ConservationStatus", "type")
        }

        target_label, target_prop = target_mapping.get(rel_type, ("Unknown", "type"))
        
        if target_label == "Unknown":
            print(f"Relación [{rel_type}] no reconocida en el esquema. Omitiendo.")
            return

        # Extraer propiedades adicionales de la relación
        properties = {
            k: (v.value if hasattr(v, 'value') else v) 
            for k, v in rel_data.items() 
            if k not in ["source", "target", "description"] and v is not None
        }

        query = f"""
        MATCH (c:Chunk {{id: $chunk_id}})
        MERGE (source:{source_label} {{{source_prop}: $source_val}})
        MERGE (target:{target_label} {{{target_prop}: $target_val}})
        MERGE (c)-[:HAS_ENTITY]->(source)
        MERGE (c)-[:HAS_ENTITY]->(target)
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        """
        
        self.neo4j.execute_query(query, {
            "chunk_id": chunk_id,
            "source_val": source_val,
            "target_val": target_val,
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

    def _resolve_species_name(self, extracted_name: str) -> str:
        """
        Utiliza el LLM para evaluar si el nombre extraído es un sinónimo, 
        variación o el mismo animal que alguno de la lista species_names. Si es nuevo, obtiene su nombre común.
        """
        # Si los nombres de la especie coinciden exactamente no se realiza la llamada al LLM
        if extracted_name in self.species_names:
            return extracted_name

        system_prompt = """You are an expert zoologist and taxonomist. 
        Your task is Entity Resolution for an animal Knowledge Graph.
        You will be given an extracted animal name and a canonical list of species.
        
        Rules:
        1. If the extracted name is a known synonym, regional name, or refers to the exact same biological species as one in the canonical list (e.g., 'Cougar' -> 'Mountain Lion', 'Orca' -> 'Killer Whale'), set status to 'MATCH' and output the exact matching name from the canonical list.
        2. If the extracted name represents a completely different species not present in the list (e.g., a new predator or prey), set status to 'NEW' and provide the most standard, common English name for this new species in 'resolved_name'.
        """

        user_message = f"Extracted Name: {extracted_name}\n\nCanonical List: {self.species_names}"

        resolution: SpeciesResolution = self.client.structured_output(
            prompt=user_message,
            schema=SpeciesResolution,
            system_prompt=system_prompt
        )

        final_name = resolution.resolved_name

        # Si el LLM determina que es una especie nueva, la añadimos a la lista
        if resolution.status == "NEW" and final_name not in self.species_names:
            self.species_names.append(final_name)
            print(f"Nueva especie añadida: {final_name}")
            
        return final_name