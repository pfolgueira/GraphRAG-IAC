from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field
from ..llm.ollama_client import OllamaClient
from ..llm.gemini_client import GeminiClient


# ==========================================
#   ENUMS
# ==========================================

class AnimalClassEnum(str, Enum):
    MAMMAL = "Mammal"
    REPTILE = "Reptile"
    BIRD = "Bird"
    FISH = "Fish"
    AMPHIBIAN = "Amphibian"

class SkeletalStructureEnum(str, Enum):
    VERTEBRATE = "Vertebrate"
    INVERTEBRATE = "Invertebrate"

class ReproductionMethodEnum(str, Enum):
    OVIPAROUS = "Oviparous"
    VIVIPAROUS = "Viviparous"
    OVOVIVIPAROUS = "Ovoviviparous"

class EnvironmentTypeEnum(str, Enum):
    AQUATIC = "Aquatic"
    TERRESTRIAL = "Terrestrial"
    AERIAL = "Aerial"

class ActivityCycleEnum(str, Enum):
    NOCTURNAL = "Nocturnal"
    DIURNAL = "Diurnal"
    CREPUSCULAR = "Crepuscular"

class SocialStructureEnum(str, Enum):
    SOLITARY = "Solitary"
    PAIR_LIVING = "Pair-living"
    FAMILY_GROUP = "Family group"
    HERD = "Herd"
    PACK = "Pack"
    COLONY = "Colony"
    EUSOCIAL = "Eusocial"

class DietTypeEnum(str, Enum):
    CARNIVORE = "Carnivore"
    HERBIVORE = "Herbivore"
    OMNIVORE = "Omnivore"

class FoodSourceEnum(str, Enum):
    GRASS = "Grass"
    LEAVES = "Leaves"
    FRUITS = "Fruits"
    SEEDS = "Seeds"
    BARK = "Bark"
    NECTAR = "Nectar"
    AQUATIC_PLANTS = "Aquatic Plants"

class ConservationStatusEnum(str, Enum):
    EX = "EX (Extinct)"
    EW = "EW (Extinct in the Wild)"
    CR = "CR (Critically Endangered)"
    EN = "EN (Endangered)"
    VU = "VU (Vulnerable)"
    NT = "NT (Near Threatened)"
    LC = "LC (Least Concern)"
    DD = "DD (Data Deficient)"
    NE = "NE (Not Evaluated)"


# ==========================================
#   ENTITIES
# ==========================================

class SpeciesNode(BaseModel):
    name: str = Field(..., description="Common name of the species in colloquial language.")
    weight_min_kg: Optional[float] = Field(None, description="Minimum recorded weight of an adult specimen in kilograms.")
    weight_max_kg: Optional[float] = Field(None, description="Maximum recorded weight of an adult specimen in kilograms.")
    length_min_m: Optional[float] = Field(None, description="Minimum recorded body length in meters.")
    length_max_m: Optional[float] = Field(None, description="Maximum recorded body length in meters.")
    height_min_m: Optional[float] = Field(None, description="Minimum recorded height to the highest point of the animal in meters.")
    height_max_m: Optional[float] = Field(None, description="Maximum recorded height to the highest point of the animal in meters.")
    top_speed_kmh: Optional[float] = Field(None, description="Maximum speed the species can reach.")
    lifespan_years: Optional[float] = Field(None, description="Lifespan of the species.")
    gestation_period_days: Optional[float] = Field(None, description="Average duration of embryo development until birth in days.")
    maturity_age_years: Optional[float] = Field(None, description="Estimated age in years at which the species reaches sexual maturity or independence.")
class FamilyNode(BaseModel):
    type: str = Field(..., description="Taxonomic classification category: Hominidae, Felidae, Canidae, Equidae, etc.")

class HabitatNode(BaseModel):
    type: str = Field(..., description="Type of natural environment or ecosystem: Forest, Desert, Grassland, Ocean, etc.")

class LocationNode(BaseModel):
    type: str = Field(..., description="The geographical location of the animal. "
            "STRICT RULE: You MUST generalize the location to a Country, Continent, Sea, or Ocean. "
            "Do NOT output specific regions, natural parks, or habitats. "
            "Examples: If the text says 'Amazon Rainforest', output 'Brazil' or 'South America'. "
            "If it says 'Serengeti', output 'Tanzania'. If it says 'Great Barrier Reef', output 'Pacific Ocean'.")
    
class AnimalClassNode(BaseModel):
    type: AnimalClassEnum = Field(..., description="Main biological category.")

class SkeletalStructureNode(BaseModel):
    type: SkeletalStructureEnum = Field(..., description="Anatomical classification.")

class ReproductionMethodNode(BaseModel):
    type: ReproductionMethodEnum = Field(..., description="Embryonic development mechanism.")

class EnvironmentTypeNode(BaseModel):
    type: EnvironmentTypeEnum = Field(..., description="Main physical environment where the species develops its life cycle.")

class ActivityCycleNode(BaseModel):
    type: ActivityCycleEnum = Field(..., description="Temporal pattern of biological activity.")

class SocialStructureNode(BaseModel):
    type: SocialStructureEnum = Field(..., description="Social organization pattern of a species.")

class DietTypeNode(BaseModel):
    type: DietTypeEnum = Field(..., description="Classification according to dietary diet.")

class FoodSourceNode(BaseModel):
    type: FoodSourceEnum = Field(..., description="Type of plant resource on which a herbivorous or omnivorous species feeds.")

class ConservationStatusNode(BaseModel):
    type: ConservationStatusEnum = Field(..., description="Extinction risk level according to IUCN.")


# ==========================================
#   RELATIONSHIPS
# ==========================================

class MemberOfFamilyRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Family")
    description: str = Field("MEMBER_OF_FAMILY", description="Indicates the taxonomic family to which the species belongs within the biological classification.")

class BelongsToClassRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Animal Class")
    description: str = Field("BELONGS_TO_CLASS", description="Specifies the biological class of the species.")

class HasSkeletalStructureRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Skeletal Structure")
    description: str = Field("HAS_SKELETAL_STRUCTURE", description="Defines the type of skeletal structure of the species.")

class ReproducesViaRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Reproduction Method")
    description: str = Field("REPRODUCES_VIA", description="Indicates the reproduction method of the species.")

class LivesInEnvironmentRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Environment Type")
    description: str = Field("LIVES_IN_ENVIRONMENT", description="Describes the physical environment where the species lives.")

class InhabitsRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Habitat")
    description: str = Field("INHABITS", description="Relates the species to the type of habitat or biome where it is usually found.")

class FoundInRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Location")
    description: str = Field("FOUND_IN", description="Indicates the geographical regions where the species is naturally present.")

class MigratesToRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Location")
    season: Optional[str] = Field(None, description="Indicates the season of the year when the migration takes place.")
    description: str = Field("MIGRATES_TO", description="Indicates the regions to which the species moves during migratory processes.")

class HasActivityCycleRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Activity Cycle")
    description: str = Field("HAS_ACTIVITY_CYCLE", description="Describes the period of the day when the species is mainly active.")

class OrganizedInRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Social Structure")
    description: str = Field("ORGANIZED_IN", description="Represents the form of social organization of the species.")

class HasDietTypeRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Diet Type")
    description: str = Field("HAS_DIET_TYPE", description="Defines the type of dietary diet of the species.")

class PreysOnRel(BaseModel):
    source: str = Field(..., description="Name of the predator Species")
    target: str = Field(..., description="Name of the prey Species")
    description: str = Field("PREYS_ON", description="Indicates that the species hunts and consumes another species as part of its diet.")

class FeedsOnRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Food Source")
    description: str = Field("FEEDS_ON", description="Represents the consumption of non-animal food sources.")

class HasConservationStatusRel(BaseModel):
    source: str = Field(..., description="Name of the Species")
    target: str = Field(..., description="Type of the Conservation Status")
    description: str = Field("HAS_CONSERVATION_STATUS", description="Indicates the conservation status of the species according to its extinction risk.")


# ==========================================
#   GLOBAL EXTRACTION CONTAINER
# ==========================================

class GraphExtraction(BaseModel):
    # Entities
    species: List[SpeciesNode] = Field(default_factory=list)
    families: List[FamilyNode] = Field(default_factory=list)
    animal_classes: List[AnimalClassNode] = Field(default_factory=list)
    skeletal_structures: List[SkeletalStructureNode] = Field(default_factory=list)
    reproduction_methods: List[ReproductionMethodNode] = Field(default_factory=list)
    environment_types: List[EnvironmentTypeNode] = Field(default_factory=list)
    habitats: List[HabitatNode] = Field(default_factory=list)
    locations: List[LocationNode] = Field(default_factory=list)
    activity_cycles: List[ActivityCycleNode] = Field(default_factory=list)
    social_structures: List[SocialStructureNode] = Field(default_factory=list)
    diet_types: List[DietTypeNode] = Field(default_factory=list)
    food_sources: List[FoodSourceNode] = Field(default_factory=list)
    conservation_statuses: List[ConservationStatusNode] = Field(default_factory=list)

    # Relationships
    member_of_family_rels: List[MemberOfFamilyRel] = Field(default_factory=list)
    belongs_to_class_rels: List[BelongsToClassRel] = Field(default_factory=list)
    has_skeletal_structure_rels: List[HasSkeletalStructureRel] = Field(default_factory=list)
    reproduces_via_rels: List[ReproducesViaRel] = Field(default_factory=list)
    lives_in_environment_rels: List[LivesInEnvironmentRel] = Field(default_factory=list)
    inhabits_rels: List[InhabitsRel] = Field(default_factory=list)
    found_in_rels: List[FoundInRel] = Field(default_factory=list)
    migrates_to_rels: List[MigratesToRel] = Field(default_factory=list)
    has_activity_cycle_rels: List[HasActivityCycleRel] = Field(default_factory=list)
    organized_in_rels: List[OrganizedInRel] = Field(default_factory=list)
    has_diet_type_rels: List[HasDietTypeRel] = Field(default_factory=list)
    preys_on_rels: List[PreysOnRel] = Field(default_factory=list)
    feeds_on_rels: List[FeedsOnRel] = Field(default_factory=list)
    has_conservation_status_rels: List[HasConservationStatusRel] = Field(default_factory=list)


# ==========================================
#   ENTITY EXTRACTOR CLASS
# ==========================================

class EntityExtractor:
    def __init__(self):
        self.client = GeminiClient()

    def extract_entities_and_relationships(
            self,
            text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts entities and relationships using structured_output to prevent KeyErrors.
        """
        system_prompt = """You are an expert zoology Knowledge Graph extractor.
        Your goal is to extract animal species, their biometrics, their classifications, and specific relationships from the text provided.
        CRITICAL RULES (ZERO TOLERANCE FOR HALLUCINATIONS)
        1. STRICT GROUNDING: You must base your extraction on the source text. DO NOT use pre-trained knowledge, common sense, or assumptions.
        2. SCHEMA & ENUM COMPLIANCE: Strictly adhere to the provided schema descriptions and ENUM constraints.
        3. ENTITY CONSISTENCY: Ensure the source and target of each relationship exactly match the extracted entities.
        """
        prompt = f"Extract zoological entities and their relationships from the following text:\n\n{text}"

        extraction: GraphExtraction = self.client.structured_output(
            prompt=prompt,
            schema=GraphExtraction,
            system_prompt=system_prompt
        )

        nodes_dict = {
            "Species": [x.model_dump(mode='json', exclude_none=True) for x in extraction.species],
            "Family": [x.model_dump(mode='json') for x in extraction.families],
            "AnimalClass": [x.model_dump(mode='json') for x in extraction.animal_classes],
            "SkeletalStructure": [x.model_dump(mode='json') for x in extraction.skeletal_structures],
            "ReproductionMethod": [x.model_dump(mode='json') for x in extraction.reproduction_methods],
            "EnvironmentType": [x.model_dump(mode='json') for x in extraction.environment_types],
            "Habitat": [x.model_dump(mode='json') for x in extraction.habitats],
            "Location": [x.model_dump(mode='json') for x in extraction.locations],
            "ActivityCycle": [x.model_dump(mode='json') for x in extraction.activity_cycles],
            "SocialStructure": [x.model_dump(mode='json') for x in extraction.social_structures],
            "DietType": [x.model_dump(mode='json') for x in extraction.diet_types],
            "FoodSource": [x.model_dump(mode='json') for x in extraction.food_sources],
            "ConservationStatus": [x.model_dump(mode='json') for x in extraction.conservation_statuses],
        }
        
        rels_dict = {
            "MEMBER_OF_FAMILY": [x.model_dump(mode='json', exclude_none=True) for x in extraction.member_of_family_rels],
            "BELONGS_TO_CLASS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.belongs_to_class_rels],
            "HAS_SKELETAL_STRUCTURE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_skeletal_structure_rels],
            "REPRODUCES_VIA": [x.model_dump(mode='json', exclude_none=True) for x in extraction.reproduces_via_rels],
            "LIVES_IN_ENVIRONMENT": [x.model_dump(mode='json', exclude_none=True) for x in extraction.lives_in_environment_rels],
            "INHABITS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.inhabits_rels],
            "FOUND_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.found_in_rels],
            "MIGRATES_TO": [x.model_dump(mode='json', exclude_none=True) for x in extraction.migrates_to_rels],
            "HAS_ACTIVITY_CYCLE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_activity_cycle_rels],
            "ORGANIZED_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.organized_in_rels],
            "HAS_DIET_TYPE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_diet_type_rels],
            "PREYS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.preys_on_rels],
            "FEEDS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.feeds_on_rels],
            "HAS_CONSERVATION_STATUS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_conservation_status_rels],
        }

        num_entities = sum(len(nodes_dict[n]) for n in nodes_dict)
        num_relationships = sum(len(rels_dict[r]) for r in rels_dict)
        print(f"Initial extraction: {num_entities} entities and {num_relationships} relationships")

        return nodes_dict, rels_dict
    
    def review_extraction(
            self,
            text: str,
            entities: Dict[str, List[Dict[str, Any]]],
            relationships: Dict[str, List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Review step to consolidate the entities and relationships extracted for a chunk.
        """

        system_prompt = """You are an expert data reviewer specialized in zoology Knowledge Graphs.
Your task is to review an initial extraction of entities and relationships from a text, correct any mistakes, remove hallucinations, and extract ANY missing information.
CRITICAL RULES:
1. STRICT GROUNDING: Every single entity or relationship you KEEP or ADD must be explicitly stated in the source text, if not, DELETE it. DO NOT use pre-trained knowledge, common sense, or assumptions.
2. SCHEMA & ENUM COMPLIANCE: Strictly adhere to the provided schema descriptions and ENUM constraints.
3. ENTITY CONSISTENCY: Ensure the source and target of each relationship exactly match the extracted entities."""

        prompt = f"""Review the text and the initial extraction (Entities and Relationships) to ensure that absolutely no invented information has been included
(everything must be explicitly stated in the text), and that all relevant entities and relationships present in the text are fully captured according to the schema.

TEXT: {text}

INITIAL ENTITIES: {entities}

INITIAL RELATIONSHIPS: {relationships}"""

        extraction: GraphExtraction = self.client.structured_output(
            prompt=prompt,
            schema=GraphExtraction,
            system_prompt=system_prompt
        )

        nodes_dict = {
            "Species": [x.model_dump(mode='json', exclude_none=True) for x in extraction.species],
            "Family": [x.model_dump(mode='json') for x in extraction.families],
            "AnimalClass": [x.model_dump(mode='json') for x in extraction.animal_classes],
            "SkeletalStructure": [x.model_dump(mode='json') for x in extraction.skeletal_structures],
            "ReproductionMethod": [x.model_dump(mode='json') for x in extraction.reproduction_methods],
            "EnvironmentType": [x.model_dump(mode='json') for x in extraction.environment_types],
            "Habitat": [x.model_dump(mode='json') for x in extraction.habitats],
            "Location": [x.model_dump(mode='json') for x in extraction.locations],
            "ActivityCycle": [x.model_dump(mode='json') for x in extraction.activity_cycles],
            "SocialStructure": [x.model_dump(mode='json') for x in extraction.social_structures],
            "DietType": [x.model_dump(mode='json') for x in extraction.diet_types],
            "FoodSource": [x.model_dump(mode='json') for x in extraction.food_sources],
            "ConservationStatus": [x.model_dump(mode='json') for x in extraction.conservation_statuses],
        }
        
        rels_dict = {
            "MEMBER_OF_FAMILY": [x.model_dump(mode='json', exclude_none=True) for x in extraction.member_of_family_rels],
            "BELONGS_TO_CLASS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.belongs_to_class_rels],
            "HAS_SKELETAL_STRUCTURE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_skeletal_structure_rels],
            "REPRODUCES_VIA": [x.model_dump(mode='json', exclude_none=True) for x in extraction.reproduces_via_rels],
            "LIVES_IN_ENVIRONMENT": [x.model_dump(mode='json', exclude_none=True) for x in extraction.lives_in_environment_rels],
            "INHABITS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.inhabits_rels],
            "FOUND_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.found_in_rels],
            "MIGRATES_TO": [x.model_dump(mode='json', exclude_none=True) for x in extraction.migrates_to_rels],
            "HAS_ACTIVITY_CYCLE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_activity_cycle_rels],
            "ORGANIZED_IN": [x.model_dump(mode='json', exclude_none=True) for x in extraction.organized_in_rels],
            "HAS_DIET_TYPE": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_diet_type_rels],
            "PREYS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.preys_on_rels],
            "FEEDS_ON": [x.model_dump(mode='json', exclude_none=True) for x in extraction.feeds_on_rels],
            "HAS_CONSERVATION_STATUS": [x.model_dump(mode='json', exclude_none=True) for x in extraction.has_conservation_status_rels],
        }

        num_entities = sum(len(nodes_dict[n]) for n in nodes_dict)
        num_relationships = sum(len(rels_dict[r]) for r in rels_dict)
        print(f"Initial extraction: {num_entities} entities and {num_relationships} relationships")
        
        return nodes_dict, rels_dict


    def summarize_entity(self, entity_name: str, descriptions: List[str]) -> str:
        """Consolidate descriptions (Entity Resolution step)."""
        system_prompt = "You are a specialist in zoological knowledge synthesis. Create a single third-person summary combining these biological descriptions."

        description_list = "\n- ".join(descriptions)
        user_message = f"Entity: {entity_name}\n\nDescriptions to merge:\n- {description_list}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        return self.client.chat(messages)

    def summarize_relationship(self, source: str, target: str, descriptions: List[str]) -> str:
        """Consolidate relationships."""
        prompt = f"Summarize the ecological or biological relationship between {source} and {target} based on: {descriptions}"
        return self.client.chat([{"role": "user", "content": prompt}])