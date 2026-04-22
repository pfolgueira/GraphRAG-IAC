from typing import List, Dict, Any, Callable, TypedDict

from ..retrieval.manual_retriever import ManualRetriever
from ..retrieval.vector_retriever import VectorRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.text2cypher import Text2CypherRetriever
from ..graph.neo4j_manager import Neo4jManager

# Definimos la estructura de la herramienta
class ToolData(TypedDict):
    function: Callable
    description: str

class RetrieverTools:
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.vector_retriever = VectorRetriever(neo4j_manager)
        self.hybrid_retriever = HybridRetriever(neo4j_manager)
        self.text2cypher = Text2CypherRetriever(neo4j_manager)
        self.manual_retriever = ManualRetriever(neo4j_manager)
        self.custom_tools: Dict[str, ToolData] = {}

    def register_custom_tool(self, name: str, function: Callable, description: str):
        """Registra una herramienta personalizada."""
        self.custom_tools[name] = {
            "function": function,
            "description": description
        }

    _GREETING_RESPONSE = (
        "Hello! I am a knowledge assistant specialized in the animal kingdom. "
        "You can ask me about their taxonomy, physical or behavioral traits, diet, habitats, and "
        "conservation status."
    )
    _OUT_OF_SCOPE_RESPONSE = (
        "Sorry, this question is outside my scope. "
        "I am designed to answer questions exclusively about zoology and animal biology. "
        "If you have a question about the animal world, I am ready to help!"
    )
    _SKILLS_RESPONSE = (
        "I can answer detailed questions about specific animals. You can ask me to explore "
        "their taxonomic classification (family, genus, species), their physical characteristics, "
        "what they eat, where they live, or their conservation status."
    )

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Obtiene las descripciones de todas las herramientas disponibles."""
        tools = [
            {
                "name": "greeting",
                "description": (
                    "Handle conversational messages that need no knowledge lookup: "
                    "greetings, farewells, or basic thanks. Do not use for questions about capabilities."
                ),
                "parameters": {}
            },
            {
                "name": "out_of_scope",
                "description": (
                    "Handle questions clearly unrelated to zoology, specific animal species, taxonomy, "
                    "or animal traits (e.g., politics, coding, sports, weather, etc.). "
                    "Questions about specific countries or regions are IN-SCOPE only if asking about "
                    "the animals that live there."
                ),
                "parameters": {}
            },
            {
                "name": "skills",
                "description": (
                    "Describe the system's capabilities. Use this when the user asks what you can do, "
                    "how you work, or what kind of questions about animals and zoology they can ask."
                ),
                "parameters": {}
            },
            {
                "name": "vector_search",
                "description": "Search for relevant information using semantic similarity. Best for finding conceptually related content.",
                "parameters": {
                    "query": "The search query in natural language"
                }
            },
            {
                "name": "hybrid_search",
                "description": "Search using both semantic similarity and keyword matching. Good for balanced retrieval.",
                "parameters": {
                    "query": "The search query in natural language"
                }
            },
            {
                "name": "text2cypher",
                "description": "Query the knowledge graph using natural language. Best for structured queries, aggregations, and complex graph traversals.",
                "parameters": {
                    "query": "The question in natural language"
                }
            },
            {
                "name": "predefined_cypher",
                "description": "Executes predefined, highly optimized graph queries for specific, complex zoology topics. "
                                "You MUST use this tool IF the user's question matches or closely relates to one of the "
                                "available predefined query categories. Do NOT use text2cypher or semantic search if the "
                                "intent matches one of these specific scenarios.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_category": {
                            "type": "string",
                            "description": (
                                "The category of the predefined query to execute. You must select the one that "
                                "best matches the user's question intent."
                            ),
                            "enum": [
                                "species_full_profile",
                                # "apex_predators_by_location",
                                # "migration_diet_analysis",
                                # "endangered_by_environment",
                                # "family_extremes_comparison"
                            ]
                        },
                        "species_name": {
                            "type": "string",
                            "description": "The name of the animal species. REQUIRED if query_category is 'species_full_profile'."
                        },

                        
                        # "location_name": {
                        #     "type": "string",
                        #     "description": "The name of the geographic location or region. REQUIRED if query_category is 'apex_predators_by_location' or 'migration_diet_analysis'."
                        # },
                        # "environment_name": {
                        #     "type": "string",
                        #     "description": "The type of environment (e.g., marine, desert). REQUIRED if query_category is 'endangered_by_environment'."
                        # },
                        # "family_name": {
                        #     "type": "string",
                        #     "description": "The taxonomic family name (e.g., Felidae). REQUIRED if query_category is 'family_extremes_comparison'."
                        # },
                        # "season_name": {
                        #     "type": "string",
                        #     "description": "The season of the year (e.g., winter, summer). REQUIRED if query_category is 'migration_diet_analysis'."
                        # }
                    },
                    "required": ["query_category"]
                }
            }
            
        ]

        # Añadir herramientas personalizadas
        for name, tool_info in self.custom_tools.items():
            tools.append({
                "name": name,
                "description": tool_info["description"],
                "parameters": {}
            })

        return tools

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Ejecuta una herramienta específica."""
        # ====================================================================
        # HERRAMIENTAS QUE NO REQUIEREN BÚSQUEDA DE CONOCIMIENTO EN EL SISTEMA
        # ====================================================================
        if tool_name == "greeting":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._GREETING_RESPONSE,
            }

        if tool_name == "out_of_scope":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._OUT_OF_SCOPE_RESPONSE,
            }

        if tool_name == "skills":
            return {
                "tool": tool_name,
                "results": [],
                "context": [],
                "direct_response": self._SKILLS_RESPONSE,
            }
        if tool_name == "predefined_cypher":
            cypher, results = self.manual_retriever.retrieve(
                query_category=kwargs.get("query_category", ""),
                species_name=kwargs.get("species_name", ""),
                # location_name=kwargs.get("location_name", ""),
                # environment_name=kwargs.get("environment_name", ""),
                # family_name=kwargs.get("family_name", ""),
                # season_name=kwargs.get("season_name", "")
            )
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": [str(r) for r in results]
            }
        if tool_name == "vector_search":
            results = self.vector_retriever.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results]
            }

        elif tool_name == "hybrid_search":
            results = self.hybrid_retriever.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "results": results,
                "context": [r["text"] for r in results]
            }

        elif tool_name == "text2cypher":
            cypher, results = self.text2cypher.retrieve(kwargs.get("query", ""))
            return {
                "tool": tool_name,
                "cypher": cypher,
                "results": results,
                "context": [str(r) for r in results]
            }

        elif tool_name in self.custom_tools:
            function = self.custom_tools[tool_name]["function"]
            results = function(**kwargs)
            return {
                "tool": tool_name,
                "results": results,
                "context": results if isinstance(results, list) else [str(results)]
            }

        else:
            raise ValueError(f"Unknown tool: {tool_name}")