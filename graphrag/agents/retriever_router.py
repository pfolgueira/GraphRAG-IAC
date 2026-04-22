from typing import List, Dict, Any, Optional
from ..llm.ollama_client import OllamaClient
from .retriever_tools import RetrieverTools
from typing import Literal
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    """Schema for the LLM's routing decision."""

    tool: Literal[
        "vector_search", "hybrid_search", "text2cypher",
        "greeting", "out_of_scope", "skills"
    ] = Field(
        ...,
        description="The name of the tool selected to handle the query."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this tool was chosen based on the question."
    )
    query: str = Field(
        ...,
        description="The reformulated or original query to be passed to the tool."
    )


class RetrieverRouter:
    def __init__(self, retriever_tools: RetrieverTools):
        self.tools = retriever_tools
        self.client = OllamaClient()

    def route(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> RouterDecision:
        """
        Selecciona la mejor herramienta para responder la pregunta.
        """
        conversation_history = conversation_history or []

        # Obtener descripciones de herramientas
        tool_descriptions = self.tools.get_tool_descriptions()
        tools_str = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tool_descriptions
        ])

#         system_prompt = f"""You are a routing assistant for a knowledge base about zoology and animal biology.
# Select the single best tool for the user's question.

# Available tools:
# {tools_str}

# ROUTING RULES — apply in order:

# 1. greeting     — Use when the message is conversational and needs NO knowledge lookup:
#    greetings ("Hello", "Hi", "Good morning"), farewells ("Goodbye", "Thanks"),
#    questions about the system ("What can you do?", "How do you work?", "What topics do you cover?").

# 2. out_of_scope — Use when the question is clearly unrelated to zoology or animals:
#    weather, sports, cooking, geography unrelated to animals, movies, music, politics.
#    Examples: "What is the capital of France?", "Who won the World Cup?".

# 3. skills - Use when the user asks about the system's capabilities in a general way, but not for specific questions about animals.

# 4. vector_search — 

# 5. hybrid_search — 

# 6. text2cypher   — 

# Choose greeting or out_of_scope FIRST before considering any retrieval tool.
# """

        system_prompt = f"""You are a routing assistant for a knowledge base about zoology and animal biology.
Select the single best tool for the user's question.

Available tools:
{tools_str}

 ROUTING RULES — apply in order:

 1. greeting     — Use when the message is conversational and needs NO knowledge lookup:
    greetings ("Hello", "Hi", "Good morning"), farewells ("Goodbye", "Thanks"),
    questions about the system ("What can you do?", "How do you work?", "What topics do you cover?").

 2. out_of_scope — Use when the question is clearly unrelated to zoology or animals:
    weather, sports, cooking, geography unrelated to animals, movies, music, politics.
    Examples: "What is the capital of France?", "Who won the World Cup?".

 3. skills - Use when the user asks about the system's capabilities in a general way, but not for specific questions about animals.

 4. predefined_cypher - Check if user's query matches a predefined Cypher query. Use when the question can be answered by a predefined Cypher query. This is for common, well-known questions that have a direct mapping to a Cypher query. 

 5. hybrid_search

 6. text2cypher 

# Choose greeting or out_of_scope FIRST before considering any retrieval tool.
"""

        messages = [
                       {"role": "system", "content": system_prompt}
                   ] + conversation_history + [
                       {"role": "user", "content": f"Question: {question}"}
                   ]

        return self.client.structured_output_with_chat(messages, schema=RouterDecision)

    def retrieve(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Selecciona la herramienta apropiada y ejecuta la búsqueda.
        """
        decision = self.route(question, conversation_history)

        tool_name = decision.tool
        query = decision.query or question

        result = self.tools.execute_tool(tool_name, query=query)
        result["routing_decision"] = decision

        return result
