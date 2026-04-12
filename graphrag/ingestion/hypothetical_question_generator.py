from typing import List
from pydantic import BaseModel, Field
from ..llm.gemini_client import GeminiClient

# ==========================================
#   HYPOTHETICAL QUESTION GENERATOR CLASS
# ==========================================

class HypotheticalQuestionGenerator:
    def __init__(self):
        self.client = GeminiClient()

    def generate_hypothetical_questions(
            self,
            text: str
    ) -> List[str]:
        """
        Generates up to 10 hypothetical questions for the given chunk text.
        """
        class HypotheticalQuestionsResponse(BaseModel):
            questions: List[str] = Field(
                default_factory=list,
                description="List of questions that can be answered by the text"
            )

        if not text or not text.strip():
            return []

        system_prompt = (
            "You are an expert in Retrieval-Augmented Generation (RAG) and your specific domain of expertise is Zoology. "
            "Your task is to generate hypothetical user search queries that are directly answered by the provided text. "
            "Think like a real user searching for information about animals. "
            "CRITICAL: Never use meta-references like 'According to the text', 'In this chunk', or 'Based on the provided information'. "
            "The questions must sound natural and independent."
        )

        prompt = (
            "Generate up to 10 hypothetical search questions that this specific chunk of text answers.\n\n"
            "Rules:\n"
            "1. Answerability: The exact answer to every question MUST be explicitly found within the chunk.\n"
            "2. User Persona: Write the questions as a human would type them into a search bar (concise, direct, and natural).\n"
            "3. Diversity: Mix specific factual questions (e.g., 'What is the average lifespan of an elephant?') with broader conceptual questions (e.g., 'How do polar bears survive in extreme cold temperatures?') if applicable.\n"
            "4. Volume: If the chunk contains dense information, return up to 10. If it is sparse or short, return up to 5.\n"
            f"Chunk:\n{text}"
        )

        try:
            response = self.client.structured_output(
                prompt=prompt,
                schema=HypotheticalQuestionsResponse,
                system_prompt=system_prompt,
            )
        except Exception:
            return []

        clean_questions: List[str] = []
        seen = set()

        for question in response.questions:
            if not isinstance(question, str):
                continue

            normalized = " ".join(question.strip().split())
            if not normalized:
                continue

            if not normalized.endswith("?"):
                normalized = f"{normalized}?"

            normalized_lower = normalized.lower()
            if normalized_lower in seen:
                continue

            seen.add(normalized_lower)
            clean_questions.append(normalized)

            if len(clean_questions) >= 10:
                break

        return clean_questions

        