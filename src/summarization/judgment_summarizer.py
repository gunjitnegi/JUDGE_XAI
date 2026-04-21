import requests
import json
from typing import List, Dict, Any

class JudgmentSummarizer:
    """
    Summarizes legal judgments section by section using a local LLM.
    Uses already-classified chunks to group text by section before summarizing.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model = model
        self.sections_to_summarize = ["FACTS", "ISSUES", "ARGUMENTS", "REASONING", "JUDGMENT"]

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 500,
                "num_ctx": 4096
            }
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return "Summary unavailable."
        except Exception as e:
            return f"Error: {str(e)}"

    def summarize(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Groups chunks by section and generates a structured summary.
        """
        # 1. Group text by section
        section_text = {sec: "" for sec in self.sections_to_summarize}
        all_statutes = set()
        case_title = "Unknown Judgment"
        
        for chunk in chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            section = meta.get("section", "UNKNOWN")
            
            if section in section_text:
                section_text[section] += text + " "
            
            # Extract case title from the first chunk if possible
            if chunk.get("chunk_id") == 1:
                first_lines = text.split('\n')[:3]
                case_title = " ".join(first_lines).strip()
            
            # Collect statutes
            statutes = meta.get("statutes_mentioned", [])
            all_statutes.update(statutes)

        # 2. Summarize each section
        summary = {
            "case_title": case_title,
            "statutes_cited": sorted(list(all_statutes)),
            "sections": {}
        }
        
        prompts = {
            "FACTS": "Summarize the key facts, background, and history of this legal case in a concise paragraph (3-5 sentences).",
            "ISSUES": "List the main legal issues or questions for consideration framed by the court. Use bullet points.",
            "ARGUMENTS": "Briefly summarize the core arguments presented by the petitioner and respondent.",
            "REASONING": "Explain the court's legal reasoning and analysis that led to the final decision. Focus on the 'Why'.",
            "JUDGMENT": "State the final decision, order, and directions of the court clearly."
        }

        for section in self.sections_to_summarize:
            content = section_text[section].strip()
            if not content:
                summary["sections"][section] = "No content found for this section."
                continue
            
            # Limit content to avoid context window overflow
            # 2500 words is roughly 3500-4000 tokens
            words = content.split()
            if len(words) > 2500:
                content = " ".join(words[:2500]) + "..."
            
            prompt = f"Section: {section}\n\nContent:\n{content}\n\nTask: {prompts[section]}"
            summary["sections"][section] = self._call_llm(prompt)

        return summary
