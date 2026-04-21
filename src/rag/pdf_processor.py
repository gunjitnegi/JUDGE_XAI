import os
import re
import json
import requests
import spacy
import fitz  # PyMuPDF
from typing import List, Dict, Any

class LegalPDFProcessor:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3.1:8b-instruct-q4_K_M"):
        self.ollama_url = ollama_url
        self.model = model
        
        # Load spacy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
            self.nlp.add_pipe('sentencizer')
        except OSError:
            import en_core_web_sm
            self.nlp = en_core_web_sm.load(disable=["ner", "tagger", "lemmatizer"])
            self.nlp.add_pipe('sentencizer')
            
        # Regex for common Indian statutes (IPC, BNS, CrPC, Section mentions, Articles)
        self.statute_pattern = re.compile(
            r'\b(IPC\s*\d+|BNS\s*\d+|CrPC\s*\d+|Section\s*\d+\s+of\s+[A-Za-z\s]+Act|Art(?:icle)?s?\.?\s*\d+(?:\(\d+\))?(?:\([a-z]\))?)\b', 
            re.IGNORECASE
        )
        
    def _clean_text(self, text: str) -> str:
        """Basic text cleanup."""
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract valid text blocks from PDF while ignoring headers/footers."""
        doc = fitz.open(pdf_path)
        valid_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            
            for b in blocks:
                # b is a tuple: (x0, y0, x1, y1, text, block_no, block_type)
                # block_type 0 is text
                if b[6] != 0:
                    continue
                    
                x0, y0, x1, y1, text = b[:5]
                text = self._clean_text(text)
                
                # Ignore empty blocks or obvious page numbers/headers
                if len(text) < 10:
                    continue
                    
                # Simple heuristic for headers/footers (adjust based on document)
                page_height = page.rect.height
                if y0 < 50 or y1 > (page_height - 50):
                    # It's likely a header or footer, but we'll include if it looks like real content
                    # We can refine this if needed, for now we aggressively filter very top/bottom
                    if len(text.split()) < 15:
                        continue
                
                valid_blocks.append({
                    "text": text,
                    "page_number": page_num + 1
                })
                
        doc.close()
        return valid_blocks

    def _chunk_blocks(self, blocks: List[Dict[str, Any]], max_words: int = 500) -> List[Dict[str, Any]]:
        """Group blocks into chunks using spacy sentences, never splitting mid-sentence."""
        chunks = []
        current_chunk_text = ""
        current_chunk_pages = set()
        current_word_count = 0
        
        for block in blocks:
            doc = self.nlp(block["text"])
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                    
                sent_words = len(sent_text.split())
                
                if current_word_count + sent_words > max_words and current_word_count > 0:
                    # Save current chunk
                    chunks.append({
                        "text": current_chunk_text.strip(),
                        "page_number": min(current_chunk_pages) if current_chunk_pages else block["page_number"]
                    })
                    # Start new chunk
                    current_chunk_text = sent_text + " "
                    current_chunk_pages = {block["page_number"]}
                    current_word_count = sent_words
                else:
                    current_chunk_text += sent_text + " "
                    current_chunk_pages.add(block["page_number"])
                    current_word_count += sent_words
                    
        # Add the last chunk
        if current_chunk_text.strip():
             chunks.append({
                "text": current_chunk_text.strip(),
                "page_number": min(current_chunk_pages) if current_chunk_pages else 1
            })
             
        return chunks

    def _classify_section_llm(self, text: str) -> str:
        """Use lightweight local LLM call to classify the section."""
        prompt = f"""You are a legal expert AI. Classify the following extract from an Indian legal judgment into ONE of these categories:
FACTS, ISSUES, ARGUMENTS, REASONING, JUDGMENT, UNKNOWN.

Reply with ONLY the single category word. Do not explain.

Extract:
{text[:1500]}
"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10}
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json().get("response", "").strip().upper()
                # Clean up response just in case the model is wordy
                for category in ["FACTS", "ISSUES", "ARGUMENTS", "REASONING", "JUDGMENT"]:
                    if category in result:
                        return category
            return "UNKNOWN"
        except Exception as e:
            print(f"LLM Classification failed: {e}")
            return "UNKNOWN"

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Main pipeline to process a PDF into enriched chunks."""
        print(f"Extracting blocks from {pdf_path}...")
        blocks = self._extract_blocks(pdf_path)
        
        print("Chunking text using Spacy...")
        chunks = self._chunk_blocks(blocks, max_words=400)
        
        enriched_chunks = []
        
        print(f"Enriching {len(chunks)} chunks with LLM & Regex...")
        for i, chunk in enumerate(chunks):
            # Extract statutes
            statutes = list(set(self.statute_pattern.findall(chunk["text"])))
            
            # Classify section
            section = self._classify_section_llm(chunk["text"])
            
            enriched_chunks.append({
                "chunk_id": i + 1,
                "text": chunk["text"],
                "metadata": {
                    "page_number": chunk["page_number"],
                    "section": section,
                    "statutes_mentioned": statutes,
                    "word_count": len(chunk["text"].split())
                }
            })
            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/{len(chunks)} chunks...")
                
        return enriched_chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = "c:/final_year/JUDGEXAI/test_pdf/civil_001_VINISHMA TECHNOLOGIES PVT_ LTD_versusSTATE OF CHHATTISGARH _ ANR_-_2025_ 10 S_C_.pdf"
        
    if os.path.exists(pdf_file):
        processor = LegalPDFProcessor()
        chunks = processor.process_pdf(pdf_file)
        
        # Save output for inspection
        output_file = "processed_chunks_test.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
        print(f"Successfully processed {len(chunks)} chunks and saved to {output_file}")
    else:
        print(f"File not found: {pdf_file}")
