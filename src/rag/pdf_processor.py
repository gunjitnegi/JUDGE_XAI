import os
import re
import json
import requests
import spacy
import fitz  # PyMuPDF
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class LegalPDFProcessor:
    def __init__(self, ollama_url: str = None, model: str = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.model = model or os.getenv("MODEL_NAME", "llama3.1:8b-instruct-q4_K_M")
        
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
        """Extract valid text blocks from PDF while ignoring headers/footers. Includes OCR fallback."""
        doc = fitz.open(pdf_path)
        valid_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("blocks")
            page_text = page.get_text("text").strip()
            
            # --- Fix D: Missing OCR Fallback ---
            if len(page_text) < 50:
                print(f"Warning: Page {page_num + 1} appears to be scanned or empty. Attempting OCR fallback...")
                try:
                    from pdf2image import convert_from_path
                    import pytesseract
                    
                    # Convert the specific page to an image (page_num is 0-indexed, pdf2image uses 1-indexed for some args but we can extract specific page)
                    # Note: convert_from_path takes first_page and last_page (1-indexed)
                    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0]).strip()
                        if ocr_text:
                            # Create a fake block spanning the page
                            valid_blocks.append({
                                "text": self._clean_text(ocr_text),
                                "page_number": page_num + 1
                            })
                            continue
                except ImportError:
                    print("Error: pdf2image or pytesseract not installed. Please run `pip install pytesseract pdf2image`. Skipping OCR.")
                except Exception as e:
                    print(f"Error during OCR on page {page_num + 1}: {e}. Ensure Tesseract-OCR is installed on Windows.")
            
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

    def _chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge blocks that don't end in sentence boundaries, apply length thresholds."""
        merged_blocks = []
        current_text = ""
        current_page = 1
        
        for block in blocks:
            text = block["text"]
            if not current_text:
                current_text = text
                current_page = block["page_number"]
            else:
                # If current_text does not end with sentence boundary, merge with next
                if not re.search(r'[.?!।"”]\s*$', current_text):
                    current_text += " " + text
                else:
                    merged_blocks.append({
                        "text": current_text,
                        "page_number": current_page
                    })
                    current_text = text
                    current_page = block["page_number"]
        if current_text:
            merged_blocks.append({
                "text": current_text,
                "page_number": current_page
            })
            
        chunks = []
        for block in merged_blocks:
            text = block["text"].strip()
            page = block["page_number"]
            
            # Min length threshold
            if len(text) < 40:
                continue
                
            # Max length threshold
            if len(text) > 1200:
                sub_chunks = self._split_large_text(text, max_chars=1200)
                for sc in sub_chunks:
                    if len(sc) >= 40:
                        chunks.append({"text": sc, "page_number": page})
            else:
                chunks.append({"text": text, "page_number": page})
                
        return chunks

    def _split_large_text(self, text: str, max_chars: int) -> List[str]:
        """Split text exceeding max_chars at nearest sentence boundary (. followed by [A-Z])."""
        if len(text) <= max_chars:
            return [text]
            
        boundary_pattern = re.compile(r'\.\s+(?=[A-Z])')
        sub_chunks = []
        
        while len(text) > max_chars:
            matches = list(boundary_pattern.finditer(text[:max_chars]))
            if matches:
                # Split at last match within limit
                last_match = matches[-1]
                split_point = last_match.start() + 1
                sub_chunks.append(text[:split_point].strip())
                text = text[split_point:].strip()
            else:
                # No boundary found inside limit, find first one outside
                match = boundary_pattern.search(text)
                if match:
                    split_point = match.start() + 1
                    sub_chunks.append(text[:split_point].strip())
                    text = text[split_point:].strip()
                else:
                    break
                    
        if text:
            sub_chunks.append(text.strip())
            
        return sub_chunks

    def _classify_section_llm(self, text: str) -> str:
        """Use lightweight zero-shot classifier to classify the section."""
        try:
            # We import here to prevent circular imports and keep startup fast
            from src.preprocessing.role_labelling import classify_single
            
            # classify_single returns (role, source, confidence)
            role, source, conf = classify_single(text, position="middle")
            
            if role:
                role_upper = role.upper()
                if role_upper == "FINAL_DECISION":
                    return "JUDGMENT"
                return role_upper
            return "UNKNOWN"
        except Exception as e:
            print(f"LLM Classification failed: {e}")
            return "UNKNOWN"

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Main pipeline to process a PDF into enriched chunks."""
        case_id = os.path.basename(pdf_path)
        print(f"Extracting blocks from {pdf_path}...")
        blocks = self._extract_blocks(pdf_path)
        
        print("Chunking text (merging incomplete sentences)...")
        chunks = self._chunk_blocks(blocks)
        
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
                    "case_id": case_id,
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
