"""
StatutesManager — Knowledge base for Indian legal statutes.
Provides O(1) lookup for IPC, BNS, and Constitutional Article definitions,
punishments, and IPC↔BNS cross-mapping.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any


class StatutesManager:
    """
    Loads statutes.jsonl + ipc_bns_mapping.json and provides
    fast lookups for the RAG pipeline and UI.
    """

    def __init__(
        self,
        jsonl_path: str = None,
        mapping_path: str = None,
    ):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        if jsonl_path is None:
            jsonl_path = os.path.join(
                project_root, "data", "raw", "statutes", "statutes.jsonl"
            )
        if mapping_path is None:
            mapping_path = os.path.join(
                project_root, "data", "raw", "statutes", "ipc_bns_mapping.json"
            )

        # Main lookup: normalized key → entry dict
        self.statutes: Dict[str, Dict[str, Any]] = {}

        # IPC → BNS mapping
        self.ipc_to_bns: Dict[str, str] = {}
        self.bns_to_ipc: Dict[str, str] = {}

        self._load_statutes(jsonl_path)
        self._load_mapping(mapping_path)

        print(
            f"[StatutesManager] Loaded {len(self.statutes)} statutes, "
            f"{len(self.ipc_to_bns)} IPC->BNS mappings."
        )

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def _load_statutes(self, path: str) -> None:
        """Load statutes.jsonl into self.statutes dict."""
        if not os.path.exists(path):
            print(f"[StatutesManager] WARNING: {path} not found.")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = self._normalize_key(entry["law"], entry["section"])
                # Keep first occurrence (dedup)
                if key not in self.statutes:
                    self.statutes[key] = entry

    def _load_mapping(self, path: str) -> None:
        """Load IPC→BNS mapping JSON."""
        if not os.path.exists(path):
            print(f"[StatutesManager] WARNING: {path} not found.")
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for ipc_key, bns_key in raw.items():
            ipc_norm = self._normalize_key("IPC", ipc_key)
            bns_norm = self._normalize_key("BNS", bns_key)
            self.ipc_to_bns[ipc_norm] = bns_norm
            self.bns_to_ipc[bns_norm] = ipc_norm

    # ------------------------------------------------------------------ #
    #  Normalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_key(law: str, section: str) -> str:
        """
        Normalize a statute reference to a canonical key.
        Examples:
            ("IPC", "IPC_302")    → "IPC_302"
            ("IPC", "302")        → "IPC_302"
            ("BNS", "103")        → "BNS_103"
            ("CONSTITUTION", "Article_14") → "ARTICLE_14"
            ("", "Art. 14")       → "ARTICLE_14"
        """
        law = law.upper().strip()
        section = section.strip()

        # Handle "Article 14", "Art.14", "Art 14", "Article_14"
        art_match = re.match(
            r"(?:Art(?:icle)?s?\.?\s*)(\d+[A-Za-z]*)", section, re.IGNORECASE
        )
        if art_match or law == "CONSTITUTION":
            if art_match:
                num = art_match.group(1).upper()
            else:
                # section might be "Article_14" or "14"
                num = re.sub(r"[^0-9A-Za-z]", "", section.replace("Article", "")).upper()
            return f"ARTICLE_{num}"

        # Handle "IPC 302", "IPC_302", "Section 302 of IPC", etc.
        # Strip the law prefix from the section if it's repeated
        section_clean = re.sub(
            r"^(IPC|BNS|CrPC|BNSS)[_\s]*", "", section, flags=re.IGNORECASE
        ).strip()

        if not section_clean:
            section_clean = section

        return f"{law}_{section_clean}".upper()

    def _parse_ref(self, ref: str) -> str:
        """
        Parse a free-form statute reference string into a normalized key.
        Examples:
            "IPC 302"         → "IPC_302"
            "Section 302 IPC" → "IPC_302"
            "IPC_302"         → "IPC_302"
            "Article 14"      → "ARTICLE_14"
            "Art.21"          → "ARTICLE_21"
            "BNS 103"         → "BNS_103"
        """
        ref = ref.strip()

        # Article pattern
        art_match = re.match(
            r"(?:Art(?:icle)?s?\.?\s*)(\d+[A-Za-z]*)", ref, re.IGNORECASE
        )
        if art_match:
            return f"ARTICLE_{art_match.group(1).upper()}"

        # "Section X of Y Act" pattern — extract number and try IPC/BNS
        sec_match = re.match(
            r"Section\s+(\d+[A-Za-z]*)\s+(?:of\s+)?(?:the\s+)?(IPC|BNS|CrPC|Indian Penal Code|Bharatiya Nyaya Sanhita)",
            ref, re.IGNORECASE,
        )
        if sec_match:
            num = sec_match.group(1).upper()
            law_raw = sec_match.group(2).upper()
            law = "IPC" if "IPC" in law_raw or "PENAL" in law_raw else law_raw
            law = "BNS" if "BNS" in law_raw or "NYAYA" in law_raw else law
            return f"{law}_{num}"

        # "IPC 302" / "IPC_302" / "BNS 103"
        direct_match = re.match(
            r"(IPC|BNS|CrPC|BNSS)[_\s]+(\d+[A-Za-z]*)", ref, re.IGNORECASE
        )
        if direct_match:
            return f"{direct_match.group(1).upper()}_{direct_match.group(2).upper()}"

        # Fallback: uppercase the whole thing and replace spaces with _
        return re.sub(r"\s+", "_", ref.upper())

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def lookup(self, ref: str) -> Optional[Dict[str, Any]]:
        """
        Look up a statute by free-form reference.
        Returns the full entry dict or None.
        """
        key = self._parse_ref(ref)
        return self.statutes.get(key)

    def get_text(self, ref: str) -> Optional[str]:
        """Get the full text description of a statute."""
        entry = self.lookup(ref)
        return entry["text"] if entry else None

    def get_title(self, ref: str) -> Optional[str]:
        """Get the title of a statute."""
        entry = self.lookup(ref)
        return entry.get("title", "") if entry else None

    def get_punishment(self, ref: str) -> Optional[str]:
        """
        Extract punishment clause from the statute text.
        Looks for "shall be punished with..." patterns.
        """
        entry = self.lookup(ref)
        if not entry:
            return None

        text = entry["text"]

        # Try to find punishment phrases
        patterns = [
            r"shall be punished with (.+?)(?:\.|$)",
            r"punishable with (.+?)(?:\.|$)",
            r"punishment[:\s]+(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                punishment = match.group(1).strip()
                # Truncate if too long
                if len(punishment) > 300:
                    punishment = punishment[:300] + "..."
                return punishment

        return None

    def get_simple_explanation(self, ref: str) -> Optional[str]:
        """Get the 'In Simple Words' explanation if available."""
        entry = self.lookup(ref)
        if not entry:
            return None

        text = entry["text"]
        # Look for the "in Simple Words" section
        simple_match = re.search(
            r"(?:in Simple Words|In Simple Words)\s*\n(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if simple_match:
            return simple_match.group(1).strip()

        return None

    def get_bns_equivalent(self, ipc_ref: str) -> Optional[Dict[str, Any]]:
        """
        Given an IPC reference, return the BNS equivalent entry.
        """
        ipc_key = self._parse_ref(ipc_ref)
        bns_key = self.ipc_to_bns.get(ipc_key)
        if bns_key:
            return self.statutes.get(bns_key)
        return None

    def get_ipc_equivalent(self, bns_ref: str) -> Optional[Dict[str, Any]]:
        """
        Given a BNS reference, return the IPC equivalent entry.
        """
        bns_key = self._parse_ref(bns_ref)
        ipc_key = self.bns_to_ipc.get(bns_key)
        if ipc_key:
            return self.statutes.get(ipc_key)
        return None

    def enrich_context(self, statute_refs: List[str]) -> str:
        """
        Given a list of statute references found in retrieved chunks,
        build a formatted reference table for the LLM prompt.

        Returns a string like:
        STATUTE REFERENCE TABLE:
        - IPC 302: Murder — Punishment: death or life imprisonment + fine
          (BNS equivalent: BNS 103)
        - Article 14: Right to Equality — Constitutional guarantee
        """
        if not statute_refs:
            return ""

        seen = set()
        lines = []

        for ref in statute_refs:
            key = self._parse_ref(ref)
            if key in seen:
                continue
            seen.add(key)

            entry = self.statutes.get(key)
            if not entry:
                continue

            title = entry.get("title", "").strip()
            punishment = self.get_punishment(ref)
            simple = self.get_simple_explanation(ref)

            line = f"- {entry['law']} {entry['section']}"
            if title:
                line += f": {title}"
            if punishment:
                line += f" — Punishment: {punishment}"
            elif simple:
                line += f" — {simple[:200]}"

            # Add BNS equivalent for IPC sections
            if entry["law"] == "IPC":
                bns_key = self.ipc_to_bns.get(key)
                if bns_key:
                    bns_entry = self.statutes.get(bns_key)
                    if bns_entry:
                        line += f"\n  (BNS equivalent: BNS Section {bns_entry['section']})"

            lines.append(line)

        if not lines:
            return ""

        return "STATUTE REFERENCE TABLE:\n" + "\n".join(lines)

    def get_all_for_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Scan chunk metadata for statute references and return
        enriched statute details for the UI panel.

        Returns a list of dicts:
        [
            {
                "ref": "IPC 302",
                "key": "IPC_302",
                "law": "IPC",
                "title": "Murder",
                "punishment": "death or life imprisonment...",
                "simple_explanation": "...",
                "bns_equivalent": {"section": "103", ...} or None,
                "pages": [3, 5],
            }
        ]
        """
        statute_map: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            # Handle both direct chunks and retriever results
            if "chunk" in chunk:
                meta = chunk["chunk"].get("metadata", {})
                page = meta.get("page_number", "?")
            else:
                page = meta.get("page_number", "?")

            refs = meta.get("statutes_mentioned", [])
            for ref in refs:
                key = self._parse_ref(ref)
                if key not in statute_map:
                    entry = self.statutes.get(key)
                    if entry:
                        bns_equiv = None
                        ipc_equiv = None
                        if entry["law"] == "IPC":
                            bns_key = self.ipc_to_bns.get(key)
                            if bns_key:
                                bns_equiv = self.statutes.get(bns_key)
                        elif entry["law"] == "BNS":
                            ipc_key = self.bns_to_ipc.get(key)
                            if ipc_key:
                                ipc_equiv = self.statutes.get(ipc_key)

                        statute_map[key] = {
                            "ref": ref,
                            "key": key,
                            "law": entry["law"],
                            "section": entry["section"],
                            "title": entry.get("title", ""),
                            "punishment": self.get_punishment(ref),
                            "simple_explanation": self.get_simple_explanation(ref),
                            "bns_equivalent": bns_equiv,
                            "ipc_equivalent": ipc_equiv,
                            "pages": [],
                        }
                    else:
                        # Unknown statute — still track it
                        statute_map[key] = {
                            "ref": ref,
                            "key": key,
                            "law": "UNKNOWN",
                            "section": ref,
                            "title": "",
                            "punishment": None,
                            "simple_explanation": None,
                            "bns_equivalent": None,
                            "ipc_equivalent": None,
                            "pages": [],
                        }

                if page not in statute_map[key]["pages"]:
                    statute_map[key]["pages"].append(page)

        return list(statute_map.values())


# ------------------------------------------------------------------ #
#  Quick test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sm = StatutesManager()

    print("\n--- Lookup Tests ---")
    for ref in ["IPC 302", "IPC 420", "Article 14", "Art.21", "BNS 103", "IPC 376"]:
        entry = sm.lookup(ref)
        if entry:
            print(f"\n{ref} → {entry['law']} {entry['section']}")
            title = sm.get_title(ref)
            if title:
                print(f"  Title: {title}")
            punishment = sm.get_punishment(ref)
            if punishment:
                print(f"  Punishment: {punishment[:150]}")
            simple = sm.get_simple_explanation(ref)
            if simple:
                print(f"  Simple: {simple[:150]}")
            bns = sm.get_bns_equivalent(ref)
            if bns:
                print(f"  BNS equivalent: BNS {bns['section']}")
        else:
            print(f"\n{ref} → NOT FOUND")

    print("\n--- Context Enrichment Test ---")
    ctx = sm.enrich_context(["IPC 302", "Article 14", "IPC 420", "Art.21"])
    print(ctx)
