import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Shared FDA-aligned ingredient line for MoviPrep / Plenvu (same combination products)
_GENERIC_ASC_PEG_SULFATE = (
    "ascorbic acid; polyethylene glycol 3350; potassium chloride; "
    "sodium ascorbate; sodium chloride; sodium sulfate"
)


class BowelPrepDrugDownloader:
    """Download bowel prep drug information from DailyMed and OpenFDA APIs"""

    def __init__(
        self,
        output_dir: str = "patient_kb/drug_labels",
        openfda_api_key: Optional[str] = None,
        clean_before_download: bool = False,
        max_dailymed_spl_results: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openfda_api_key = openfda_api_key or os.environ.get("OPENFDA_API_KEY", "") or None
        self.clean_before_download = clean_before_download
        # One SPL per drug avoids conflicting label versions in RAG (duplicate chunks).
        self.max_dailymed_spl_results = max(1, max_dailymed_spl_results)

        # Create subdirectories
        (self.output_dir / "dailymed_xml").mkdir(exist_ok=True)
        (self.output_dir / "dailymed_json").mkdir(exist_ok=True)
        (self.output_dir / "openfda_json").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)

        # Search keys (DailyMed/OpenFDA brand-style names) -> metadata.
        # Optional "openfda_components": explicit generic-name search terms for OpenFDA
        # when you do not want the heuristic from "generic".
        # Optional "openfda_application_number": NDA/ANDA/BLA string (e.g. NDA022372); when
        # set, OpenFDA is queried by application number first (avoids ambiguous brand hits).
        self.prep_agents = {
            "MoviPrep": {
                "generic": _GENERIC_ASC_PEG_SULFATE,
            },
            "SUPREP": {
                "generic": "magnesium sulfate; potassium sulfate; sodium sulfate",
                "openfda_application_number": "NDA022372",
            },
            "GoLYTELY": {
                "generic": (
                    "polyethylene glycol 3350; potassium chloride; sodium bicarbonate; "
                    "sodium chloride; sodium sulfate anhydrous"
                ),
            },
            "PEG-3350": {
                "generic": "polyethylene glycol 3350",
                "labeled_for_colonoscopy_prep": False,
                "rag_product_note": (
                    "Generic polyethylene glycol 3350 powder; labeling often targets "
                    "constipation/osmotic laxation, not necessarily colonoscopy prep."
                ),
            },
            "NuLYTELY": {
                "generic": (
                    "polyethylene glycol 3350; potassium chloride; "
                    "sodium bicarbonate; sodium chloride"
                ),
                "openfda_application_number": "NDA019797",
            },
            "Plenvu": {
                "generic": _GENERIC_ASC_PEG_SULFATE,
            },
            "Clenpiq": {
                "generic": "citric acid; magnesium oxide; sodium picosulfate",
            },
        }

    @staticmethod
    def _rag_metadata_from_agent_info(info: Dict[str, Any]) -> Dict[str, Any]:
        """Metadata for RAG: prep indication, notes, application id."""
        meta: Dict[str, Any] = {
            "labeled_for_colonoscopy_prep": info.get("labeled_for_colonoscopy_prep", True),
        }
        if info.get("rag_product_note"):
            meta["product_note"] = info["rag_product_note"]
        if info.get("openfda_application_number"):
            meta["openfda_application_number"] = info["openfda_application_number"]
        return meta

    def _rag_metadata_for_drug_name(self, drug_name: str) -> Dict[str, Any]:
        info = self.prep_agents.get(drug_name)
        if not info:
            return {"labeled_for_colonoscopy_prep": True}
        return self._rag_metadata_from_agent_info(info)

    def _enrich_label_doc_for_rag(self, doc: Dict[str, Any], drug_name: str) -> None:
        doc["rag_metadata"] = self._rag_metadata_for_drug_name(drug_name)

    def _clear_run_artifacts(self) -> None:
        """Remove prior JSON/XML outputs so a new run does not mix with old drugs."""
        for sub in ("dailymed_xml", "dailymed_json", "openfda_json"):
            d = self.output_dir / sub
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()
        consolidated = self.output_dir / "processed" / "consolidated_drug_labels.json"
        if consolidated.is_file():
            consolidated.unlink()

    def _request_get(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> requests.Response:
        """GET with retry on 429 / 503."""
        backoff = 1.0
        max_attempts = 6
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                if r.status_code in (429, 503) and attempt < max_attempts - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                return r
            except requests.RequestException as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
        if last_exc:
            raise last_exc
        raise RuntimeError("request failed without exception")

    @staticmethod
    def _escape_openfda_term(term: str) -> str:
        """Escape backslashes and double-quotes for use inside a Lucene quoted phrase."""
        return term.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _normalize_application_number(raw: str) -> str:
        """Normalize to OpenFDA-style application id (e.g. NDA022372)."""
        s = raw.strip().upper().replace(" ", "")
        if s.startswith("NDA") or s.startswith("ANDA") or s.startswith("BLA"):
            return s
        digits = "".join(c for c in s if c.isdigit())
        if not digits:
            return s
        # Default to NDA if no prefix (matches common human drug NDAs)
        return f"NDA{digits}"

    @staticmethod
    def _stable_openfda_result_id(result: Dict) -> str:
        for key in ("id", "set_id", "effective_time"):
            v = result.get(key)
            if v is not None and v != "":
                return str(v)
        payload = json.dumps(result, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def search_dailymed(self, drug_name: str) -> List[str]:
        """Search DailyMed for drug and return SET IDs"""
        print(f"\nSearching DailyMed for: {drug_name}")

        search_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        params = {"drug_name": drug_name}

        try:
            response = self._request_get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            set_ids = []
            if "data" in data:
                for item in data["data"]:
                    set_id = item.get("setid")
                    title = item.get("title", "")
                    if set_id:
                        set_ids.append(set_id)
                        print(f"  Found: {title} (SET ID: {set_id})")

            return set_ids

        except Exception as e:
            print(f"  Error searching DailyMed: {e}")
            return []

    def download_dailymed_label(self, drug_name: str, set_id: str) -> Tuple[Dict, Optional[str]]:
        """Download drug label from DailyMed using SET ID. Returns (label_data, error_message)."""
        print(f"  Downloading DailyMed label for SET ID: {set_id}")

        xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"

        try:
            response = self._request_get(xml_url, timeout=30)
            response.raise_for_status()

            xml_file = self.output_dir / "dailymed_xml" / f"{drug_name}_{set_id}.xml"
            xml_file.write_bytes(response.content)
            print(f"    Saved XML: {xml_file}")

            root = ET.fromstring(response.content)
            label_data = self.parse_dailymed_xml(root, drug_name, set_id)

            json_file = self.output_dir / "dailymed_json" / f"{drug_name}_{set_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)
            print(f"    Saved JSON: {json_file}")

            return label_data, None

        except Exception as e:
            err = str(e)
            print(f"    Error downloading label: {e}")
            return {}, err

    def parse_dailymed_xml(self, root: ET.Element, drug_name: str, set_id: str) -> Dict:
        """Parse DailyMed XML and extract structured information"""

        ns = {"hl7": "urn:hl7-org:v3"}

        label_data = {
            "drug_name": drug_name,
            "set_id": set_id,
            "source": "DailyMed",
            "sections": {},
        }

        section_codes = {
            "34067-9": "indications_and_usage",
            "34068-7": "dosage_and_administration",
            "34070-3": "contraindications",
            "34071-1": "warnings_and_precautions",
            "34073-7": "drug_interactions",
            "34084-4": "adverse_reactions",
            "34076-0": "patient_counseling_information",
            "43685-7": "warnings",
            "42229-5": "special_populations",
        }

        for section in root.findall(".//hl7:section", ns):
            code_elem = section.find(".//hl7:code", ns)
            if code_elem is None:
                continue
            code = code_elem.get("code")
            if code not in section_codes:
                continue

            title_elem = section.find(".//hl7:title", ns)
            text_elems = section.findall(".//hl7:text", ns)
            if not text_elems:
                continue

            section_name = section_codes[code]
            chunks = []
            for text_elem in text_elems:
                chunks.append(self.extract_text_from_element(text_elem))
            text_content = "\n\n".join(c for c in chunks if c and c.strip())

            if not text_content.strip():
                continue

            if section_name in label_data["sections"]:
                prev = label_data["sections"][section_name]["content"]
                text_content = f"{prev}\n\n{text_content}"

            label_data["sections"][section_name] = {
                "title": title_elem.text if title_elem is not None else section_name,
                "content": text_content,
            }

        return label_data

    def extract_text_from_element(self, element: ET.Element) -> str:
        """Recursively extract text from XML element"""
        text_parts = []

        if element.text:
            text_parts.append(element.text.strip())

        for child in element:
            child_text = self.extract_text_from_element(child)
            if child_text:
                text_parts.append(child_text)
            if child.tail:
                text_parts.append(child.tail.strip())

        return " ".join(filter(None, text_parts))

    def search_openfda(
        self,
        drug_name: str,
        generic_name: Optional[str],
        *,
        openfda_components: Optional[List[str]] = None,
        application_number: Optional[str] = None,
        skip_generic_if_brand_hits: bool = True,
    ) -> List[Dict]:
        """Search OpenFDA for drug label information."""
        print(f"\nSearching OpenFDA for: {drug_name}")

        base_url = "https://api.fda.gov/drug/label.json"
        safe_brand = self._escape_openfda_term(drug_name)
        brand_query = f'openfda.brand_name:"{safe_brand}"'

        all_results: List[Dict] = []
        seen_ids: set = set()

        def run_query(search: str) -> Optional[Dict]:
            params: Dict[str, Any] = {"search": search, "limit": 5}
            if self.openfda_api_key:
                params["api_key"] = self.openfda_api_key
            response = self._request_get(base_url, params=params, timeout=30)
            if response.status_code == 404:
                print(f"  No results found for query: {search}")
                return None
            response.raise_for_status()
            return response.json()

        # 0) Application number (NDA/ANDA/BLA) — authoritative; avoids wrong brand/generic hits
        if application_number:
            nda = self._normalize_application_number(application_number)
            app_query = f'openfda.application_number:"{self._escape_openfda_term(nda)}"'
            print(f"  Trying application number: {nda}")
            try:
                data = run_query(app_query)
                if data and data.get("results"):
                    print(
                        f"  Found {len(data['results'])} result(s) for application {nda}"
                    )
                    for result in data["results"]:
                        rid = self._stable_openfda_result_id(result)
                        if rid not in seen_ids:
                            all_results.append(result)
                            seen_ids.add(rid)
                    time.sleep(0.5)
                    return all_results
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code != 404:
                    print(f"  Error searching OpenFDA (application): {e}")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.5)
            print("  No label for application number; falling back to brand/generic search")

        # 1) Brand search
        try:
            data = run_query(brand_query)
            if data and "results" in data:
                print(f"  Found {len(data['results'])} results for brand query")
                for result in data["results"]:
                    rid = self._stable_openfda_result_id(result)
                    if rid not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(rid)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code != 404:
                print(f"  Error searching OpenFDA (brand): {e}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(0.5)

        if skip_generic_if_brand_hits and all_results:
            return all_results

        # 2) Generic component queries
        if openfda_components:
            components = openfda_components[:3]
        elif generic_name:
            components = self.extract_key_components(generic_name)
        else:
            components = []

        for component in components:
            safe_c = self._escape_openfda_term(component)
            q = f'openfda.generic_name:"{safe_c}"'
            try:
                data = run_query(q)
                if data and "results" in data:
                    print(f"  Found {len(data['results'])} results for query: {q}")
                    for result in data["results"]:
                        rid = self._stable_openfda_result_id(result)
                        if rid not in seen_ids:
                            all_results.append(result)
                            seen_ids.add(rid)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code != 404:
                    print(f"  Error searching OpenFDA: {e}")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(0.5)

        return all_results

    def extract_key_components(self, generic_name: str) -> List[str]:
        """Extract searchable components from generic name (longest / most specific first)."""
        key_terms = [
            "polyethylene glycol 3350",
            "PEG-3350",
            "sodium picosulfate",
            "sodium ascorbate",
            "ascorbic acid",
            "magnesium sulfate",
            "potassium sulfate",
            "sodium sulfate",
            "sodium bicarbonate",
            "potassium chloride",
            "magnesium oxide",
            "citric acid",
            "sodium phosphate",
        ]

        generic_lower = generic_name.lower()
        matched: List[str] = []
        for term in key_terms:
            if term.lower() in generic_lower:
                matched.append(term)

        matched.sort(key=len, reverse=True)
        return matched[:2]

    def process_openfda_results(self, results: List[Dict], drug_name: str) -> Dict:
        """Process OpenFDA results and extract relevant sections"""

        if not results:
            return {}

        result = results[0]

        processed_data = {
            "drug_name": drug_name,
            "source": "OpenFDA",
            "sections": {},
        }

        section_mapping = {
            "indications_and_usage": "indications_and_usage",
            "dosage_and_administration": "dosage_and_administration",
            "contraindications": "contraindications",
            "warnings_and_cautions": "warnings_and_precautions",
            "warnings": "warnings",
            "precautions": "precautions",
            "adverse_reactions": "adverse_reactions",
            "drug_interactions": "drug_interactions",
            "use_in_specific_populations": "special_populations",
            "patient_counseling_information": "patient_counseling_information",
            "information_for_patients": "patient_information",
        }

        for openfda_key, our_key in section_mapping.items():
            if openfda_key in result:
                content = result[openfda_key]
                if isinstance(content, list):
                    content = " ".join(content)

                processed_data["sections"][our_key] = {
                    "title": our_key.replace("_", " ").title(),
                    "content": content,
                }

        if "openfda" in result:
            processed_data["brand_names"] = result["openfda"].get("brand_name", [])
            processed_data["generic_names"] = result["openfda"].get("generic_name", [])

        return processed_data

    def _write_failures(self, entries: List[Dict]) -> None:
        path = self.output_dir / "failures.json"
        existing: List[Dict] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
            except json.JSONDecodeError:
                pass

        seen = set()
        merged: List[Dict] = []
        for d in existing + entries:
            key = (d.get("drug"), d.get("source"), d.get("set_id"), d.get("error"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        if entries:
            print(f"  Updated {path} ({len(entries)} new failure record(s) this run)")

    def download_all_drugs(self) -> List[Dict]:
        """Download information for all bowel prep agents"""

        if self.clean_before_download:
            self._clear_run_artifacts()

        print("=" * 70)
        print("Starting download of bowel prep drug labels")
        print("=" * 70)

        all_drugs_data: List[Dict] = []
        run_failures: List[Dict] = []

        for drug_name, info in self.prep_agents.items():
            print(f"\n{'=' * 70}")
            print(f"Processing: {drug_name}")
            print(f"Generic: {info['generic']}")
            print(f"{'=' * 70}")

            warnings: List[str] = []
            drug_data: Dict[str, Any] = {
                "brand_name": drug_name,
                "generic_name": info["generic"],
                "openfda_application_number": info.get("openfda_application_number"),
                "rag_metadata": self._rag_metadata_from_agent_info(info),
                "dailymed_data": [],
                "openfda_data": {},
                "dailymed_ok": False,
                "openfda_ok": False,
                "warnings": warnings,
            }

            set_ids = self.search_dailymed(drug_name)
            if not set_ids:
                warnings.append("DailyMed search returned no SPL matches")
                run_failures.append(
                    {
                        "drug": drug_name,
                        "source": "DailyMed_search",
                        "error": "No SPL matches",
                    }
                )

            for set_id in set_ids[: self.max_dailymed_spl_results]:
                label_data, err = self.download_dailymed_label(drug_name, set_id)
                if label_data:
                    drug_data["dailymed_data"].append(label_data)
                elif err:
                    warnings.append(f"DailyMed download failed for set_id {set_id}: {err}")
                    run_failures.append(
                        {
                            "drug": drug_name,
                            "source": "DailyMed_download",
                            "set_id": set_id,
                            "error": err,
                        }
                    )
                time.sleep(1)

            drug_data["dailymed_ok"] = len(drug_data["dailymed_data"]) > 0

            openfda_components = info.get("openfda_components")
            openfda_results = self.search_openfda(
                drug_name,
                info["generic"],
                openfda_components=openfda_components,
                application_number=info.get("openfda_application_number"),
            )
            if openfda_results:
                processed_openfda = self.process_openfda_results(openfda_results, drug_name)
                drug_data["openfda_data"] = processed_openfda
                drug_data["openfda_ok"] = True

                openfda_file = self.output_dir / "openfda_json" / f"{drug_name}_openfda.json"
                with open(openfda_file, "w", encoding="utf-8") as f:
                    json.dump(processed_openfda, f, indent=2, ensure_ascii=False)
                print(f"  Saved OpenFDA data: {openfda_file}")
            else:
                warnings.append("OpenFDA returned no label results")
                drug_data["openfda_ok"] = False
                run_failures.append(
                    {
                        "drug": drug_name,
                        "source": "OpenFDA",
                        "error": "No label results",
                    }
                )

            time.sleep(1)
            all_drugs_data.append(drug_data)

        self._write_failures(run_failures)

        summary_file = self.output_dir / "all_drugs_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_drugs_data, f, indent=2, ensure_ascii=False)

        all_ok = all(
            d.get("dailymed_ok") and d.get("openfda_ok") for d in all_drugs_data
        )
        print(f"\n{'=' * 70}")
        if all_ok:
            print("Run finished: all drugs have DailyMed and OpenFDA data.")
        else:
            print(
                "Run finished: some drugs are missing DailyMed or OpenFDA data "
                "(see warnings per drug and failures.json)."
            )
        print(f"Summary saved to: {summary_file}")
        print(f"{'=' * 70}")

        return all_drugs_data

    def create_consolidated_dataset(self) -> List[Dict]:
        """Load JSON outputs only for drugs in prep_agents (avoids stale products from old runs)."""

        print("\nCreating consolidated dataset...")

        consolidated: List[Dict] = []
        dailymed_dir = self.output_dir / "dailymed_json"
        openfda_dir = self.output_dir / "openfda_json"

        # prep_agents order: DailyMed JSONs per drug, then OpenFDA (stable grouping for RAG).
        for drug_name in self.prep_agents:
            prefix = f"{drug_name}_"
            dailymed_paths = sorted(
                p
                for p in dailymed_dir.iterdir()
                if p.is_file() and p.suffix == ".json" and p.name.startswith(prefix)
            )
            # Stale runs may leave multiple SPL JSONs; keep one file per drug (newest on disk).
            if len(dailymed_paths) > 1:
                dailymed_paths = [max(dailymed_paths, key=lambda p: p.stat().st_mtime)]

            for json_file in dailymed_paths:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                self._enrich_label_doc_for_rag(doc, drug_name)
                consolidated.append(doc)

            openfda_file = openfda_dir / f"{drug_name}_openfda.json"
            if openfda_file.is_file():
                with open(openfda_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                self._enrich_label_doc_for_rag(doc, drug_name)
                consolidated.append(doc)

        consolidated_file = self.output_dir / "processed" / "consolidated_drug_labels.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        print(f"Consolidated dataset saved to: {consolidated_file}")
        print(f"Total entries: {len(consolidated)}")

        return consolidated


if __name__ == "__main__":
    downloader = BowelPrepDrugDownloader(
        output_dir="src/data_processing/patient_kb/drug_labels",
        clean_before_download=False,
    )

    all_data = downloader.download_all_drugs()
    consolidated = downloader.create_consolidated_dataset()

    print("\n" + "=" * 70)
    print("Download complete!")
    print("Check the 'src/data_processing/patient_kb/drug_labels' directory for all files")
    print("=" * 70)
