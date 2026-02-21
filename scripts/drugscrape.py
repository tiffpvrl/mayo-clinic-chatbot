import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import time
from typing import List, Dict
import pandas as pd

class BowelPrepDrugDownloader:
    """Download bowel prep drug information from DailyMed and OpenFDA APIs"""
    
    def __init__(self, output_dir: str = "patient_kb/drug_labels"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "dailymed_xml").mkdir(exist_ok=True)
        (self.output_dir / "dailymed_json").mkdir(exist_ok=True)
        (self.output_dir / "openfda_json").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        
        # Bowel prep agents with brand names and generic names
        self.prep_agents = {
            "MoviPrep": {"generic": "PEG-3350, sodium sulfate, sodium chloride, potassium chloride"},
            "SUPREP": {"generic": "sodium sulfate, potassium sulfate, magnesium sulfate"},
            "GoLYTELY": {"generic": "PEG-3350 and electrolytes"},
            "CoLyte": {"generic": "PEG-3350 and electrolytes"},
            "Prepopik": {"generic": "sodium picosulfate, magnesium oxide, citric acid"},
            "Clenpiq": {"generic": "sodium picosulfate, magnesium oxide, citric acid"},
            "Plenvu": {"generic": "PEG-3350, sodium ascorbate, sodium sulfate"},
            "GaviLyte-C": {"generic": "PEG-3350 and electrolytes"},
            "NuLYTELY": {"generic": "PEG-3350 and electrolytes"},
            "TriLyte": {"generic": "PEG-3350 and electrolytes"},
            "OsmoPrep": {"generic": "sodium phosphate"}  # Limited use
        }
        
    def search_dailymed(self, drug_name: str) -> List[str]:
        """Search DailyMed for drug and return SET IDs"""
        print(f"\nSearching DailyMed for: {drug_name}")
        
        # DailyMed search API
        search_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        params = {"drug_name": drug_name}
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
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
    
    def download_dailymed_label(self, set_id: str, drug_name: str) -> Dict:
        """Download drug label from DailyMed using SET ID"""
        print(f"  Downloading DailyMed label for SET ID: {set_id}")
        
        # Get XML version for complete data
        xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
        
        try:
            response = requests.get(xml_url, timeout=10)
            response.raise_for_status()
            
            # Save raw XML
            xml_file = self.output_dir / "dailymed_xml" / f"{drug_name}_{set_id}.xml"
            xml_file.write_bytes(response.content)
            print(f"    Saved XML: {xml_file}")
            
            # Parse and extract key sections
            root = ET.fromstring(response.content)
            label_data = self.parse_dailymed_xml(root, drug_name, set_id)
            
            # Save as JSON
            json_file = self.output_dir / "dailymed_json" / f"{drug_name}_{set_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)
            print(f"    Saved JSON: {json_file}")
            
            return label_data
            
        except Exception as e:
            print(f"    Error downloading label: {e}")
            return {}
    
    def parse_dailymed_xml(self, root: ET.Element, drug_name: str, set_id: str) -> Dict:
        """Parse DailyMed XML and extract structured information"""
        
        # XML namespaces used in DailyMed
        ns = {'hl7': 'urn:hl7-org:v3'}
        
        label_data = {
            "drug_name": drug_name,
            "set_id": set_id,
            "source": "DailyMed",
            "sections": {}
        }
        
        # Key sections to extract
        section_codes = {
            "34067-9": "indications_and_usage",
            "34068-7": "dosage_and_administration",
            "34070-3": "contraindications",
            "34071-1": "warnings_and_precautions",
            "34073-7": "drug_interactions",
            "34084-4": "adverse_reactions",
            "34076-0": "patient_counseling_information",
            "43685-7": "warnings",
            "42229-5": "special_populations"
        }
        
        # Extract all sections
        for section in root.findall('.//hl7:section', ns):
            code_elem = section.find('.//hl7:code', ns)
            if code_elem is not None:
                code = code_elem.get('code')
                title_elem = section.find('.//hl7:title', ns)
                text_elem = section.find('.//hl7:text', ns)
                
                if code in section_codes and text_elem is not None:
                    section_name = section_codes[code]
                    # Get all text content
                    text_content = self.extract_text_from_element(text_elem)
                    label_data["sections"][section_name] = {
                        "title": title_elem.text if title_elem is not None else section_name,
                        "content": text_content
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
        
        return ' '.join(filter(None, text_parts))
    
    def search_openfda(self, drug_name: str, generic_name: str = None) -> List[Dict]:
        """Search OpenFDA for drug label information"""
        print(f"\nSearching OpenFDA for: {drug_name}")
        
        base_url = "https://api.fda.gov/drug/label.json"
        
        # Build search queries
        search_queries = [
            f'openfda.brand_name:"{drug_name}"',
        ]
        
        # If generic name provided, search for key components
        if generic_name:
            # Extract key components (e.g., "PEG-3350" from full generic name)
            key_components = self.extract_key_components(generic_name)
            for component in key_components:
                search_queries.append(f'openfda.generic_name:"{component}"')
        
        all_results = []
        seen_set_ids = set()  # Avoid duplicates
        
        for query in search_queries:
            params = {
                "search": query,
                "limit": 5
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if "results" in data:
                    print(f"  Found {len(data['results'])} results for query: {query}")
                    # Add only unique results
                    for result in data["results"]:
                        # Use set_id or a hash to identify uniqueness
                        result_id = result.get("set_id", str(hash(str(result))))
                        if result_id not in seen_set_ids:
                            all_results.append(result)
                            seen_set_ids.add(result_id)
                
                time.sleep(0.5)  # Rate limiting
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"  No results found for query: {query}")
                else:
                    print(f"  Error searching OpenFDA: {e}")
            except Exception as e:
                print(f"  Error: {e}")
        
        return all_results

    def extract_key_components(self, generic_name: str) -> List[str]:
        """Extract searchable components from generic name"""
        # Common bowel prep components
        components = []
        
        key_terms = [
            "PEG-3350",
            "polyethylene glycol 3350",
            "sodium sulfate",
            "sodium picosulfate",
            "magnesium oxide",
            "sodium phosphate"
        ]
        
        generic_lower = generic_name.lower()
        
        for term in key_terms:
            if term.lower() in generic_lower:
                components.append(term)
        
        return components[:2]  # Limit to 2 key components to avoid too many searches
    
    def process_openfda_results(self, results: List[Dict], drug_name: str) -> Dict:
        """Process OpenFDA results and extract relevant sections"""
        
        if not results:
            return {}
        
        # Use the first result (most relevant)
        result = results[0]
        
        processed_data = {
            "drug_name": drug_name,
            "source": "OpenFDA",
            "sections": {}
        }
        
        # Key sections to extract
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
            "information_for_patients": "patient_information"
        }
        
        for openfda_key, our_key in section_mapping.items():
            if openfda_key in result:
                content = result[openfda_key]
                # OpenFDA returns lists, join them
                if isinstance(content, list):
                    content = ' '.join(content)
                
                processed_data["sections"][our_key] = {
                    "title": our_key.replace('_', ' ').title(),
                    "content": content
                }
        
        # Add brand and generic names if available
        if "openfda" in result:
            processed_data["brand_names"] = result["openfda"].get("brand_name", [])
            processed_data["generic_names"] = result["openfda"].get("generic_name", [])
        
        return processed_data
    
    def download_all_drugs(self):
        """Download information for all bowel prep agents"""
        
        print("="*70)
        print("Starting download of bowel prep drug labels")
        print("="*70)
        
        all_drugs_data = []
        
        for drug_name, info in self.prep_agents.items():
            print(f"\n{'='*70}")
            print(f"Processing: {drug_name}")
            print(f"Generic: {info['generic']}")
            print(f"{'='*70}")
            
            drug_data = {
                "brand_name": drug_name,
                "generic_name": info["generic"],
                "dailymed_data": [],
                "openfda_data": {}
            }
            
            # 1. Search and download from DailyMed
            set_ids = self.search_dailymed(drug_name)
            for set_id in set_ids[:2]:  # Limit to first 2 results
                label_data = self.download_dailymed_label(set_id, drug_name)
                if label_data:
                    drug_data["dailymed_data"].append(label_data)
                time.sleep(1)
            
            # 2. Search OpenFDA with both brand and generic components
            openfda_results = self.search_openfda(drug_name, info["generic"])  # Pass generic name
            if openfda_results:
                processed_openfda = self.process_openfda_results(openfda_results, drug_name)
                drug_data["openfda_data"] = processed_openfda
                
                # Save OpenFDA data
                openfda_file = self.output_dir / "openfda_json" / f"{drug_name}_openfda.json"
                with open(openfda_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_openfda, f, indent=2, ensure_ascii=False)
                print(f"  Saved OpenFDA data: {openfda_file}")
            
            time.sleep(1)
            
            all_drugs_data.append(drug_data)
        
        # Save combined summary
        summary_file = self.output_dir / "all_drugs_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_drugs_data, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*70}")
        print(f"All data downloaded successfully!")
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*70}")
        
        return all_drugs_data
    
    def create_consolidated_dataset(self):
        """Create a consolidated, structured dataset from all sources"""
        
        print("\nCreating consolidated dataset...")
        
        consolidated = []
        
        # Load all DailyMed JSON files
        dailymed_dir = self.output_dir / "dailymed_json"
        for json_file in dailymed_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                consolidated.append(data)
        
        # Load all OpenFDA JSON files
        openfda_dir = self.output_dir / "openfda_json"
        for json_file in openfda_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                consolidated.append(data)
        
        # Save consolidated dataset
        consolidated_file = self.output_dir / "processed" / "consolidated_drug_labels.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        print(f"Consolidated dataset saved to: {consolidated_file}")
        print(f"Total entries: {len(consolidated)}")
        
        return consolidated


if __name__ == "__main__":
    # Initialize downloader
    downloader = BowelPrepDrugDownloader(output_dir="patient_kb/drug_labels")
    
    # Download all drug information
    all_data = downloader.download_all_drugs()
    
    # Create consolidated dataset
    consolidated = downloader.create_consolidated_dataset()
    
    print("\n" + "="*70)
    print("Download complete!")
    print(f"Check the 'patient_kb/drug_labels' directory for all files")
    print("="*70)