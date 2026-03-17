import random
import time
from typing import Dict, Any

class MedicationLabelDetector:
    """
    Mock AI Vision module for Hackathon purposes.
    Simulates extracting text from an image of a medication label 
    and parsing it into JSON, including meal_relation (before/after food).
    """
    
    def __init__(self):
        # A few common hackathon demo prescriptions the "AI" can detect.
        # meal_relation reflects real-world usage: e.g. Metformin is taken WITH/AFTER food,
        # Omeprazole is taken BEFORE food (30 min before).
        self.mock_responses = [
            {"medication_name": "Metformin",    "dose": "500mg",  "meal_relation": "after"},
            {"medication_name": "Amlodipine",   "dose": "5mg",    "meal_relation": "fixed"},
            {"medication_name": "Lisinopril",   "dose": "10mg",   "meal_relation": "fixed"},
            {"medication_name": "Atorvastatin", "dose": "20mg",   "meal_relation": "after"},
            {"medication_name": "Omeprazole",   "dose": "20mg",   "meal_relation": "before"},
        ]

    def extract_from_image(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Takes raw image bytes and returns the parsed medication JSON.
        Simulates network latency and vision processing.
        Returns medication_name, dose, and meal_relation.
        """
        # Simulate processing delay
        time.sleep(1.5)
        
        # In a real app we would call Gemini or OCR here:
        # e.g., model.generate_content([image_blob, "Extract med name, dose, and before/after food as JSON"])
        
        # Fallback hackathon mock:
        response = random.choice(self.mock_responses)
        return response
