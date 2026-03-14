import random
import time
from typing import Dict, Any

class MedicationLabelDetector:
    """
    Mock AI Vision module for Hackathon purposes.
    Simulates extracting text from an image of a medication label 
    and parsing it into JSON.
    """
    
    def __init__(self):
        # A few common hackathon demo prescriptions that the "AI" can detect
        self.mock_responses = [
            {"medication_name": "Metformin", "dose": "500mg"},
            {"medication_name": "Amlodipine", "dose": "5mg"},
            {"medication_name": "Lisinopril", "dose": "10mg"},
            {"medication_name": "Atorvastatin", "dose": "20mg"},
            {"medication_name": "Omeprazole", "dose": "20mg"},
        ]

    def extract_from_image(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Takes raw image bytes and returns the parsed medication JSON.
        Simulates network latency and vision processing.
        """
        # Simulate processing delay
        time.sleep(1.5)
        
        # In a real app we would call Gemini or OCR here:
        # e.g., model.generate_content([image_blob, "Extract med name and dose as JSON"])
        
        # Fallback hackathon mock:
        response = random.choice(self.mock_responses)
        return response
