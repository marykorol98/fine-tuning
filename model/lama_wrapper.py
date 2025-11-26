import json
import logging
from ollama import Client

logger = logging.getLogger(__name__)

class ModelLamaLLM:    
    def __init__(self, model_name: str):
        self.model_name: str = model_name
    
    @property
    def model(self) -> Client:
        return Client()
        
    def generate_output(self, prompt: str) -> str:
        response = self.model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]

        try:
            data = json.loads(content)
            return data
        except:
            print("Модель вывела невалидный JSON, пробую исправить...")
            # Простейший фикс
            content = content[content.find("[") : content.rfind("]") + 1]
            return json.loads(content)