import openai
import json
import time
import logging
logger = logging.getLogger(__name__)

class ModelGPT:
    MODEL_NAME = "gpt-4"
    
    def __init__(self, max_tokens: int = 512, temperature = 0.7):
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
    
    @property
    def model(self):
        return openai.ChatCompletion
        
    def generate_output(self, instruction: str, context: str) -> str:
        prompt = f"Instruction: {instruction}\nContext: {json.dumps(context, ensure_ascii=False)}\nOutput:"
        for _ in range(3):  # простая retry логика
            try:
                response = self.model.create(
                    model=self.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Error: {e}, retrying in 5s...")
                time.sleep(5)
        return "Error generating output"