"""
LLM interface module for Dream Team framework.

Provides a wrapper for Gemini API with caching and error handling.
"""

import google.generativeai as genai
from typing import List, Dict, Optional
import os
import time
import json
import re


class GeminiLLM:
    """Wrapper for Gemini API with caching and error handling"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        self.model = genai.GenerativeModel(model_name)

        # Simple stats
        self.total_calls = 0
        self.total_tokens = 0

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: str = "text"  # "text" or "json"
    ) -> str:
        """Generate response with retry logic"""

        temp = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries):
            try:
                # Create model with system instruction if provided
                if system_instruction:
                    model = genai.GenerativeModel(
                        self.model_name,
                        system_instruction=system_instruction
                    )
                else:
                    model = self.model

                # Configure generation
                generation_config = {
                    "temperature": temp,
                }

                if response_format == "json":
                    generation_config["response_mime_type"] = "application/json"

                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                self.total_calls += 1
                # Note: Gemini API doesn't easily expose token counts like OpenAI
                # Could estimate with len(prompt.split()) but not accurate

                return response.text

            except Exception as e:
                print(f"⚠️  Gemini API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """Generate JSON response"""
        response_text = self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            response_format="json"
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """Multi-turn chat (for meetings)"""
        temp = temperature if temperature is not None else self.temperature

        # Convert to Gemini format
        # messages format: [{"role": "user"/"assistant", "content": "..."}]

        chat = self.model.start_chat(history=[])

        for msg in messages[:-1]:  # All but last
            role = "user" if msg["role"] == "user" else "model"
            chat.history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Send last message
        response = chat.send_message(
            messages[-1]["content"],
            generation_config={"temperature": temp}
        )

        self.total_calls += 1
        return response.text


# Global instance (lazy init)
_llm_instance = None


def get_llm(**kwargs) -> GeminiLLM:
    """Get or create global LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = GeminiLLM(**kwargs)
    return _llm_instance
