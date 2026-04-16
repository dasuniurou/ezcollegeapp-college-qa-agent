"""
Unified LLM client wrapping Claude (Anthropic), OpenAI, and Google Gemini.
Configure the active model in config.yaml under llm.default_model.
"""

import os
from pathlib import Path
from typing import Optional

# Auto-load .env from the project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass


class LLMClient:
    def __init__(self, model: str = "claude", model_id: Optional[str] = None, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._defaults = {
            "claude": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "gemini": "gemini-1.5-pro",
        }
        self.model_id = model_id or self._defaults.get(model)
        if self.model_id is None:
            raise ValueError(f"Unknown model provider: {model}. Choose from: claude, openai, gemini")
        self._client = self._init_client()

    def _init_client(self):
        if self.model == "claude":
            try:
                import anthropic
            except ImportError:
                raise ImportError("Run: pip install anthropic")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set")
            return anthropic.Anthropic(api_key=api_key)

        elif self.model == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("Run: pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            return openai.OpenAI(api_key=api_key)

        elif self.model == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Run: pip install google-generativeai")
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model_id)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.model == "claude":
            return self._generate_claude(prompt, system_prompt)
        elif self.model == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.model == "gemini":
            return self._generate_gemini(prompt, system_prompt)

    def _generate_claude(self, prompt: str, system_prompt: Optional[str]) -> str:
        kwargs = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def _generate_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, prompt: str, system_prompt: Optional[str]) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._client.generate_content(full_prompt)
        return response.text.strip()


def load_llm_from_config(config: dict) -> LLMClient:
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("default_model", "claude")
    model_cfg = llm_cfg.get("models", {}).get(provider, {})
    return LLMClient(
        model=provider,
        model_id=model_cfg.get("model_id"),
        max_tokens=model_cfg.get("max_tokens", 1024),
    )
