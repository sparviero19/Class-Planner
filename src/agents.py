from typing import Optional

from openai import OpenAI
from anthropic import Anthropic
import google.genai as googleai
from ollama import Client as OllamaClient
from abc import ABC, abstractmethod
from pathlib import Path
from config.definitions import ROOT_DIR, google_api_key
import json
from datetime import datetime, timedelta
import fitz

class Agent(ABC):

    def __init__(self, name, model, instructions, tools):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools

        self.history = [{"role": "user", "content": None}]
        self.response = None

    @abstractmethod
    def chat(self, prompt):
        return self._call_llm(prompt)

    @abstractmethod
    def _call_llm(self, prompt):
        raise NotImplementedError()

    def load_pdfs(self, paths: list[Path], use_cache=True):
        """Extracts text locally for the local Ollama model"""

        if isinstance(paths, str):
            paths = [Path(paths)]
        elif isinstance(paths, Path):
            paths = [paths]
        elif isinstance(paths, list) and all(isinstance(val, str) for val in paths):
            paths = [Path(path) for path in paths]

        self.uploaded_pdfs_paths = paths
        self.context_text = ""

        for path in paths:
            print(f"Locally extracting text from: {path.name}")
            try:
                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()

                self.context_text += f"\n\n--- Document: {path.name} ---\n{text}\n"
                doc.close()
            except Exception as e:
                print(f"Failed to extract text from {path.name}: {e}")
        # return paths for compatibility with the load_pdf of the GeminiAgent
        return paths

class GeminiAgent(Agent):

    def __init__(self, name, model, instructions, tools=None, api_key=None):
        Agent.__init__(self, name, model, instructions, tools)
        self.api_key = api_key
        if api_key is None:
            self.agent_api = googleai.Client()
        else:
            self.agent_api = googleai.Client(api_key=api_key)
        self.history = ""
        self.current_chat = None
        self.uploaded_pdfs = []
        self.uploaded_pdfs_paths = []
        self.cache_file = Path(ROOT_DIR) / "data/cache/file_cache.json"
        self._ensure_cache_directory()

    def _ensure_cache_directory(self):
        """Ensure the cache directory exists"""
        cache_dir = self.cache_file.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.cache_file.exists():
            self._save_cache({})

    def _load_cache(self):
        """Load the file cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cache(self, cache_data):
        """Save the file cache to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def _get_file_hash(self, path: Path):
        """Get a unique identifier for a file based on path and modification time"""
        stat = path.stat()
        return f"{path.name}_{stat.st_size}_{stat.st_mtime}"

    def chat(self, prompt):
        return self._call_llm(prompt).text

    def _call_llm(self, prompt, history=None):
        if history is None:
            history = self.history

        messages = []
        # add cached pdfs through google genai cacheing method
        if len(self.uploaded_pdfs) > 0:
            messages += self.uploaded_pdfs
        messages.append(history)

        messages.append(prompt)
        self.response = self.agent_api.models.generate_content(
            model=self.model,
            contents=messages,
            config=googleai.types.GenerateContentConfig(
                system_instruction=self.instructions,
                temperature=0.0
            ))
        return self.response

    def _call_llm_chat(self, prompt):
        messages = []
        if len(self.uploaded_pdfs) > 0:
            messages += self.uploaded_pdfs
        messages += prompt
        response = self.current_chat.send_message(messages)
        self.response = response
        return response

    def load_pdfs(self, paths: str | Path | list[str] | list[Path], use_cache=True):
        if isinstance(paths, str):
            paths = [Path(paths)]
        if isinstance(paths, Path):
            paths = [paths]
        if isinstance(paths, list):
            if all(isinstance(val, str) for val in paths):
                paths = [Path(path) for path in paths]
            elif not all(isinstance(val, Path) for val in paths):
                raise TypeError("paths must be str, Path or list of str, or list of Path")
        
        self.uploaded_pdfs_paths = paths
        self.uploaded_pdfs = []
        
        cache = self._load_cache() if use_cache else {}
        updated_cache = {}
        
        for path in paths:
            file_hash = self._get_file_hash(path)
            
            # Check if file is in cache and still valid
            if use_cache and file_hash in cache:
                cached_entry = cache[file_hash]
                try:
                    # Try to get the file info to verify it still exists on Google's servers
                    file_info = self.agent_api.files.get(name=cached_entry['file_name'])
                    print(f"Using cached file: {path.name} (expires: {cached_entry.get('expires_at', 'N/A')})")
                    self.uploaded_pdfs.append(file_info)
                    updated_cache[file_hash] = cached_entry
                    continue
                except Exception as e:
                    print(f"Cached file {path.name} no longer valid, re-uploading...")
            
            # Upload new file
            print(f"Uploading: {path.name}")
            uploaded_file = self.agent_api.files.upload(file=path)
            self.uploaded_pdfs.append(uploaded_file)
            
            # Store in cache
            updated_cache[file_hash] = {
                'file_name': uploaded_file.name,
                'path': str(path),
                'uploaded_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=2)).isoformat()  # Files typically expire after 48 hours
            }
        
        # Merge with existing cache entries that weren't accessed
        for key, value in cache.items():
            if key not in updated_cache:
                updated_cache[key] = value
        
        self._save_cache(updated_cache)
        return self.uploaded_pdfs

    def clear_cache(self):
        """Clear the file cache and delete all cached files from Google's servers"""
        cache = self._load_cache()
        for file_hash, cached_entry in cache.items():
            try:
                self.agent_api.files.delete(name=cached_entry['file_name'])
                print(f"Deleted cached file: {cached_entry['path']}")
            except Exception as e:
                print(f"Could not delete {cached_entry['path']}: {e}")
        self._save_cache({})

    def __del__(self):
        # Don't delete files automatically - they're cached
        # Only clear uploaded_pdfs reference
        self.uploaded_pdfs = []


class AnthropicAgent(Agent):

    # SyncPage[ModelInfo](data=[
    #     ModelInfo(id='claude-sonnet-4-6', created_at=datetime.datetime(2026, 2, 17, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Sonnet 4.6', type='model'),
    #     ModelInfo(id='claude-opus-4-6', created_at=datetime.datetime(2026, 2, 4, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Opus 4.6', type='model'),
    #     ModelInfo(id='claude-opus-4-5-20251101', created_at=datetime.datetime(2025, 11, 24, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Opus 4.5', type='model'),
    #     ModelInfo(id='claude-haiku-4-5-20251001', created_at=datetime.datetime(2025, 10, 15, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Haiku 4.5', type='model'),
    #     ModelInfo(id='claude-sonnet-4-5-20250929', created_at=datetime.datetime(2025, 9, 29, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Sonnet 4.5', type='model'),
    #     ModelInfo(id='claude-opus-4-1-20250805', created_at=datetime.datetime(2025, 8, 5, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Opus 4.1', type='model'),
    #     ModelInfo(id='claude-opus-4-20250514', created_at=datetime.datetime(2025, 5, 22, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Opus 4', type='model'),
    #     ModelInfo(id='claude-sonnet-4-20250514', created_at=datetime.datetime(2025, 5, 22, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Sonnet 4', type='model'),
    #     ModelInfo(id='claude-3-haiku-20240307', created_at=datetime.datetime(2024, 3, 7, 0, 0, tzinfo=datetime.timezone.utc),
    #               display_name='Claude Haiku 3', type='model')], has_more=False, first_id='claude-sonnet-4-6', last_id='claude-3-haiku-20240307')
    def __init__(self, name, model, instructions, tools=None, api_key=None):
        Agent.__init__(self, name, model, instructions, tools)

        if api_key:
            self.agent_api = Anthropic(api_key=api_key)
        else:
            self.agent_api = Anthropic()

        self.output_tokens = 10000
        if "3.5" in model or "3.7" in model:
            self.output_tokens = 8192
        elif "3" in model:
            self.output_tokens = 4096

        self.history = []

    def chat(self, prompt):
        output = self._call_llm(prompt)
        return output.content[0].text

    def _call_llm(self, prompt, history=None, max_tokens=None):
        if not max_tokens:
            max_tokens = self.output_tokens

        if history is None:
            history = self.history

        full_prompt = prompt
        if hasattr(self, 'context_text') and self.context_text:
            full_prompt = f"Here are the reference materials:\n{self.context_text}\n\nTask: {prompt}"

        messages = history + [{"role": "user", "content": full_prompt}]

        self.response = self.agent_api.messages.create(
            model=self.model,
            system=self.instructions,
            messages=messages,
            max_tokens=max_tokens,
        )
        return self.response


class OpenAIAgent(Agent):

    def __init__(self, name, model, instructions, tools=None, api_key=None, temperature=0.0):
        Agent.__init__(self, name, model, instructions, tools)
        if api_key:
            self.agent_api = OpenAI(api_key=api_key)
        else:
            self.agent_api = OpenAI()
        self.history = []

        if model.startswith("gpt-5") or model.startswith("o3"):
            self.reasoning_effort = "minimal" #unused at the moment
            self.temperature = 1.0
        else:
            self.temperature = temperature

    def chat(self, prompt):
        return self._call_llm(prompt).choices[0].message.content

    def _call_llm(self, prompt, history=None):

        if history is None:
            history = self.history

        messages = [{"role": "system", "content": self.instructions}]
        messages += history

        full_prompt = prompt
        if hasattr(self, 'context_text') and self.context_text:
            full_prompt = f"Here are the reference materials:\n{self.context_text}\n\nTask: {prompt}"

        messages += [{"role": "user", "content": full_prompt}]
        self.response = self.agent_api.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return self.response


class OllamaAgent(Agent):

    def __init__(self, name, model, instructions, tools=None):
        Agent.__init__(self, name, model, instructions, tools)
        self.agent_api = OllamaClient()

        self.history = []

    def chat(self, prompt):
        return self._call_llm(prompt)['message']['content']

    def _call_llm(self, prompt, history=None):
        if history is None:
            history = self.history

        messages = [{"role": "system", "content": self.instructions}]
        messages += history

        full_prompt = prompt
        if hasattr(self, 'context_text') and self.context_text:
            full_prompt = f"Here are the reference materials:\n{self.context_text}\n\nTask: {prompt}"

        messages += [{"role": "user", "content": full_prompt}]
        
        self.response = self.agent_api.chat(
            model=self.model,
            messages=messages,
            options={'temperature': 0.0}
        )
        return self.response
