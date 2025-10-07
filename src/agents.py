from openai import OpenAI
from anthropic import Anthropic
import google.genai as googleai
from abc import ABC, abstractmethod
from pathlib import Path
from config.definitions import ROOT_DIR, google_api_key
import json
from datetime import datetime, timedelta


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


class GeminiAgent(Agent):
    # google_models = set([
    #     "gemini-2.0-flash",
    #     "gemini-2.5-flash",
    #     "gemini-2.5-flash-preview",
    #     "gemini-2.5-pro",
    # ])

    def __init__(self, name, model, instructions, manage_history=False, tools=None):
        Agent.__init__(self, name, model, instructions, tools)
        self.agent_api = googleai.Client()
        self.history = manage_history
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

    def chat(self, prompt, new_chat=False):
        if self.history:
            if self.current_chat is None or new_chat:
                self.current_chat = self.agent_api.chats.create(
                    model=self.model,
                    config = googleai.types.GenerateContentConfig(
                        system_instruction=self.instructions,
                        temperature=0.0,
                ))
            return self._call_llm_chat(prompt).text
        else:
            return self._call_llm(prompt).text

    def _call_llm(self, prompt, history=None):
        messages = []
        if len(self.uploaded_pdfs) > 0:
            messages += self.uploaded_pdfs
        if history is not None:
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
    # anthropic_models = set([
    #     "claude-opus-4-1-20250805",
    #     "claude-opus-4-1",
    #     "claude-opus-4-20250514",
    #     "claude-opus-4-0",
    #     "claude-sonnet-4-20250514",
    #     "claude-sonnet-4-0",
    #     "claude-3-7-sonnet-20250219",
    #     "claude-3-7-sonnet-latest",
    #     "claude-3-5-haiku-20241022",
    #     "claude-3-5-haiku-latest",
    #     "claude-3-5-sonnet-20241022",
    #     "claude-3-5-sonnet-latest",
    #     "claude-3-5-sonnet-20240620",
    #     "claude-3-opus-20240229",
    #     "claude-3-opus-latest",
    #     "claude-3-haiku-20240307"])

    def __init__(self, name, model, instructions, tools):
        Agent.__init__(self, name, model, instructions, tools)
        self.agent_api = Anthropic()

    def chat(self, prompt):
        return self._call_llm(prompt).content[0].text

    def _call_llm(self, prompt, history=None):
        if history is None:
            history = self.history
        messages = [{"role": "system", "content": self.instructions}] + history + [{"role": "user", "content": prompt}]
        return self.agent_api.messages.create(model=self.model_api, messages=messages, max_tokens=1000)


class OpenAIAgent:

    def __init__(self, name, model, instructions, tools):
        Agent.__init__(self, name, model, instructions, tools)
        self.agent_api = OpenAI()
        self.history = [{"role": "user", "content": None}]

    def chat(self, prompt):
        return self._call_llm(prompt).choices[0].message.content

    def _call_llm(self, prompt, history=None):
        messages = [{"role": "system", "content": self.instructions}]
        if history is not None:
            messages += history
        messages += [{"role": "user", "content": prompt}]
        self.response = self.agent_api.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return self.response
