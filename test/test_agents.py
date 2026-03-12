import os
from pathlib import Path
from dotenv import load_dotenv
from config.definitions import ROOT_DIR

# Load API keys from your .env file
load_dotenv()
api_keys = {
    'google': os.getenv('GOOGLE_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'openai': os.getenv('OPENAI_API_KEY')
}

from src.agents import GeminiAgent, AnthropicAgent, OllamaAgent, OpenAIAgent

TEST_GEMINI = False
TEST_ANTHROPIC = False
TEST_OPEAI = False
TEST_OLLAMA = True

def test_agents():
    print("--- Starting Agent Sandbox Test ---")

    # 1. Grab a small test PDF from your existing inputs (e.g., a syllabus)
    # Change this path to point to a real, small PDF in your project
    test_pdf_path = ROOT_DIR / Path("data/test/Syllabus IoT e Sistemi 3D v.1.3.pdf")

    if not test_pdf_path.exists():
        print(f"❌ Could not find test PDF at {test_pdf_path}. Please update the path.")
        return

    test_prompt = "Based on the provided document, write a 2-sentence summary of the main topic."
    system_instruction = "You are a helpful teaching assistant."

    # --- Test 1: Anthropic (Local Extraction + API Call) ---
    if TEST_ANTHROPIC:
        if api_keys.get('anthropic'):
            print("\n🧪 Testing AnthropicAgent...")
            try:
                claude = AnthropicAgent(
                    name="TestClaude",
                    model="claude-3-haiku-20240307",  # Use a fast/cheap model for testing
                    instructions=system_instruction,
                    api_key=api_keys['anthropic']
                )

                print("1. Testing Local PDF Extraction...")
                extracted_text = claude.load_pdfs([test_pdf_path])
                print(f"✓ Extracted {len(extracted_text)} characters.")

                print("2. Testing LLM Chat Call...")
                response = claude.chat(test_prompt)
                print(f"✓ Response: {response}")
            except Exception as e:
                print(f"❌ Anthropic Test Failed: {e}")
        else:
            print("\n⏭️ Skipping Anthropic (No API Key found)")

    # --- Test 2: Gemini (Google File API Call) ---
    if TEST_GEMINI:
        if api_keys.get('google'):
            print("\n🧪 Testing GeminiAgent...")
            try:
                gemini = GeminiAgent(
                    name="TestGemini",
                    model="gemini-3.1-flash-lite-preview",
                    instructions=system_instruction,
                    manage_history=False,
                    api_key=api_keys['google']
                )

                print("1. Testing Google File API Upload...")
                gemini.load_pdfs([test_pdf_path])
                print("✓ File uploaded/cached successfully.")

                print("2. Testing LLM Chat Call...")
                response = gemini.chat(test_prompt)
                print(f"✓ Response: {response}")
            except Exception as e:
                print(f"❌ Gemini Test Failed: {e}")
        else:
            print("\n⏭️ Skipping Gemini (No API Key found)")

    # --- Test 3: OpenAI (Local Extraction + API Call) ---
    if TEST_OPEAI:
        if api_keys.get('openai'):
            print("\n🧪 Testing OpenAIAgent...")
            try:
                openai = OpenAIAgent(
                    name="TestOpenAI",
                    model="gpt-5-nano-2025-08-07",  # Use a fast/cheap model for testing
                    instructions=system_instruction,
                    api_key=api_keys['openai'],
                    temperature=1.0,
                )

                print("1. Testing Local PDF Extraction...")
                extracted_text = openai.load_pdfs([test_pdf_path])
                print(f"✓ Extracted {len(extracted_text)} characters.")

                print("2. Testing LLM Chat Call...")
                response = openai.chat(test_prompt)
                print(f"✓ Response: {response}")
            except Exception as e:
                print(f"❌ OpenAI Test Failed: {e}")
        else:
            print("\n⏭️ Skipping OpenAI (No API Key found)")
    # --- Test 4: Ollama (Local Extraction + API Call) ---
    if TEST_OLLAMA:

        print("\n🧪 Testing OllamaAgent...")
        try:
            openai = OllamaAgent(
                name="TestOllama",
                model="ministral-3:latest",  # Use a fast/cheap model for testing
                instructions=system_instruction,
            )

            print("1. Testing Local PDF Extraction...")
            extracted_text = openai.load_pdfs([test_pdf_path])
            print(f"✓ Extracted {len(extracted_text)} characters.")

            print("2. Testing LLM Chat Call...")
            response = openai.chat(test_prompt)
            print(f"✓ Response: {response}")
        except Exception as e:
            print(f"❌ Ollama Test Failed: {e}")

if __name__ == "__main__":
    test_agents()