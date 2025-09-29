import os
from dotenv import load_dotenv
from pathlib import Path
from colorama import init, Fore, Back, Style

init(autoreset=True)

try:
    ROOT_DIR = str(Path(__file__).resolve().parent.parent)
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive sessions)
    ROOT_DIR = str(Path.cwd())

openai_api_key = None
anthropic_api_key = None
google_api_key = None
deepseek_api_key = None
groq_api_key = None

def load_api_keys():
    # Check for API keys
    load_dotenv(override=True)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')

    if openai_api_key:
        print(Style.BRIGHT + Fore.GREEN + f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print(Fore.RED + "OpenAI API Key not set", "red")

    if anthropic_api_key:
        print(Style.BRIGHT + Fore.GREEN + f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
    else:
        print(Fore.RED + "Anthropic API Key not set (and this is optional)")

    if google_api_key:
        print(Style.BRIGHT + Fore.GREEN + f"Google API Key exists and begins {google_api_key[:2]}")
    else:
        print(Fore.RED + "Google API Key not set (and this is optional)")

    if deepseek_api_key:
        print(Style.BRIGHT + Fore.GREEN + f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
    else:
        print(Fore.RED + "DeepSeek API Key not set (and this is optional)")

    if groq_api_key:
        print(Style.BRIGHT + Fore.GREEN + f"Groq API Key exists and begins {groq_api_key[:4]}")
    else:
        print(Fore.RED + "Groq API Key not set (and this is optional)")

    print()

    return {}