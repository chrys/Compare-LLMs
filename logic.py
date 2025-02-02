import os
import json

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.callbacks import CallbackManager
from opik.integrations.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.anthropic import Anthropic

opik_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([opik_callback_handler])

class LLMCache:
    _cache = {}
    
    @classmethod
    def get_llm(cls, model_option):
        # Default embedding model
        default_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        
        # LLM configurations
        llm_configs = {
            "gpt-4o": {
                "api_key_source": "openai",
                "llm_factory": lambda api_key: OpenAI(model="gpt-4o", api_key=api_key),
                "embed_model": default_embed_model
            },
            "Claude Sonnet": {
                "api_key_source": "anthropic",
                "llm_factory": lambda api_key: Anthropic(model="claude-3-5-sonnet-20240620", api_key=api_key),
                "embed_model": default_embed_model
            },
            "gemini 1.5 pro": {
                "api_key_source": "google",
                "llm_factory": lambda api_key: Gemini(model="models/gemini-1.5-pro-latest", api_key=api_key),
                "embed_model": default_embed_model
            },
            "gpt o1": {
                "api_key_source": "openai",
                "llm_factory": lambda api_key: OpenAI(model="o1-mini", api_key=api_key),
                "embed_model": default_embed_model
            }
        }
        
        # Check if model is in cache
        if model_option not in cls._cache:
            try:
                # Get configuration
                if model_option not in llm_configs:
                    raise ValueError(f"Unsupported model: {model_option}")
                
                config = llm_configs[model_option]
                
                # Load API key
                api_key = cls.load_api_key(config['api_key_source'])
                
                # Create LLM
                llm = config['llm_factory'](api_key)
                embed_model = config['embed_model']
                
                # Cache the models
                cls._cache[model_option] = {
                    'llm': llm,
                    'embed_model': embed_model
                }
                
                # Update LlamaIndex settings
                Settings.llm = llm
                Settings.embed_model = embed_model
            
            except Exception as e:
                print(f"Error loading {model_option}: {e}")
                return None, None
        
        # Return cached models
        cached_model = cls._cache[model_option]
        return cached_model['llm'], cached_model['embed_model']
    
    @staticmethod
    def load_api_key(key):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            api_key = config['my_api'].get(key)
            if not api_key:
                raise ValueError(f"Key '{key}' not found in the configuration.")
            return api_key
        except Exception as e:
            print(f"Error loading API key: {e}")
            raise

def index_document(file_path):
    global query_engine
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = index.as_query_engine()

    # Remove the file after indexing
    os.remove(file_path)
    
def ask_llm(input):
    response = query_engine.query(input)
    answer = response.response
    return answer