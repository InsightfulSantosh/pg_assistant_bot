from langchain.memory import ConversationBufferMemory
from env_utils import validate_environment
from pandas_agent import SmartPandasAgent
from rag_system import create_rag_system
from dotenv import load_dotenv
load_dotenv()

def setup_hybrid_system(csv_path):
    agent_instance = SmartPandasAgent(csv_path)
    rag_system = create_rag_system(csv_path, agent_instance.memory)
    return agent_instance, rag_system, agent_instance.memory

def initialize_system(csv_path):
    validate_environment()
    pandas_agent, rag_system, memory = setup_hybrid_system(csv_path)
    return pandas_agent, rag_system, memory
