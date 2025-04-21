import vertexai
from vertexai import agent_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages

load_dotenv()

model = "gemini-2.0-flash"

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    # (you must create your safety settings using the previous step first).
    "safety_settings": safety_settings,
}

vertexai.init(
    project="project-gcp-tst",
    location="us-central1",
    staging_bucket="gs://logs-middleware-chatwoot-tst",
)

# Get authenticated Google Cloud identity token
def get_auth_token():
    """Get an authentication token for Google Cloud Run functions."""
    auth_req = Request()
    return id_token.fetch_id_token(auth_req, "https://fn-consultaproductosmenu-547721852192.us-central1.run.app")

# Tool 1: Query customer information
def consulta_clientes(nombre_cliente: str, telefono_cliente: str) -> dict:
    """
    Busca información de clientes basado en el nombre y teléfono.
    
    Args:
        nombre_cliente: Nombre del cliente a buscar.
        telefono_cliente: Número telefónico del cliente.
        
    Returns:
        dict: Información del cliente si existe, incluyendo código, nombre, teléfono y dirección.
              También incluye un campo 'isExistent' que indica si el cliente existe.
    """
    try:
        token = get_auth_token()
        url = "https://fn-consultaclientes-547721852192.us-central1.run.app"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "nombreCliente": nombre_cliente,
            "telefonoCliente": telefono_cliente
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e), "isExistent": False}

# Tool 2: Get menu images
def imagenes_menu() -> dict:
    """
    Obtiene las imágenes disponibles del menú.
    
    Returns:
        dict: Información sobre las imágenes disponibles del menú.
    """
    try:
        token = get_auth_token()
        url = "https://fn-imagenesmenu-547721852192.us-central1.run.app"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Tool 3: Query product attributes
def consulta_atributos(category_name: str) -> dict:
    """
    Consulta los atributos disponibles para una categoría de producto.
    
    Args:
        category_name: Nombre de la categoría a consultar (ej: "Pizza", "CAFES").
        
    Returns:
        dict: Atributos y valores aceptables para esa categoría de producto.
    """
    try:
        token = get_auth_token()
        url = "https://fn-consultaatributos-547721852192.us-central1.run.app/searchCategory"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "categoryName": category_name,
            "sheetNames": ["AtributosMenu"],
            "tabSheet": "CATEGORIA"
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Tool 4: Query menu products
def consulta_productos_menu(
    product_name: str, 
    search_mode: str = "products", 
    max_results: int = 5, 
    single_result: bool = False
) -> dict:
    """
    Consulta productos del menú por nombre, categoría o precio.
    
    Args:
        product_name: Texto para buscar productos. Puede ser nombre del producto, 
                      categoría o precio según el modo de búsqueda.
        search_mode: Modo de búsqueda, puede ser "products" (predeterminado), 
                     "categories" o "price".
        max_results: Número máximo de resultados a devolver.
        single_result: Si es True, devuelve solo el primer resultado exacto.
        
    Returns:
        dict: Lista de productos que coinciden con la búsqueda.
    """
    try:
        token = get_auth_token()
        url = "https://fn-consultaproductosmenu-547721852192.us-central1.run.app"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "productName": product_name,
            "searchMode": search_mode,
            "maxResults": max_results,
            "singleResult": single_result
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Chat history integration with Firestore
def get_session_history(session_id: str):
    """
    Get chat message history from Firestore for a specific session.
    
    Args:
        session_id: Unique identifier for the chat session
        
    Returns:
        FirestoreChatMessageHistory: Chat history manager for the session
    """
    client = firestore.Client(project="project-gcp-tst")
    return FirestoreChatMessageHistory(
        client=client,
        session_id=session_id,
        collection="restaurant-chat-history",
        encode_message=False,
    )

# Custom prompt template for the agent
custom_prompt_template = {
    "user_input": lambda x: x["input"],
    "history": lambda x: x["history"],
    "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
} | ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de restaurante que ayuda a los clientes a pedir comida, consultar el menú y obtener información sobre productos. Usa las herramientas disponibles para obtener información precisa."),
    ("placeholder", "{history}"),
    ("user", "{user_input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Initialize the agent with chat history and custom prompt
agent = agent_engines.LangchainAgent(
    model=model,
    tools=[
        consulta_clientes,
        imagenes_menu,
        consulta_atributos,
        consulta_productos_menu
    ],
    model_kwargs=model_kwargs,
    chat_history=get_session_history,
    prompt=custom_prompt_template,
)

# Helper function for making queries with session ID
def query_agent(user_input, session_id=None):
    """
    Query the agent with user input and optional session ID for conversation memory.
    
    Args:
        user_input: The user's message or question
        session_id: Optional session identifier for maintaining conversation context
        
    Returns:
        The agent's response
    """
    config = {}
    if session_id:
        config = {"configurable": {"session_id": session_id}}
    
    return agent.query(input=user_input, config=config)