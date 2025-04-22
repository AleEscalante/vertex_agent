import vertexai
from vertexai import agent_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os
import requests
import logging
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents.format_scratchpad.tools import format_to_tool_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools_logs.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("vertex_tools")

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
    
    REGLAS DE USO:
    - EJECUTA esta función INMEDIATAMENTE en cuanto tengas tanto el nombre como el número de teléfono.
    - NO envíes mensajes intermedios diciendo que verificarás sin haber ejecutado esta función.
    - Después de ejecutar, SIEMPRE analiza el campo "isExistent" para determinar los siguientes pasos.
    - Si isExistent=true, agrega "clienteExiste": true al JSON de respuesta y pregunta si quiere usar la dirección registrada.
    - Si isExistent=false, agrega "clienteExiste": false al JSON y solicita la dirección completa.
    
    Args:
        nombre_cliente: Nombre del cliente a buscar.
        telefono_cliente: Número telefónico del cliente.
        
    Returns:
        dict: Información del cliente si existe, incluyendo código, nombre, teléfono y dirección.
              También incluye un campo 'isExistent' que indica si el cliente existe.
    """
    logger.info(f"TOOL EXECUTED: consulta_clientes - nombre: {nombre_cliente}, telefono: {telefono_cliente}")
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
        result = response.json()
        logger.info(f"consulta_clientes result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in consulta_clientes: {str(e)}")
        return {"error": str(e), "isExistent": False}

# Tool 2: Get menu images
def imagenes_menu() -> dict:
    """
    Obtiene las imágenes disponibles del menú.
    
    REGLAS DE USO:
    - Ejecuta esta función cuando el cliente solicite ver el menú, la carta o el catálogo.
    - También cuando pregunten por "qué tienen", "qué ofrecen" o expresiones similares.
    - En tu respuesta, incluye TODOS los enlaces devueltos, cada uno en su propio formato <url>.
    - No inventes enlaces; usa exactamente los proporcionados por esta función.
    - Los enlaces deben aparecer en el campo "mensaje" del JSON de respuesta.
    
    Returns:
        dict: Información sobre las imágenes disponibles del menú.
    """
    logger.info("TOOL EXECUTED: imagenes_menu")
    try:
        token = get_auth_token()
        url = "https://fn-imagenesmenu-547721852192.us-central1.run.app"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers)
        result = response.json()
        logger.info(f"imagenes_menu result summary: {len(result)} images received")
        return result
    except Exception as e:
        logger.error(f"Error in imagenes_menu: {str(e)}")
        return {"error": str(e)}

# Tool 3: Query product attributes
def consulta_atributos(category_name: str) -> dict:
    """
    Consulta los atributos disponibles para una categoría de producto.
    
    REGLAS DE USO:
    - Ejecuta esta función DESPUÉS de identificar la categoría de un producto que el cliente quiere ordenar.
    - Usa la respuesta para informar al cliente sobre qué atributos/extras puede seleccionar.
    - Cada atributo puede tener un costo adicional; asegúrate de incluirlo en el precio total.
    - Verifica que los valores proporcionados por el cliente sean válidos según la respuesta.
    - Si el cliente selecciona un valor no válido, notifícale y solicita una opción válida.
    
    Args:
        category_name: Nombre de la categoría a consultar (ej: "Pizza", "CAFES").
        
    Returns:
        dict: Atributos y valores aceptables para esa categoría de producto.
    """
    logger.info(f"TOOL EXECUTED: consulta_atributos - category: {category_name}")
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
        result = response.json()
        logger.info(f"consulta_atributos result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in consulta_atributos: {str(e)}")
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
    
    REGLAS DE USO:
    - Usa search_mode="products" cuando el cliente busque por nombre de producto específico.
    - Usa search_mode="categories" cuando el cliente busque por categoría de productos.
    - Usa search_mode="price" cuando el cliente busque productos por precio o rango de precios.
    - Limita la respuesta a máximo 5 resultados para no sobrecargar al cliente.
    - Usa single_result=True cuando necesites información detallada de un producto específico.
    - Al agregar productos al pedido, asegúrate de capturar correctamente nombre, cantidad y atributos.
    - Después de agregar un producto, consulta si el cliente desea agregar algo más.
    
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
    logger.info(f"TOOL EXECUTED: consulta_productos_menu - product: {product_name}, mode: {search_mode}, max_results: {max_results}, single: {single_result}")
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
        result = response.json()
        logger.info(f"consulta_productos_menu result summary: {len(result.get('products', []))} products found")
        return result
    except Exception as e:
        logger.error(f"Error in consulta_productos_menu: {str(e)}")
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
    ("system", 
     "Eres un asistente de restaurante que ayuda a los clientes a pedir comida, consultar el menú y obtener información sobre productos. Usa las herramientas disponibles"
     "para obtener información precisa."),
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