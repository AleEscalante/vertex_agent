import os
import logging
import json
import datetime
import traceback

# Import tools from index.py
from index import (
    consulta_clientes, 
    imagenes_menu,
    consulta_atributos,
    consulta_productos_menu,
    agent
)

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"tools_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tools_test")

def log_response(tool_name, params, response):
    """Format and log the response from a tool"""
    logger.info(f"===== Testing {tool_name} =====")
    logger.info(f"Parameters: {json.dumps(params, indent=2, ensure_ascii=False)}")
    
    try:
        # Try to format as JSON for better readability
        if isinstance(response, dict) or isinstance(response, list):
            logger.info(f"Response: {json.dumps(response, indent=2, ensure_ascii=False)}")
        else:
            logger.info(f"Response: {response}")
    except Exception:
        logger.info(f"Response: {response}")
    
    logger.info("=" * 50)

def test_consulta_clientes():
    """Test the customer query tool with various names and phone numbers"""
    logger.info("\n\n--- TESTING CONSULTA CLIENTES TOOL ---")
    
    test_cases = [
        {"nombre_cliente": "Juan Pérez", "telefono_cliente": "88887777"},
        {"nombre_cliente": "María López", "telefono_cliente": "99991111"},
        {"nombre_cliente": "Cliente Inexistente", "telefono_cliente": "00000000"}
    ]
    
    for params in test_cases:
        try:
            response = consulta_clientes(**params)
            log_response("consulta_clientes", params, response)
        except Exception as e:
            logger.error(f"Error testing consulta_clientes with {params}: {str(e)}")
            logger.error(traceback.format_exc())

def test_imagenes_menu():
    """Test the menu images tool"""
    logger.info("\n\n--- TESTING IMAGENES MENU TOOL ---")
    
    try:
        response = imagenes_menu()
        log_response("imagenes_menu", {}, response)
    except Exception as e:
        logger.error(f"Error testing imagenes_menu: {str(e)}")
        logger.error(traceback.format_exc())

def test_consulta_atributos():
    """Test the product attributes tool with various categories"""
    logger.info("\n\n--- TESTING CONSULTA ATRIBUTOS TOOL ---")
    
    test_cases = [
        {"category_name": "Pizza"},
        {"category_name": "CAFES"},
        {"category_name": "Hamburguesas"}
    ]
    
    for params in test_cases:
        try:
            response = consulta_atributos(**params)
            log_response("consulta_atributos", params, response)
        except Exception as e:
            logger.error(f"Error testing consulta_atributos with {params}: {str(e)}")
            logger.error(traceback.format_exc())

def test_consulta_productos_menu():
    """Test the product menu query tool with various search modes"""
    logger.info("\n\n--- TESTING CONSULTA PRODUCTOS MENU TOOL ---")
    
    test_cases = [
        {"product_name": "pizza margarita", "max_results": 3},
        {"product_name": "cafes", "search_mode": "categories", "max_results": 5},
        {"product_name": "10", "search_mode": "price", "max_results": 3},
        {"product_name": "Nuditos con ajo y parmesano", "single_result": True}
    ]
    
    for params in test_cases:
        try:
            response = consulta_productos_menu(**params)
            log_response("consulta_productos_menu", params, response)
        except Exception as e:
            logger.error(f"Error testing consulta_productos_menu with {params}: {str(e)}")
            logger.error(traceback.format_exc())

def test_agent_queries():
    """Test the agent with various prompts"""
    logger.info("\n\n--- TESTING AGENT QUERIES ---")
    
    test_prompts = [
        # Query menu items by price
        "¿Qué pizzas tienen un precio menor a 15 dólares?",
        
        # Search for specific product by name
        "Muéstrame información sobre la pizza margarita",
        
        # Search for products by category
        "¿Qué productos tienen en la categoría de cafés?",
        
        # Customer information lookup
        "Busca la información del cliente Juan Pérez con teléfono 88887777",
        
        # Product attributes query
        "¿Qué atributos o extras puedo agregar a una pizza?"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nTesting prompt: {prompt}")
        try:
            # Log which tools the agent might use for this prompt
            if "pizza" in prompt.lower() and "precio" in prompt.lower():
                logger.info("Expected tool: consulta_productos_menu with search_mode=price")
            elif "categoría" in prompt.lower() or "categoria" in prompt.lower():
                logger.info("Expected tool: consulta_productos_menu with search_mode=categories")
            elif "información sobre" in prompt.lower() or "muéstrame" in prompt.lower():
                logger.info("Expected tool: consulta_productos_menu with search_mode=products")
            elif "cliente" in prompt.lower() and "teléfono" in prompt.lower():
                logger.info("Expected tool: consulta_clientes")
            elif "atributos" in prompt.lower() or "extras" in prompt.lower():
                logger.info("Expected tool: consulta_atributos")
            
            response = agent.query(input=prompt)
            
            # Log the response
            if hasattr(response, 'text'):
                logger.info(f"Response: {response.text}")
            else:
                logger.info(f"Response: {response}")
                
            # Log more details if available
            if hasattr(response, 'metadata') and response.metadata:
                logger.info(f"Metadata: {json.dumps(response.metadata, indent=2, ensure_ascii=False)}")
                
        except Exception as e:
            logger.error(f"Error testing agent query with prompt '{prompt}': {str(e)}")
            logger.error(traceback.format_exc())
            
        logger.info("=" * 50)

def main():
    """Main function to run all tests"""
    logger.info("Starting tools tests...")
    
    # Test individual tools
    test_consulta_clientes()
    test_imagenes_menu()
    test_consulta_atributos()
    test_consulta_productos_menu()
    
    # Test agent queries
    test_agent_queries()
    
    logger.info("All tests completed. Results saved to log file.")
    print(f"Tests completed. Log file saved to: {log_file}")

if __name__ == "__main__":
    main()
