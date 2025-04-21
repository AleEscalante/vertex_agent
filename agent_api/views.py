import logging
import json
from datetime import datetime
import pytz

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Import the query_agent function instead of directly using agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index import query_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'api_logs.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_api")

class AgentEndpoint(APIView):
    """
    API endpoint for interacting with the Vertex AI agent
    """
    
    def get_el_salvador_datetime(self):
        """Get current date and time in El Salvador timezone"""
        el_salvador_tz = pytz.timezone('America/El_Salvador')
        current_time = datetime.now(el_salvador_tz)
        return current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def post(self, request, *args, **kwargs):
        try:
            # Get data from the request
            data = request.data
            
            if 'messages' not in data or not isinstance(data['messages'], list):
                return Response(
                    {"error": "Invalid request format. 'messages' array is required."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Extract user message
            user_messages = [msg for msg in data['messages'] if msg.get('role') == 'user']
            if not user_messages:
                return Response(
                    {"error": "No user message found in the request."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get the last user message
            user_message = user_messages[-1]['content']
            
            # Check if it's the first interaction
            is_first_interaction = data.get('isFirstInteraction', False)
            
            # Get current date and time in El Salvador
            current_datetime = self.get_el_salvador_datetime()
            
            # Get or generate session ID
            session_id = data.get('session_id')
            if not session_id:
                # Generate a unique session ID if not provided
                import uuid
                session_id = str(uuid.uuid4())
                logger.info(f"Generated new session ID: {session_id}")
            else:
                logger.info(f"Using provided session ID: {session_id}")
            
            # Format the message for the agent
            formatted_message = f"""Mensaje del usuario: {user_message}"""
            
            # Log the incoming request
            logger.info(f"Received request - First Interaction: {is_first_interaction}")
            logger.info(f"User message: {user_message}")
            
            # Call the agent with session ID
            agent_response = query_agent(formatted_message, session_id)
            
            # Extract only the output part from the response
            response_content = ""
            if hasattr(agent_response, 'output'):
                response_content = agent_response.output
            elif isinstance(agent_response, dict) and 'output' in agent_response:
                response_content = agent_response['output']
            elif isinstance(agent_response, str):
                # Try to parse string as JSON if it looks like a dictionary
                if '{' in agent_response and '}' in agent_response:
                    try:
                        response_dict = eval(agent_response)
                        if isinstance(response_dict, dict) and 'output' in response_dict:
                            response_content = response_dict['output']
                    except:
                        response_content = agent_response
                else:
                    response_content = agent_response
            else:
                # Fallback to string representation if we can't extract output
                response_content = str(agent_response)
            
            # Format the response in the OpenAI API style format
            response_data = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        }
                    }
                ],
                "metadata": {
                    "timestamp": current_datetime,
                    "isFirstInteraction": is_first_interaction,
                    "session_id": session_id
                }
            }
            
            # Log the response
            logger.info(f"Agent response content: {response_content}")
            
            return Response(response_data)
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
