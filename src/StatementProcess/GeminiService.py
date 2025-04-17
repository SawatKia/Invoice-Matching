import time
import os

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from google import genai
from google.genai import types

from log_utils import get_logger

class GeminiClient:
    """
    Class to handle interactions with Google's Gemini API.
    Implemented as a Singleton to ensure only one instance exists.
    """
    
    # Class variable to hold the singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to implement the singleton pattern
        """
        if cls._instance is None:
            cls._instance = super(GeminiClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_key: str = None, model_name: str = None) -> None:
        """
        Initialize the Gemini client.
        Will only initialize once, subsequent calls will be ignored.
        
        Args:
            api_key: API key for authentication
            model_name: Name of the model to use
        """
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        load_dotenv()
        self.logger = get_logger()
        self.start_time = 0
        self.end_time = 0
        self.api_key = ""
        self.model_name = ""
        self.isConnected = False

        self.systemprompt_path = Path("data/files/systemPrompts.txt")
        if not os.path.exists(self.systemprompt_path):
            self.logger.error(f"System prompt file does not exist: {self.systemprompt_path}")
            raise FileNotFoundError(f"System prompt file does not exist: {self.systemprompt_path}")
        
        self.init_config(api_key, model_name)
        self.client = genai.Client(api_key=self.api_key)

        self.isConnected = self.conection_test()
        if not self.isConnected:
            self.logger.error("Gemini API client connection failed")
            raise ConnectionError("Gemini API client connection failed")
        
        # Mark as initialized
        self._initialized = True
        self.logger.debug(f"Initialized GeminiClient with model: {self.model_name}")

    def init_config(self, api_key: str, model_name: str) -> None:
        """
        Initialize the Gemini client configuration.
        """
        # Get API key
        self.api_key = os.getenv("GEMINI_API_KEY", api_key)
        if not self.api_key:
            self.logger.error("API key is missing")
            raise ValueError("API key is missing")
        self.logger.debug(f"API Key: {'*' * len(self.api_key)}")
        
        # Get model name with default
        self.model_name = os.getenv("GEMINI_MODEL_NAME", model_name or "gemini-2.0-flash")
        if not self.model_name:
            self.model_name = "gemini-2.0-flash"  # Set default model
        self.logger.debug(f"Model Name: {self.model_name}")

    def conection_test(self) -> bool:
        """
        Test the connection to the Gemini API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Get model info
            model_info = self.client.models.get(model=self.model_name)
            
            # Log model details
            self.logger.debug(
                f"Model {self.model_name} metadata:\n"
                f"\tInput limit: {model_info.input_token_limit}, "
                f"Output limit: {model_info.output_token_limit}"
            )
            
            # Check if model info is valid
            if not model_info:
                self.logger.error("No model information available")
                raise Exception("No model information available")
                
            # Test token counting
            test_response = self.client.models.count_tokens(
                model=self.model_name,
                contents="Test connection"
            )
            self.logger.debug(f"Token count response: {test_response}")
            
            if not test_response or not hasattr(test_response, 'total_tokens'):
                self.logger.error("Invalid token count response")
                raise Exception("Invalid token count response")
                
            self.logger.info("Connection to Gemini API successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False
        
    def _read_system_prompt(self, file_path: Optional[str] = None) -> str:
        """
        Read system prompt from a file.
        
        Args:
            file_path: Path to the system prompt file
            
        Returns:
            Content of the system prompt file or empty string if file not found
        """

        if not file_path:
            file_path = self.systemprompt_path
        self.logger.debug(f"Reading system prompt from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.logger.debug(f"Successfully read system prompt ({len(content)} chars)")
                return content
        except FileNotFoundError:
            self.logger.error(f"System prompt file not found: {file_path}")
            raise FileNotFoundError(f"System prompt file not found: {file_path}")
            
    def generate_content(self, prompt: str, system_prompt_path: Optional[str] = None, 
                         max_retries: int = 3, retry_delay: int = 60) -> str:
        """
        Generate content using the Gemini API with retry functionality.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt_path: Path to the system prompt file (optional)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            The generated text response with markdown formatting removed
        """
        if not self.isConnected:
            self.logger.warning("Gemini API client is not connected, trying to connect...")
            status =self.conection_test()
            if not status:
                self.logger.error("Gemini API client connection failed")
                raise ConnectionError("Gemini API client connection failed")
        system_instruction = self._read_system_prompt(system_prompt_path)
        config = types.GenerateContentConfig(system_instruction=system_instruction)
        
        attempts = 0
        while attempts < max_retries:
            try:
                self.logger.debug(f"Sending request to Gemini API (attempt {attempts+1}/{max_retries})")
                self.start_time = time.time()
                
                # Get response from Gemini API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=config,
                    contents=prompt
                )
                self.end_time = time.time()
                duration = self.end_time - self.start_time
                
                self.logger.info(f"API request \x1b[5;38;5;14mcompleted\x1b[0m in {duration:.2f} seconds")
                
                # Extract text and clean markdown formatting
                self.logger.info("remove markdown formatting")
                response_text = response.text.strip()
                if response_text.startswith('```'):
                    # Remove opening ```json or ``` and closing ```
                    lines = response_text.split('\n')
                    # Remove first and last lines containing ```
                    content_lines = lines[1:-1]
                    # Remove 'json' from first line if present
                    if content_lines[0].strip() == 'json':
                        content_lines = content_lines[1:]
                    response_text = '\n'.join(content_lines).strip()
                
                # Log success with sample of response
                self.logger.debug(f"Cleaned response length: {len(response_text)} chars...")
                
                return response_text
                
            except Exception as e:
                attempts += 1
                error_type = type(e).__name__
                self.logger.error(f"API request failed with {error_type}: {str(e)}")
                
                if attempts >= max_retries:
                    self.logger.error(f"Maximum retry attempts ({max_retries}) reached")
                    raise
                
                self.logger.debug(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return ""
        
    @classmethod
    def get_instance(cls, api_key: str = None, model_name: str = None):
        """
        Get or create the singleton instance of GeminiClient.
        
        Args:
            api_key: API key for authentication
            model_name: Name of the model to use
            
        Returns:
            Singleton GeminiClient instance
        """
        # This will either create a new instance or return the existing one
        return cls(api_key=api_key, model_name=model_name)