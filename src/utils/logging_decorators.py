import time
import asyncio
from loguru import logger
from functools import wraps
from typing import Callable, Any
from fastapi import Request
from pydantic import BaseModel

def log_request_response_time(func: Callable) -> Callable:
    """
    Decorator to log the request, response, and execution time for FastAPI endpoints.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with logging.
    """
    
    def _get_request_info(*args, **kwargs) -> str:
        """Extract request information from function arguments."""
        request_obj = args[0] if args else kwargs.get('request')
        
        if isinstance(request_obj, Request):
            return f"method={request_obj.method}, url={request_obj.url}"
        elif isinstance(request_obj, BaseModel):
            return str(request_obj.model_dump())
        else:
            return str(request_obj)
    
    def _log_result(req_info: str, start_time: float, result=None, exception=None):
        """Log the result of function execution."""
        duration = time.time() - start_time
        if exception:
            logger.error(f"Request: {req_info}, Exception: {exception}, Time taken: {duration:.3f} seconds")
        else:
            logger.info(f"Request: {req_info}, Response: {result}, Time taken: {duration:.3f} seconds")
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        req_info = _get_request_info(*args, **kwargs)
        
        try:
            response = await func(*args, **kwargs)
            _log_result(req_info, start_time, response)
            return response
        except Exception as e:
            _log_result(req_info, start_time, exception=e)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        req_info = _get_request_info(*args, **kwargs)
        
        try:
            response = func(*args, **kwargs)
            _log_result(req_info, start_time, response)
            return response
        except Exception as e:
            _log_result(req_info, start_time, exception=e)
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper 