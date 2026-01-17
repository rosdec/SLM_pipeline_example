"""
Tool executor using Function Gemma.

Executes planned actions using constrained function calling.
"""

import ollama
from hr_tools import FUNCTIONS, TOOLS


class ToolExecutor:
    """Executes HR tools using Function Gemma."""
    
    def __init__(self):
        """
        Initialize the tool executor.
        
        Args:
        """
        self.functions = FUNCTIONS
        self.tools = TOOLS
    
    def call_function(self, step: dict):
        """
        Execute a single step using function calling.
        
        Args:
            step: The action step to execute
            
        Returns:
            The model's response including tool calls
        """
        messages = [
            {
                "role": "user",
                "content": f"Select the correct function and arguments for the following HR action step:\n{step}"
            }
        ]
        
        print(f"\n[STEP]\n{step}")
        
        response = ollama.chat(
            model="functiongemma",
            messages=messages,
            tools=self.tools
        )
        
        return response
    
    def execute_step(self, step: dict):
        """
        Execute a step and call the actual function.
        
        Args:
            step: The action step to execute
        """
        response = self.call_function(step)
        
        if response["message"]["role"] == "assistant" and "tool_calls" in response["message"]:
            tool_call = response["message"]["tool_calls"][0]
            fn = self.functions.get(tool_call.function.name)
            
            if not fn:
                print(f"[WARN] Unknown function: {tool_call.function.name}")
                return
            
            print(
                f"[EXECUTE] {tool_call.function.name} with arguments {tool_call.function.arguments}"
            )
            
            fn(**tool_call.function.arguments)
