import json
from ollama import chat
from rich import print # For nice printing

# 1. Define your tool/function
def get_weather(city: str) -> str:
    """Get the current weather for a city. Args: city: The name of the city Returns: A string describing the weather"""
    # In a real app, this would call a weather API
    return json.dumps({'city': city, 'temperature': 22, 'unit': 'celsius', 'condition': 'sunny'})

messages = [{'role': 'user', 'content': 'What is the weather in Paris?'}]

# 2. Call Ollama with the 'functiongemma' model
response = chat(
    model='functiongemma',
    messages=messages,
    tools=[get_weather] # Pass your function here
)
print(f"[RESPONSE]\n{response}")
# 3. Check the response for a tool call
if response['message']['role'] == 'assistant' and 'tool_calls' in response['message']:
    tool_call = response['message']['tool_calls'][0]
    print(f"Gemma wants to call: {tool_call.function.name} with args: {tool_call.function.arguments}")

    # 4. Execute the tool call (your code)
    if tool_call.function.name == 'get_weather':
        # Call the actual function with arguments
        tool_output = get_weather(tool_call.function.arguments['city'])

        # 5. Send the tool output back to the model for a final answer
        messages.append(response['message']) # Add Gemma's tool call message
        messages.append({'role': 'tool', 'content': tool_output}) # Add tool output

        final_response = chat(
            model='functiongemma',
            messages=messages,
            tools=[get_weather]
        )
        print(f"Final Answer: {final_response['message']['content']}")
else:
    print(f"Final Answer: {response['message']['content']}")
