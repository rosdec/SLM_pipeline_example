from ollama import chat

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    "New York": "22°C",
    "London": "15°C",
    "Tokyo": "18°C",
  }
  return temperatures.get(city, "Unknown")

messages = [{
    "role":"developer", 
    "content":f"""           
<start_of_turn>developer
You are a model that can do function calling with the following functions

<start_function_declaration>
declaration: get_temperature
    {{description:<escape>
        Gets the current weather in a given location.<escape>,
        parameters:
        {{properties:
            {{
                location:
                    {{
                        description:<escape>The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"<escape>,
                        type:<escape>STRING<escape>
                    }},
                required:[<escape>location<escape>],
                type:<escape>OBJECT<escape>
            }}
        }}
    }}
<end_function_declaration>
<end_of_turn>
"""},
    {"role":"user",
     "content":"""
        <start_of_turn>user
            Hey, what's the weather in Tokyo right now?
        <end_of_turn>
     """}]

response = chat(model="functiongemma", messages=messages, tools=[get_temperature], think=False)

print(response.message.content)