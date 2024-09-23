from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json

from db import lookup_population

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")


# Define the function that OpenAI will call
def get_population(city, state):
    population = lookup_population(city, state)
    if population:
        return {"population": str(population)}
    else:
        return {"error": "Population data not found for the given city and state."}

function_map = {
    'get_population': get_population
}

# OpenAI function definition
functions = [
    {
        "name": "get_population",
        "description": "Get the population of a city in a given state",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                },
                "state": {
                    "type": "string",
                    "description": "The name of the state. Use capitalized two-letter codes for states, e.g., 'CA' for 'California'"
                }
            },
            "required": ["city", "state"]
        }
    }
]

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]

    while True:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            functions=functions,
            function_call="auto"
        )

        message = response.choices[0].message
        messages.append({"role": message.role, "content": message.content or "", "function_call": message.function_call})

        if message.function_call is not None:
            # The assistant is requesting a function call
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            # Retrieve the function from the function_map
            function_to_call = function_map.get(function_name)

            if function_to_call:
                # Call the function with the provided arguments
                try:
                    function_response = function_to_call(**function_args)
                except Exception as e:
                    function_response = {"error": str(e)}
            else:
                # Function not found
                function_response = {"error": f"Function '{function_name}' not found."}

            # Append the function response to the messages
            messages.append({
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response)
            })
        else:
            # The assistant has provided a response; exit the loop
            break

    # Return the assistant's reply
    return jsonify({'response': message.content})

# Start the Flask app
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
