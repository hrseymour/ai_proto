from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import yaml
import logging
from typing import Dict, Any

from db import init_db_pool, lookup_city, lookup_value

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
app.debug = True

init_db_pool()  # pre-open

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

function_map = {
    'lookup_city': lookup_city,
    'lookup_value': lookup_value,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_city",
            "description": config['functions']['lookup_city']['description'] % "",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": config['functions']['lookup_city']['parameters']['city']},
                    "state": {"type": "string", "description": config['functions']['lookup_city']['parameters']['state']}
                },
                "required": ["city", "state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_value",
            "description": config['functions']['lookup_value']['description'] % "",
            "parameters": {
                "type": "object",
                "properties": {
                    "geokey": {"type": "string", "description": config['functions']['lookup_value']['parameters']['geokey']},
                    "type": {"type": "string", "description": config['functions']['lookup_value']['parameters']['type']}
                },
                "required": ["geokey", "type"]
            }
        }
    }
]

@app.route('/ask', methods=['POST'])
def ask_question():
    app.logger.info("Ask begin")
    data = request.json
    question = data.get('question')

    messages = [
        {"role": "system", "content": config['system_message']},
        {"role": "user", "content": question}
    ]

    num_tokens = 0
    for step in range(5):  # prevent infinite loop
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        num_tokens += response.usage.total_tokens

        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content or ""})

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                function_to_call = function_map.get(function_name)

                if function_to_call:
                    try:
                        app.logger.info(f"{step+1}. Calling: {function_name}({function_args})")
                        function_response = function_to_call(**function_args)
                    except Exception as e:
                        function_response = {"error": str(e)}
                else:
                    function_response = {"error": f"Function '{function_name}' not found."}

                # Append the function call to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": function_name, "arguments": tool_call.function.arguments}
                        }
                    ]
                })

                # Append the function response to messages
                messages.append({
                    "role": "tool",
                    "content": json.dumps(function_response),
                    "tool_call_id": tool_call.id
                })
        else:
            # The assistant has provided a response; exit the loop
            break

    app.logger.info(f"Ask end: {num_tokens} tokens used")
    return jsonify({'response': message.content})

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)