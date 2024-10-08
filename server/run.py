from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import yaml
import logging
# Gradio
import gradio as gr
import threading

from db import init_db_pool, lookup_city, lookup_values

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
    'lookup_values': lookup_values,
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
            "name": "lookup_values",
            "description": config['functions']['lookup_values']['description'] % "",
            "parameters": {
                "type": "object",
                "properties": {
                    "geokey": {"type": "string", "description": config['functions']['lookup_values']['parameters']['geokey']},
                    "types": {
                        "type": "array", 
                        "description": config['functions']['lookup_values']['parameters']['types'],
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["geokey", "types"]
            }
        }
    }
]

def process_conversation(question, history=None):
    app.logger.info("Ask begin")

    # Basic sanitization: limiting length and removing dangerous phrases
    MAX_QUESTION_LENGTH = 320
    dangerous_phrases = ["ignore all", "disregard", "forget previous"]

    if not question or len(question) > MAX_QUESTION_LENGTH:
        return {"error": "Invalid question"}

    for phrase in dangerous_phrases:
        if phrase in question.lower():
            return {"error": "Invalid question"}

    messages = [
        {"role": "system", "content": config['system_message']}
    ]

    # Add history to messages if provided
    if history:
        for entry in history:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["response"]})

    messages.append({"role": "user", "content": "The user wants to ask: " + question})

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
    return {"response": message.content}

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    history = data.get('history', [])

    result = process_conversation(question, history)
    return jsonify(result)

# Gradio UI
def gradio_chat(message, history):
    history_dict = [{"question": q, "response": r} for q, r in history]
    result = process_conversation(message, history_dict)
    return result["response"]

def new_conversation():
    return None

with gr.Blocks() as iface:
    chatbot = gr.Chatbot(height=800)
    with gr.Column():
        msg = gr.Textbox(
            label="What's your question about US cities?",
            placeholder="Type your question here...",
            lines=2
        )
        with gr.Row():
            send_btn = gr.Button("Send", size="sm")
            new_btn = gr.Button("New", size="sm")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = gradio_chat(user_message, history[:-1])
        history[-1][1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    send_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    new_btn.click(new_conversation, None, chatbot, queue=False)

# Start the Flask app and Gradio UI
if __name__ == '__main__':
    def run_flask():
        app.run(host='0.0.0.0', port=8080, debug=False)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Launch Gradio UI
    iface.launch(server_name="0.0.0.0", server_port=8081)