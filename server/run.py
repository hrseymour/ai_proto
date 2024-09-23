from flask import Flask, request, jsonify
from openai import OpenAI
import psycopg2
import os
import json

app = Flask(__name__)

# PostgreSQL database connection details
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# Connect to PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

# Function to lookup the population of a city in a state
def lookup_population(city, state):
    conn = get_db_connection()
    cursor = conn.cursor()
    # SELECT population FROM geoplaceview WHERE city='Burlingame' AND state='CA';
    cursor.execute("SELECT population FROM geoplaceview WHERE LOWER(city)=LOWER(%s) AND state=%s", (city, state))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    else:
        return None

# Define the function that OpenAI will call
def get_population(city, state):
    population = lookup_population(city, state)
    if population:
        return {"population": population}
    else:
        return {"error": "Population data not found for the given city and state."}

# OpenAI function definition
tools = [
    {
        "type": "function",
        "function": {
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
                        "description": "The name of the state. Use capitalized two letter codes for states, e.g. 'CA' for 'California'"
                    }
                },
                "required": ["city", "state"]
            }
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
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content})

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            if tool_call.function.name == 'get_population':
                function_args = json.loads(tool_call.function.arguments)
                city = function_args.get('city')
                state = function_args.get('state')
                function_response = get_population(city, state)
                
                messages.append({
                    "role": "function",
                    "name": "get_population",
                    "content": json.dumps(function_response)
                })

    return jsonify({'response': messages[-1]['content']})

# Start the Flask app
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
