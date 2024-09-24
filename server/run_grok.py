from flask import Flask, request, jsonify
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

import os
import json

from db import get_population

app = Flask(__name__)

@tool
def get_population_tool(city, state):
    '''Fetches the population of the given city and state.  Population of 0 means "Unknown".
    - city (str): The name of a city, e.g. "San Francisco"
    - state (str): The 2 letter code for the state, e.g. "CA".  If a user specifies the full state name (e.g. "California), pass it in here as "CA".
    
    If the "state" is not specified, do your best to infer it.
    '''
    data = get_population(city, state)
    # return {
    #     'city': city,
    #     'state': state,
    #     'population': data['population']
    # }
    
    return 0 if data['population'] == 'Unknown' else int(data['population'])
    

functions_map = {
    "get_population_tool": get_population_tool
}

def call_functions(llm_with_tools, user_prompt):
    system_prompt = 'You are a helpful assistant that answers questions about the population of cities.'

    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = functions_map.get(tool_call["name"].lower())
        if not selected_tool:
            continue

        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    return llm_with_tools.invoke(messages).content

llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model=os.getenv('GROQ_MODEL'))
tools = [get_population_tool]
llm_with_tools = llm.bind_tools(tools)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    response = call_functions(llm_with_tools, question)
    return jsonify({'answer': response})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
