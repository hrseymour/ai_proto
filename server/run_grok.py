from flask import Flask, request, jsonify
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

import os
import json
import yaml
import logging
from typing import Dict, Any, List

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

@tool
def lookup_city_tool(city: str, state: str) -> Dict[str, Any]:
    '''TBD'''
    spec = lookup_city(city, state)
    return spec
    
lookup_city_tool.description = \
        config['functions']['lookup_city']['description'] % \
        config['functions']['lookup_city']['args']

@tool
def lookup_values_tool(geokey: str, types: List[str]) -> List[Dict[str, Any]]:
    '''TBD'''
    spec = lookup_values(geokey, types)
    return spec

lookup_values_tool.description = \
        config['functions']['lookup_values']['description'] % \
        config['functions']['lookup_values']['args']

functions_map = {
    "lookup_city_tool": lookup_city_tool,
    "lookup_values_tool": lookup_values_tool
}

def call_functions(llm_with_tools, user_prompt):
    system_prompt = config['system_message']
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    
    for step in range(5):  # prevent infinite loop
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            break
        
        for tool_call in ai_msg.tool_calls:
            selected_tool = functions_map.get(tool_call["name"].lower())
            if not selected_tool:
                continue

            app.logger.info(f'{step+1}. Calling: {tool_call["name"]}({tool_call["args"]})')

            tool_output = selected_tool.invoke(tool_call["args"])
            if not isinstance(tool_output, str):
                tool_output = json.dumps(tool_output)

            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    final_answer = messages[-1].content \
        if isinstance(messages[-1], AIMessage) \
        else llm_with_tools.invoke(messages).content
    return final_answer 

llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model=os.getenv('GROQ_MODEL'))
tools = list(functions_map.values())
llm_with_tools = llm.bind_tools(tools)

@app.route('/ask', methods=['POST'])
def ask_question():
    app.logger.info("Ask begin")
    data = request.json
    question = data.get('question')

    response = call_functions(llm_with_tools, question)
    
    app.logger.info("Ask end")
    return jsonify({'answer': response})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
