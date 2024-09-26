from flask import Flask, request, jsonify
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

import os
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

@tool
def lookup_city_tool(city: str, state: str) -> Dict[str, Any]:
    '''TBD'''
    
    lookup_city_tool.__doc__ = \
        config['functions']['lookup_city']['description'] % \
        config['functions']['lookup_city']['args']
 
    spec = lookup_city(city, state)
    return spec
    
@tool
def lookup_value_tool(geokey: str, type: str) -> Dict[str, Any]:
    '''TBD'''
    
    lookup_value_tool.__doc__ = \
        config['functions']['lookup_value']['description'] % \
        config['functions']['lookup_value']['args']
 
    spec = lookup_value(geokey, type)
    return spec

functions_map = {
    "lookup_city_tool": lookup_city_tool,
    "lookup_value_tool": lookup_value_tool
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
