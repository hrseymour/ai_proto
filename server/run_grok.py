from flask import Flask, request, jsonify
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq

import os
import logging
from typing import Dict, Any

from db import get_db_connection, lookup_city, lookup_value

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
app.debug = True

get_db_connection()  # pre-open

@tool
def lookup_city_tool(city: str, state: str) -> Dict[str, Any]:
    '''Lookup the JSON specification of the given city, state.
    
    Args:
    - city (str): The name of a city, e.g. "San Francisco".
    - state (str): The 2 capital letter code for the state, e.g. "CA".  
                   If a user specifies the full state name (e.g. "California"), pass it in here as "CA".
                   If the state is not specified, do your best to infer it.
                   
    Here is a sample return dict:
    
    {
        'city': 'Palo Alto', 
        'city_geokey': '0655282', 
        'county': 'Santa Clara', 
        'county_geokey': '06085', 
        'state': 'CA', 
        'state_geokey': '06', 
        'longitude': -122.1430195, 
        'latitude': 37.4418834, 
        'source_table': 1
    }
    
    The geokey keys are used to lookup values using the "lookup_value_tool" tool, which can lookup
    socio-economic values for the city, county or state (using their geokey).
    
    The returned JSON spec (dict) also teaches you values like: county, latitude and longitude.
    
    An empty dict {} is returned for not found.
    '''
    spec = lookup_city(city, state)
    return spec
    
@tool
def lookup_value_tool(geokey: str, type: str) -> Dict[str, Any]:
    '''Fetch the latest value for the passed in type for the city (or county or state) identified by geokey.
    
    Args:
    - geokey (str): The geokey for the city (or county or state).  The "lookup_city_tool" tool can be used
                    to find geokey.  For a city, geokey is FIPS State code + FIPS City code, so the user
                    could also simply used the Geokey in a question without looking it up.
    - type (str): The type of the value being looked up.  Available types are:
    
        Population
        BachelorsRate
        BelowPovertyRate
        LaborForceRate
        MedianHomeValue
        PerCapitaIncome
        PopulationMedianAge
        RentOccupiedRatio
        UnemploymentRate
        
    Hopefully the type names are self-explanatory.  
                   
    Here is a sample return dict:
    
    { 
        'geokey': '0655282', 
        'type': 'Population', 
        'date': '2022-01-01', 
        'value': 67901.0
    }
    
    The date tells you the date of the latest reading.
    
    An empty dict {} is returned for not found.
    '''
    spec = lookup_value(geokey, type)
    return spec

functions_map = {
    "lookup_city_tool": lookup_city_tool,
    "lookup_value_tool": lookup_value_tool
}

def call_functions(llm_with_tools, user_prompt):
    system_prompt = """
        You are a helpful assistant that answers questions about cities.  It is usually a two step process
        to lookup numerical values like population.  Use "lookup_city_tool" to find a geokey for a city.  Then
        use "lookup_value_tool" to lookup a value like population.  For example, if someone asks "What is the
        population of Palo Alto, California"?  You will call:
        
        lookup_city_tool("Palo Alto", "CA")
        lookup_value_tool("0655282", "Population")
        
        The first call returns city_geokey for you ("0655282") and the second call returns the population
        that was requested (e.g. 67901).
        
        When it comes to value types, here are display instructions:
            
        Population and PopulationMedianAge are integers.  Sample display: 28,405
        Rate and Ratio values are percentages (0-100).   Sample display: 54.3%
        MedianHomeValue and PerCapitaIncome are $ values.   Sample display: $123,456
    """
    
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
    app.run(debug=True)
