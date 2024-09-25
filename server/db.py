import psycopg2
import os
from typing import Dict, Tuple, Any

# PostgreSQL database connection details
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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

def select(query: str, params: Tuple[Any, ...]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    
    cols = [desc[0] for desc in cursor.description]
    result = cursor.fetchone()
    conn.close()

    if result:
        return dict(zip(cols, result))
    else:
        return {}
    
# Function to lookup the population of a city in a state
def lookup_city(city: str, state: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            city, geokey AS city_geokey, 
            county, parentgeokey AS county_geokey,
            state, SUBSTR(geokey, 1, 2) AS state_geokey,
            longitude, latitude, source_table
        FROM geoplaceview 
        WHERE LOWER(city)=LOWER(%s) AND state=%s
        ORDER BY source_table, population DESC NULLS LAST
        LIMIT 1
        """, (city, state))
    
    cols = [desc[0] for desc in cursor.description]
    result = cursor.fetchone()
    conn.close()

    if result:
        return dict(zip(cols, result))
    else:
        return {}



# Define the function that OpenAI will call
def lookup_value(geokey: str, type: str) -> Dict[str, Any]:
    type = type.lower() + '5yr'
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT ON (geokey) geokey, REPLACE(type, '5Yr', '') AS type, date::text, value
        FROM GeoDataP
        WHERE GeoKey = %s AND LOWER(Type) = %s
        ORDER BY geokey, date DESC;
        """, (geokey, type))
    
    cols = [desc[0] for desc in cursor.description]
    result = cursor.fetchone()
    conn.close()

    if result:
        return dict(zip(cols, result))
    else:
        return {}

if __name__ == '__main__':
    spec = lookup_city('Palo Alto', 'CA')
    pop = lookup_value(spec['city_geokey'], 'Population')
    print (pop)