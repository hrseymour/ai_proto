from psycopg2 import pool
import os
import logging
from typing import Dict, Tuple, Any, List

# PostgreSQL database connection details
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# Create a connection pool
connection_pool = None

def init_db_pool():
    global connection_pool
    if connection_pool is None:
        connection_pool = pool.SimpleConnectionPool(
            1, 20,  # Min 1, Max 20 connections
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

def get_db_connection():
    global connection_pool
    if connection_pool is None:
        init_db_pool()
    return connection_pool.getconn()

def release_db_connection(conn):
    global connection_pool
    if connection_pool:
        connection_pool.putconn(conn)

def select(query: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    logging.info(f"SELECT {params}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(query, params)
    cols = [desc[0] for desc in cursor.description]
    results = cursor.fetchall()

    release_db_connection(conn)  # Return connection to pool

    logging.info(f"SELECT complete")

    return [dict(zip(cols, row)) for row in results] if results else []
    
# Function to lookup the population of a city in a state
def lookup_city(city: str, state: str) -> Dict[str, Any]:
    query = """
        SELECT
            city, geokey AS city_geokey, 
            county, parentgeokey AS county_geokey,
            state, SUBSTR(geokey, 1, 2) AS state_geokey,
            longitude, latitude, source_table
        FROM GeoPlaceViewAlt
        WHERE LOWER(city)=LOWER(%s) AND state=%s
        ORDER BY source_table, population DESC NULLS LAST
        LIMIT 1
        """
    
    params =  (city, state)
    cities = select(query, params)
    return cities[0] if cities else {}

def lookup_values(geokey: str, types: List[str]) -> List[Dict[str, Any]]:
    types = [t.lower() + '5yr' for t in types]
    
    query = """
        SELECT DISTINCT ON (geokey, type) 
            geokey, 
            REPLACE(type, '5Yr', '') AS type, 
            date::text, 
            value
        FROM GeoDataP
        WHERE GeoKey = %s AND LOWER(Type) IN %s
        ORDER BY geokey, type, date DESC;
        """

    params = (geokey, tuple(types))
    return select(query, params)

if __name__ == '__main__':
    spec = lookup_city('Palo Alto', 'CA')
    pop = lookup_values(spec['city_geokey'], ['Population', 'BachelorsRate'])
    print (pop)