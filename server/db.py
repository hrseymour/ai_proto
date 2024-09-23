import psycopg2
import os

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

