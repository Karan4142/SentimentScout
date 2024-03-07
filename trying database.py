import mysql.connector

# Database configuration
db_config = {
    'host': 'aws-demo.cb6wyouss3ie.eu-north-1.rds.amazonaws.com',
    'user': 'admin',
    'password': 'Karan4142',
    'database': 'User'
}

# Create a connection to the database
conn = mysql.connector.connect(**db_config)

# Create a cursor object using the connection
cursor = conn.cursor()

# Example query
query = 'SELECT * FROM User_table'

# Execute the query
cursor.execute(query)

# Fetch the results
results = cursor.fetchall()
for row in results:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()
