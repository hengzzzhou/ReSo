import sqlite3

def view_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        print("error table")
    else:
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
        for table in tables:
            table_name = table[0]
            
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 63;")
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    print(row)
    cursor.close()
    conn.close()

if __name__ == '__main__':
    db_path = "ReSo/DB/llm_agent_train.db"
    view_database(db_path)




