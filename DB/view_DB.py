import sqlite3

def view_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if not tables:
        print("没有找到任何表。")
    else:
        print("数据库中的表：")
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
        for table in tables:
            table_name = table[0]
            print(f"\n表 '{table_name}' 的数据：")
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 63;")
                rows = cursor.fetchall()
                if rows:
                    for row in rows:
                        print(row)
                else:
                    print("  没有数据。")
            except Exception as e:
                print(f"  无法查询表 '{table_name}'，错误信息：{e}")

    # 关闭连接
    cursor.close()
    conn.close()

if __name__ == '__main__':
    db_path = "/cpfs01/shared/mabasic/zhouheng/ReSo/DB/llm_agent_train.db"
    view_database(db_path)




