from initialize import initialize_system
from hybrid_query import hybrid_query

def main():
    csv_path = "data/professionals_in_pg.csv"
    pandas_agent, rag_system, memory = initialize_system(csv_path)

    print("PG RAG Chat Assistant — type 'exit' to quit")
    print("Type your query about the PG dataset. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        try:
            response = hybrid_query(user_input, pandas_agent, rag_system)
            print("Assistant:", response)
        except Exception as e:
            print("Assistant [error]:", str(e))

if __name__ == "__main__":
    main()
