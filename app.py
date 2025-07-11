import streamlit as st
import pandas as pd
from collections import Counter

from initialize import setup_hybrid_system as initialize_system
from hybrid_query import hybrid_query
from question_parser import rewrite_user_question


st.set_page_config(page_title="PG Assistant", layout="wide")
st.title("🏠 PG Accommodation Assistant")

# ──────────────────────────────────────────────
# Load CSV with caching
# ──────────────────────────────────────────────

def load_csv(csv_path):
    return pd.read_csv(csv_path)

# ──────────────────────────────────────────────
# Main Streamlit chat logic
# ──────────────────────────────────────────────
def main():
    with st.sidebar:
        st.header("🧠 Conversation Memory")
        if "chat_history" in st.session_state and st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg['content']}")
        else:
            st.info("No conversation yet.")

    csv_path = "data/formated_data/professionals_in_pg.csv"
    df = load_csv(csv_path)

    # Always reinitialize for safety
    if "memory" not in st.session_state:
        pandas_agent, rag_system, memory = initialize_system(csv_path)
        st.session_state.pandas_agent = pandas_agent
        st.session_state.rag_system = rag_system
        st.session_state.memory = memory
    else:
        pandas_agent = st.session_state.pandas_agent
        rag_system = st.session_state.rag_system
        memory = st.session_state.memory

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Accept new user message
    user_input = st.chat_input("Ask a question about PG data...")
    if user_input:
        question = user_input.strip()
        rewritten_question = rewrite_user_question(question, df)

        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": rewritten_question})
        memory.chat_memory.add_user_message(rewritten_question)
        with st.chat_message("user"):
            st.markdown(rewritten_question)

        # Generate answer
        with st.chat_message("assistant"):
            try:
                result = hybrid_query(rewritten_question, pandas_agent, rag_system, memory)
                st.markdown(f"_(Rewritten question: {result[2] if isinstance(result, tuple) and len(result) == 3 else 'N/A'})_")

                if isinstance(result, tuple):
                    answer, fig = result[0], result[1]
                else:
                    answer, fig = result, None

                st.markdown(f"**Answer:** {answer}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"**Answer:** {answer}"})
                memory.chat_memory.add_ai_message(answer)

                if fig is not None:
                    st.pyplot(fig)

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()