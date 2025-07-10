from rag_system import query_rag
from pandas_agent import SmartPandasAgent
import re
from question_parser import rewrite_user_question



def is_data_analysis_query(question):

    analysis_keywords = [
        'average', 'mean', 'median', 'count', 'sum', 'max', 'min',
        'filter', 'group', 'sort', 'compare', 'statistics', 'analysis',
        'how many', 'what is the', 'find all', 'list all', 'show me',
        'rent', 'price', 'city', 'profession', 'rating', 'accommodation',
        "them", "they", "those", 
        "above", "thier","there","how many","top",
        'calculate', 'total', 'highest', 'lowest', 'best', 'worst',
        'pg_name', 'pg_type', 'company', 'working_mode', 'age', 'gender',
        'stay_duration', 'join_date', 'amenities', 'languages_spoken',"plot", "graph"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in analysis_keywords)


def is_failure_response(answer):

    failure_indicators = [
        "cannot", "can't", "unable", "not sure", "error", "no information",
        "i don't know", "unsure", "data not available", "not found", "nothing",
        "fail", "failed", "could not", "couldn't"
    ]
    
    answer_lower = answer.lower().strip()

    # Check explicit failure keywords
    if any(indicator in answer_lower for indicator in failure_indicators):
        return True

    # Check for failure patterns that include zero/none but aren't valid results
    failure_zero_patterns = [
        r"\bno .* found\b",           # "no records found", "no data found"
        r"\bnothing found\b",         # "nothing found"
        r"\bno results\b",            # "no results"
        r"\bempty\b.*\bresult\b",     # "empty result"
        r"\bzero results\b",          # "zero results" 
        r"\bno data available\b",     # "no data available"
        r"\bcount is 0\b",            # "The count is 0"
        r"\bis 0\b",                  # "Average salary is 0"
        r"\bare 0\b",                 # "Results are 0"
        r"\b0\s+(records?|results?|entries?|items?)\b"  # "0 engineers", "0 records"
    ]
    # Check for failure patterns
    if any(re.search(pattern, answer_lower) for pattern in failure_zero_patterns):
        return True
    
    # Treat '0', 'zero', 'none' as failure only if they're isolated
    if re.fullmatch(r'\s*(0|0\.0+|zero|none)\s*', answer_lower):
        return True
    # Do not treat number-only results (e.g., '3') as failures
    
    return False


def hybrid_query(question, pandas_agent, rag_system, memory=None, force_rag=False):
    print(f"🔍 [Router] Question: {question}")

    if pandas_agent:
        df_context = pandas_agent.df
    else:
        from rag_system import _rag_df
        df_context = _rag_df

    rewritten_question = rewrite_user_question(question, df_context)
    print(f"📝 Rewritten Question: {rewritten_question}")

    if force_rag:
        print("Forced RAG route")
        return query_rag(rag_system, rewritten_question, memory), None, rewritten_question

    if is_data_analysis_query(rewritten_question) and pandas_agent:
        print("Detected as data analysis query")
        result = pandas_agent.query(rewritten_question)
        answer, fig = result if isinstance(result, tuple) else (result, None)
        if is_failure_response(answer):
            print("Pandas agent failed, falling back to RAG")
            if memory:
                memory.chat_memory.add_user_message(question)
                memory.chat_memory.add_ai_message(answer)
            return query_rag(rag_system, rewritten_question, memory), fig, rewritten_question
        print("Answered by pandas agent")
        return answer, fig, rewritten_question

    print("Routed directly to RAG (not data-analysis or no pandas_agent)")
    return query_rag(rag_system, rewritten_question, memory), None, rewritten_question