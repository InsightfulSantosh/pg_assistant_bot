from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

def rewrite_user_question(user_input: str, df: pd.DataFrame, model_name: str = "gemini-1.5-pro") -> str:
    """
    Rewrite the user's question based on the DataFrame's column names and values.

    Args:
        user_input (str): The raw user question.
        df (pd.DataFrame): The DataFrame to align with.
        model_name (str): Gemini model version.

    Returns:
        str: Rewritten question aligned with dataset structure.
    """
    def extract_col(df, date_threshold=0.8, freq_threshold=0.1):
        obj_df = df.select_dtypes(include="object")
        date_cols = []
        for col in obj_df.columns:
            parsed = pd.to_datetime(obj_df[col], errors='coerce', format='mixed')
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= date_threshold:
                date_cols.append(col)
        filtered_df = obj_df.drop(columns=date_cols, errors='ignore')
        columns = []
        for col in filtered_df.columns:
            if filtered_df[col].nunique() / filtered_df.shape[0] <= freq_threshold:
                columns.append(col)
        return columns

    unique_values = pd.unique(df[extract_col(df)].values.ravel())
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    template = """
    You are a question rewriter that aligns user input with actual dataset structure.

    Rewrite the user's question using:
    - Only column names from the dataset when possible.
    - Only values from the dataset (as provided below), unless they are common formats like dates.
    - If a column is not listed but clearly resembles a known one (e.g., 'join date'), retain it and correct the spelling minimally (e.g., `Join_Date`).
    - If a column or value is not listed and cannot be confidently corrected, preserve it as-is (do not reject or warn).
    - Use exact formatting:
    - Match user typos or approximations to the closest known values.
    - Do not introduce new columns or values unless inferred with high confidence.
    - Keep pronouns like "they", "those", "their", "that", "above", "them" unchanged.
    - For dates like '2023-08-16', accept and retain them even if not listed in values.

    ## Column Names:
    {columns}

    ## Representative Values:
    {unique_values}

    ## User Question:
    {user_input}

    ## Corrected Question:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "user_input": user_input,
        "columns": ", ".join(df.columns),
        "unique_values": unique_values
    })

if __name__ == "__main__":
    df= pd.read_csv("data/professionals_in_pg.csv")
    print(rewrite_user_question("how many profesion category in mumbay",df))