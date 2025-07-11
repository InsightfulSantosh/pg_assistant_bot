import re
import pandas as pd
import difflib
from typing import ClassVar, List, Set, Tuple, Dict

from pydantic import PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Strict referents you must preserve as-is
FORBIDDEN_REFERENTS: Set[str] = {
    "there", "those", "them", "these", "that", "above", "their",
    "dataset", "entire data", "whole data", "full data"
}

# Patterns that should NEVER be expanded or resolved
FORBIDDEN_PATTERNS = [
    r'\btop\s+\d+\b',           # top 5, top 10, etc.
    r'\bbottom\s+\d+\b',        # bottom 3, bottom 7, etc.
    r'\bfirst\s+\d+\b',         # first 2, first 8, etc.
    r'\blast\s+\d+\b',          # last 4, last 6, etc.
    r'\bhighest\s+\d+\b',       # highest 5, etc.
    r'\blowest\s+\d+\b',        # lowest 3, etc.
    r'\btop\s+\w+\b',           # top few, top many, etc.
    r'\bbest\s+\d+\b',          # best 5, etc.
    r'\bworst\s+\d+\b',         # worst 3, etc.
]

def extract_forbidden_terms(text: str) -> List[str]:
    forbidden = [word for word in FORBIDDEN_REFERENTS if word in text.lower()]
    return forbidden

def detect_forbidden_patterns(text: str) -> List[str]:
    """Detect patterns that should never be expanded (e.g., 'top 5', 'bottom 3')"""
    detected = []
    for pattern in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        detected.extend(matches)
    return detected

def get_fuzzy_matches(
    user_input: str,
    candidates: List[str],
    case_map: Dict[str, str],
    cutoff=0.90
) -> Dict[str, str]:
    """Returns a mapping of user phrases to dataset-correct casing via fuzzy match."""
    matches = {}
    tokens = re.findall(r"\b[\w()]+(?: [\w()]+)*\b", user_input.lower())
    for word in tokens:
        close = difflib.get_close_matches(word, candidates, n=1, cutoff=cutoff)
        if close:
            canonical = close[0]
            matches[word] = case_map.get(canonical, canonical)
    return matches

class RewriteQuestionTool(BaseTool):
    name: ClassVar[str] = "StrictQuestionRewriter"
    description: ClassVar[str] = (
        "Strictly rewrites user questions to align with the dataset schema and known values. "
        "Fixes typos using fuzzy matching, but does not alter referents or structure."
    )

    _df: pd.DataFrame = PrivateAttr()
    _llm: ChatGoogleGenerativeAI = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(self, df: pd.DataFrame, model_name: str = "gemini-1.5-pro"):
        super().__init__()
        self._df = df
        self._model_name = model_name
        self._llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def extract_values(
        self,
        date_threshold: float = 0.8,
        freq_threshold: float = 0.1,
        topn: int = 50
    ) -> Tuple[List[str], List[str]]:
        obj_df = self._df.select_dtypes(include="object")
        date_cols = []
        for col in obj_df.columns:
            try:
                parsed = pd.to_datetime(obj_df[col], errors="coerce", format="mixed")
                if parsed.notna().mean() >= date_threshold:
                    date_cols.append(col)
            except Exception:
                continue
        filtered_df = obj_df.drop(columns=date_cols, errors="ignore")
        values: Set[str] = set()
        for col in filtered_df.columns:
            if filtered_df[col].nunique() / len(filtered_df) <= freq_threshold:
                top_vals = filtered_df[col].dropna().value_counts().head(topn).index.tolist()
                values.update(str(v).strip() for v in top_vals if v and len(str(v).strip()) < 100)
        return list(self._df.columns), sorted(values)

    def _run(self, user_input: str) -> str:
        # Extract schema and values
        columns, values = self.extract_values()

        # Maps for original casing
        columns_map = {col.lower(): col for col in columns}
        values_map = {val.lower(): val for val in values}
        combined_map = {**columns_map, **values_map}
        combined_keys = list(combined_map.keys())

        # Fuzzy-correct the user input
        fuzzy_replacements = get_fuzzy_matches(user_input, combined_keys, combined_map)
        corrected_input = user_input
        for old, new in fuzzy_replacements.items():
            corrected_input = re.sub(rf"\b{re.escape(old)}\b", new, corrected_input, flags=re.IGNORECASE)

        # Detect forbidden referents and patterns to be preserved
        present_forbidden = extract_forbidden_terms(user_input)
        forbidden_patterns = detect_forbidden_patterns(user_input)

        # Build prompt
        template = """
You are a MINIMAL question rewriter. Your ONLY job is to fix obvious typos in column names and values and provide output in lower case and singular form.

🚫 ABSOLUTELY FORBIDDEN:
- DO NOT expand numerical references like "top N", "bottom N", "first N", "last N", etc.
- DO NOT replace or resolve referent words: "there", "those", "them", "these", "that", "above", "their"
- DO NOT infer or expand vague references to actual values
- DO NOT improve grammar, verb form, or sentence structure
- DO NOT add explanations or extra words
- DO NOT change meaning or intent
- DO NOT output plural forms if plural is present—convert to singular (e.g., "boys" → "boy", "tries" → "try", "cities" → "city")

✅ ONLY ALLOWED:
- Fix obvious typos in column names and values
- Convert plural to singular where applicable
- Use exact casing from the lists below

1. **Column Matching**
   - Use only the column names listed below.
   - Format column names using single quotes.
   - Match typos or plurals to the closest actual column name.
   - Do not invent or infer new columns.

2. **Value Matching**
   - Use only the values listed below.
   - Format values using single quotes.
   - Match typos or plurals to the closest actual value.
   - Use the exact casing shown.

DETECTED PATTERNS TO PRESERVE: {forbidden_patterns}
DETECTED REFERENTS TO PRESERVE: {present_forbidden}

Column Names: {columns}
Values: {unique_values}

User Question: {corrected_input}

Rewritten Question (FIX TYPOS ONLY):""".strip()


        prompt = PromptTemplate.from_template(template)
        chain = prompt | self._llm | StrOutputParser()

        rewritten = chain.invoke({
            "corrected_input": corrected_input,
            "columns": ", ".join(columns),
            "unique_values": ", ".join(values),
            "forbidden_patterns": ", ".join(forbidden_patterns) if forbidden_patterns else "None",
            "present_forbidden": ", ".join(present_forbidden) if present_forbidden else "None"
        }).strip()

        return rewritten


# Helper function to use the RewriteQuestionTool easily
def rewrite_user_question(question: str, df: pd.DataFrame) -> str:
    return RewriteQuestionTool(df)._run(question)

if __name__ == "__main__":
    df = pd.read_csv("data/formated_data/professionals_in_pg.csv")
 
    print(rewrite_user_question("how many profesion category in mumbay", df))