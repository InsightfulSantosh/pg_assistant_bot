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

def extract_forbidden_terms(text: str) -> List[str]:
    return [word for word in FORBIDDEN_REFERENTS if word in text.lower()]

def get_fuzzy_matches(
    user_input: str,
    candidates: List[str],
    case_map: Dict[str, str],
    cutoff=0.75
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
        topn: int = 30
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

        # Detect forbidden referents to be preserved
        present_forbidden = extract_forbidden_terms(user_input)

        # Build prompt
        template = """
You are a strict question rewriter that aligns user queries precisely to the dataset's schema and known values.

Your job is to rewrite the user's question by strictly following these rules:

---

### 🔒 STRICT RULES:

1. **Column Matching**
   - Use only the column names listed below.
   - Correct approximate column names to the closest actual column.
   - Do not invent or infer new columns beyond what's listed.

2. **Value Matching**
   - Use only the representative values provided below.
   - Fix typos or fuzzy matches to the closest known value when clear.
   - Use the exact casing shown below.

3. **Referent Preservation**
   - 🔴 ABSOLUTELY DO NOT reword, expand, guess, resolve, or replace these terms:
     {forbidden_terms}
   - They must appear exactly as in the original input.

4. **Intent Preservation**
   - Preserve the original meaning of the question.
   - Return only the corrected question—no explanation or extra words.

5. **No grammar correction**
   - DO NOT improve the grammar or sentence structure. Only fix known column/value matches.

---

## Column Names:
{columns}

## Representative Values:
{unique_values}

## User Question:
{corrected_input}

## Rewritten Question:
""".strip()

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self._llm | StrOutputParser()

        rewritten = chain.invoke({
        
            "corrected_input": corrected_input,
            "columns": ", ".join(columns),
            "unique_values": ", ".join(values),
            "forbidden_terms": ', '.join(FORBIDDEN_REFERENTS)
        }).strip()

        return rewritten


# Helper function to use the RewriteQuestionTool easily
def rewrite_user_question(question: str, df: pd.DataFrame) -> str:
    return RewriteQuestionTool(df)._run(question)

if __name__ == "__main__":
    df = pd.read_csv("data/professionals_in_pg.csv")
    # Convert column names and string values to lowercase
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    print(rewrite_user_question("how many profesion category in mumbay", df))