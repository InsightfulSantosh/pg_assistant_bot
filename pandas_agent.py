import os
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import initialize_agent, Tool

class SmartPandasAgent:
    def __init__(self, csv_path: str, model_name="gemini-1.5-pro"):
        self.csv_path = csv_path
        self.model_name = model_name
        self.df_full = None
        self.df = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.current_filters = {}
        self.last_entity_memory = {}

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        self._load()
        self._build_agent_tool()

    def _load(self):
        self.df_full = pd.read_csv(self.csv_path)
        self.df = self.df_full.copy()

        system_prompt = """
You are a data analysis assistant. You will be asked questions about a pandas DataFrame `df`.
- Always assume df is already filtered as needed.
- Do not apply additional filtering logic.
- If asked about top/bottom/maximum/minimum/average/etc., infer and compute intelligently using pandas.
- Do not guess column names; use only those in df.
"""

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            model_kwargs={"system_instruction": system_prompt}
        )

    def _build_agent_tool(self):
        self.agent_tool = create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="zero-shot-react-description"
        )

        self.langchain_agent = initialize_agent(
            tools=[
                Tool(
                    name="Pandas DataFrame Tool",
                    func=lambda q: self.agent_tool.run(q),
                    description="Use this tool to analyze the DataFrame `df` using pandas. The df is already filtered if needed."
                )
            ],
            llm=self.llm,
            agent="chat-conversational-react-description",
            memory=self.memory,
            verbose=True,
        )

    def reset_filters(self):
        self.df = self.df_full.copy()
        self.current_filters = {}
        self.last_entity_memory = {}
        self._build_agent_tool()

    def _apply_combined_entity_memory_filter(self):
        if not self.last_entity_memory:
            return
        self.df = self.df_full.copy()
        for col, val in self.last_entity_memory.items():
            if col in self.df.columns:
                if isinstance(val, list):
                    pattern = '|'.join([re.escape(str(v)) for v in val])
                else:
                    pattern = re.escape(str(val))
                self.df = self.df[self.df[col].astype(str).str.contains(pattern, case=False, na=False)]

        print(f"[DEBUG] Filtered df shape: {self.df.shape}")
        self._build_agent_tool()

    def _apply_context_filter(self, query: str, retry=False):
        lowered_query = query.lower()
        context_words = ["these", "those", "them","above","that", "there", "their"]
        use_memory = any(word in lowered_query for word in context_words)

        if use_memory:
            print("[DEBUG] Using entity memory filter due to contextual words.")
            self._apply_combined_entity_memory_filter()
            return

        new_filters = {}
        for col in self.df_full.columns:
            unique_values = self.df_full[col].dropna().astype(str).unique().tolist()
            matches = [val for val in unique_values if str(val).lower() in lowered_query]
            if matches:
                new_filters[col] = matches[0] if len(matches) == 1 else matches

        for col, new_val in new_filters.items():
            if col in self.current_filters and self.current_filters[col] != new_val:
                print(f"[INFO] Conflicting filter on '{col}', resetting filters.")
                self.reset_filters()
                break

        self.current_filters.update(new_filters)

        self.df = self.df_full.copy()
        for col, val in self.current_filters.items():
            if col in self.df.columns:
                if isinstance(val, list):
                    pattern = '|'.join([re.escape(str(v)) for v in val])
                    self.df = self.df[self.df[col].astype(str).str.contains(pattern, case=False, na=False)]
                else:
                    self.df = self.df[self.df[col].astype(str).str.lower() == str(val).lower()]

        if self.df.empty:
            if retry:
                print("[WARN] Filtered DataFrame is empty after retry. Skipping filter.")
                self.df = self.df_full.copy()
                return
            self.reset_filters()
            self._apply_context_filter(query, retry=True)

        self._build_agent_tool()

    def _rewrite_with_context(self, query: str) -> str:
        context_words = ["these", "those", "them","above","that", "there", "their"]
        if any(word in query.lower() for word in context_words):
            if self.last_entity_memory:
                filter_desc = ", ".join([f"{k} in {v}" for k, v in self.last_entity_memory.items()])
                return f"# NOTE: df refers to rows where {filter_desc}\n{query}"
        return query

    def _check_for_global_query(self, query: str) -> bool:
        """
        Checks if the query implies a need to reset filters and use the full dataset.
        """
        global_keywords = [
            "in the entire dataset", "overall", "in total", "whole data",
            "across all", "full data", "entire dataframe", "without filters"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in global_keywords)

    def _update_entity_memory_from_output(self, output: str, question: str):
        for col in self.df_full.columns:
            values = self.df_full[col].dropna().astype(str).unique().tolist()
            if all(v.replace('.', '', 1).isdigit() for v in values if isinstance(v, str)):
                continue
            if 2 <= len(values) < 100:
                matched = [v for v in values if v in output or v.lower() in question.lower()]
                if matched:
                    self.last_entity_memory[col] = list(set(self.last_entity_memory.get(col, []) + matched))
                    print(f"[INFO] Updated entity memory: {col} -> {self.last_entity_memory[col]}")

    def query(self, question: str):
        try:
            if any(word in question.lower() for word in ["reset", "start over", "clear filters", "ignore previous"]):
                self.reset_filters()
                return "Filters cleared. You can start fresh.", None

            if self._check_for_global_query(question):
                print("[INFO] Global context detected, resetting filters.")
                self.reset_filters()

            self._apply_context_filter(question)
            self._apply_combined_entity_memory_filter()

            rewritten = self._rewrite_with_context(question)
            result = self.langchain_agent.run(rewritten)

            self._update_entity_memory_from_output(result, question)

            fig = plt.gcf()
            if fig and fig.get_axes() and any(ax.has_data() for ax in fig.get_axes()):
                return result, fig

            return result, None

        except Exception as e:
            return f"Error: {str(e)}", None
