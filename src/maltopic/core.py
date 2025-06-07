import json

import pandas as pd
from tqdm import tqdm

from . import prompts, utils
from .stats import MALTopicStats


class MALTopic:
    def __init__(self, api_key: str, default_model_name: str, llm_type: str):
        self.api_key = api_key
        self.default_model_name = default_model_name
        self.llm_type = llm_type.lower()

        self.stats = MALTopicStats()

        self.llm_client = self._select_agent(api_key, default_model_name)

    def _select_agent(self, api_key: str, model_name: str):
        if self.llm_type == "openai":
            from .llms import openai

            return openai.OpenAIClient(api_key, model_name, stats_tracker=self.stats)
        raise ValueError("Invalid LLM api type. Choose 'openai'.")

    def enrich_free_text_with_structured_data(
        self,
        *,
        survey_context: str,
        free_text_column: str,
        structured_data_columns: list[str],
        df: pd.DataFrame,
        examples: list[str] = [],
    ) -> pd.DataFrame:
        """
        Enrich free text responses with structured data from other columns.

        Args:
            survey_context: Context about the survey to provide to the LLM
            free_text_column: Name of the column containing free text responses
            free_text_definition: Description of what the free text represents
            structured_data_columns: List of column names containing structured data
            df: Pandas DataFrame containing the data
            examples: Optional list of example enrichments for few-shot learning

        Returns:
            DataFrame with added column '{free_text_column}_enriched' containing enriched text
        """
        utils.validate_dataframe(df, [free_text_column] + structured_data_columns)

        instructions: str = prompts.ENRICH_INST.format(
            survey_context=survey_context,
            free_text_column=free_text_column,
            structured_data_columns=", ".join(structured_data_columns),
            examples="\n".join(examples) if examples else "No examples provided.",
        )

        enriched_column = f"{free_text_column}_enriched"
        results: list[str] = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching free text"):
            free_text = str(row[free_text_column])
            if not free_text or pd.isna(free_text):
                results.append("")
                continue

            # Format structured data
            structured_data: list[str] = []
            for col in structured_data_columns:
                if not pd.isna(row[col]):
                    structured_data.append(f"{col}: {row[col]}")
            structured_context: str = "\n".join(structured_data)

            # Build complete prompt for this row
            row_prompt: str = (
                f"{free_text_column}: {free_text}\n\n"
                f"Structured data:\n{structured_context}\n\n"
                f"Enriched Response:"
            )

            try:
                enriched_text: str = self.llm_client.generate(
                    instructions=instructions, input=row_prompt
                )
                results.append(enriched_text.strip())
            except Exception as e:
                error_msg = f"Error generating enriched text: {str(e)}"
                results.append(error_msg)

        df[enriched_column] = results
        return df

    def generate_topics(
        self, *, topic_mining_context: str, df: pd.DataFrame, enriched_column: str
    ) -> list[dict[str, str]]:
        """
        Generate topics from enriched text responses.

        Args:
            topic_mining_context: Context about the survey to provide to the LLM. Add the what and why of the topic mining to make this useful.
            df: Pandas DataFrame containing the data
            enriched_column: Name of the column containing enriched text responses

        Returns:
            List of dictionaries, each representing a topic with 'id', 'name', and 'description'
        """
        utils.validate_dataframe(df, [enriched_column])

        instructions = prompts.TOPIC_INST.format(survey_context=topic_mining_context)

        all_columns: list[str] = df[enriched_column].dropna().tolist()
        labeled_columns = [
            f"{i+1}: {response}" for i, response in enumerate(all_columns)
        ]

        # Try to process all data at once first
        input_text = "\n\n".join(labeled_columns)

        try:
            return utils.generate_topics_from_text(
                self.llm_client, instructions, input_text
            )
        except Exception as e:
            # Check if it's a token limit error
            if utils.is_token_limit_error(e):
                print(f"Token limit exceeded, splitting into batches...")
                return utils.generate_topics_with_batching(
                    self.llm_client,
                    instructions,
                    labeled_columns,
                    topic_mining_context,
                    self.default_model_name,
                )
            else:
                raise RuntimeError(f"Error generating topics: {str(e)}")

    def deduplicate_topics(
        self,
        *,
        topics: list[dict[str, str]],
        survey_context: str,
    ) -> list[dict[str, str]]:
        """
        Intelligently deduplicate topics using LLM to identify and merge overlapping topics.

        This function uses the LLM to smartly combine topics that have significant overlap
        and are not unique, while keeping genuinely unique topics as-is. It performs
        semantic deduplication rather than simple string matching.

        Args:
            topics: List of topic dictionaries to deduplicate
            survey_context: Context about the survey to help LLM make better decisions

        Returns:
            List of deduplicated topic dictionaries with the same structure as input
        """
        if not topics:
            return []

        if len(topics) <= 1:
            return topics.copy()

        # Validate topic structure
        utils.validate_topic_structure(topics)

        instructions = prompts.DEDUP_TOPICS_INST.format(survey_context=survey_context)

        # Format topics for LLM processing
        topics_json = json.dumps(topics, indent=2)

        try:
            raw_response = self.llm_client.generate(
                instructions=instructions,
                input=f"Topics to deduplicate:\n{topics_json}",
            )

            deduplicated_topics = utils.parse_topics_response(raw_response)

            # Validate that the output maintains the same structure
            utils.validate_topic_structure(deduplicated_topics)

            print(
                f"Deduplicated {len(topics)} topics into {len(deduplicated_topics)} unique topics"
            )
            return deduplicated_topics

        except Exception as e:
            print("Warning: Topic deduplication failed: {str(e)}")
            print("Returning original topics without deduplication.")
            return topics.copy()

    def get_stats(self) -> dict:
        """
        Get comprehensive usage statistics for the MALTopic instance.

        Returns:
            Dictionary containing detailed usage statistics including:
            - Total tokens used (input, output, total)
            - Number of API calls (successful, failed, total)
            - Average tokens per call
            - Success rate
            - Model-specific breakdowns
            - Recent call history
        """
        return self.stats.get_summary()

    def print_stats(self):
        """
        Print a formatted summary of usage statistics to the console.

        This method provides a user-friendly display of all tracked metrics
        including token usage, API call statistics, and performance metrics.
        """
        self.stats.print_summary()

    def reset_stats(self):
        """
        Reset all usage statistics to their initial state.

        This method clears all tracked data and restarts the statistics
        collection from a clean state.
        """
        self.stats.reset()
