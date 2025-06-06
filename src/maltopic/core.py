import json

import pandas as pd
from tqdm import tqdm

from . import prompts, utils


class MALTopic:
    def __init__(self, api_key: str, default_model_name: str, llm_type: str):
        self.api_key = api_key
        self.default_model_name = default_model_name
        self.llm_type = llm_type.lower()

        self.llm_client = self._select_agent(api_key, default_model_name)

    def _select_agent(self, api_key: str, model_name: str):
        if self.llm_type == "openai":
            from .llms import openai

            return openai.OpenAIClient(api_key, model_name)
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
            return self._generate_topics_from_text(instructions, input_text)
        except Exception as e:
            # Check if it's a token limit error
            if utils.is_token_limit_error(e):
                print(f"Token limit exceeded, splitting into batches...")
                return self._generate_topics_with_batching(
                    instructions, labeled_columns, topic_mining_context
                )
            else:
                raise RuntimeError(f"Error generating topics: {str(e)}")

    def _generate_topics_from_text(
        self, instructions: str, input_text: str
    ) -> list[dict[str, str]]:
        """
        Generate topics from a single text input.

        Args:
            instructions: The instruction prompt for the LLM
            input_text: The input text containing all responses

        Returns:
            List of topic dictionaries
        """
        raw_response = self.llm_client.generate(
            instructions=instructions, input=input_text
        )

        return self._parse_topics_response(raw_response)

    def _generate_topics_with_batching(
        self, instructions: str, labeled_columns: list[str], topic_mining_context: str
    ) -> list[dict[str, str]]:
        """
        Generate topics using batching when token limits are exceeded.

        Args:
            instructions: The instruction prompt for the LLM
            labeled_columns: List of labeled response strings
            topic_mining_context: Context for topic mining

        Returns:
            Consolidated list of topic dictionaries
        """
        try:
            batches = utils.split_text_into_batches(
                labeled_columns,
                max_tokens_per_batch=100000,
                model_name=self.default_model_name,
            )
        except ImportError:
            # Fallback to simple batching if tiktoken is not available
            batch_size = max(1, len(labeled_columns) // 4)  # Split into ~4 batches
            batches = [
                labeled_columns[i : i + batch_size]
                for i in range(0, len(labeled_columns), batch_size)
            ]

        print(f"Processing {len(batches)} batches...")

        all_topics = []

        for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
            batch_input = "\n\n".join(batch)

            try:
                batch_topics = self._generate_topics_from_text(
                    instructions, batch_input
                )
                all_topics.extend(batch_topics)
                print(
                    f"Batch {i+1}/{len(batches)}: Generated {len(batch_topics)} topics"
                )
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                continue

        return self._consolidate_topics(all_topics)

    def _parse_topics_response(self, raw_response: str) -> list[dict[str, str]]:
        """
        Parse the LLM response into topic dictionaries.

        Args:
            raw_response: Raw JSON response from LLM

        Returns:
            List of topic dictionaries
        """
        topics = []

        try:
            parsed_topics = json.loads(raw_response)
            for topic in parsed_topics:
                for key in topic:
                    if key != "representative_words" and not isinstance(
                        topic[key], str
                    ):
                        topic[key] = str(topic[key])
                topics.append(topic)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {raw_response[:100]}..."
            )
        except Exception as e:
            raise ValueError(f"Error processing topics: {str(e)}")

        return topics

    def _consolidate_topics(
        self,
        all_topics: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """
        Consolidate topics from multiple batches, removing duplicates and merging similar ones.

        This is a dumb(er) method. Use the dedup agent for a smarter consolidation.

        Args:
            all_topics: List of all topics from different batches

        Returns:
            Consolidated list of unique topics
        """
        if not all_topics:
            return []

        # Simple deduplication based on topic names
        seen_names = set()
        unique_topics = []

        for topic in all_topics:
            topic_name = topic.get("name", "").lower().strip()
            if topic_name and topic_name not in seen_names:
                seen_names.add(topic_name)
                unique_topics.append(topic)

        print(
            f"Consolidated {len(all_topics)} topics into {len(unique_topics)} unique topics"
        )
        return unique_topics
