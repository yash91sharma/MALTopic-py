import pandas as pd
from tqdm import tqdm
import json

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
        survey_context: str,
        free_text_column: str,
        free_text_definition: str,
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
            free_text_definition=free_text_definition,
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
        self, survey_context: str, df: pd.DataFrame, enriched_column: str
    ) -> list[dict[str, str]]:
        """
        Generate topics from enriched text responses.

        Args:
            survey_context: Context about the survey to provide to the LLM
            df: Pandas DataFrame containing the data
            enriched_column: Name of the column containing enriched text responses

        Returns:
            List of dictionaries, each representing a topic with 'id', 'name', and 'description'
        """
        utils.validate_dataframe(df, [enriched_column])

        instructions = prompts.TOPIC_INST.format(survey_context=survey_context)

        all_columns: list[str] = df[enriched_column].dropna().tolist()
        labeled_columns = [
            f"{i+1}: {response}" for i, response in enumerate(all_columns)
        ]
        input_text = "\n\n".join(labeled_columns)

        try:
            raw_response = self.llm_client.generate(
                instructions=instructions, input=input_text
            )
        except Exception as e:
            raise RuntimeError(f"Error generating topics: {str(e)}")

        topics = []

        try:
            parsed_topics = json.loads(raw_response)
            for topic in parsed_topics:
                # Convert representative_words to string if it's a list
                if "representative_words" in topic and isinstance(
                    topic["representative_words"], list
                ):
                    topic["representative_words"] = ", ".join(
                        topic["representative_words"]
                    )

                # Ensure all values are strings
                for key in topic:
                    if not isinstance(topic[key], str):
                        topic[key] = str(topic[key])

                topics.append(topic)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {raw_response[:100]}..."
            )
        except Exception as e:
            raise ValueError(f"Error processing topics: {str(e)}")

        return topics
