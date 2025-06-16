from langchain.prompts import PromptTemplate
from vendor.openai_chatgpt import gpt_4_1_nano
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Literal

class ClassifyResult(BaseModel):
    """Model for sentiment classification result."""
    sentiment: Literal["positive", "negative"] = Field(description="The result of the classification: 'positive' for Positive, 'negative' for Negative")


def create_sentiment_classifier():
    """
    Create a sentiment classification chain.

    Returns:
        callable: A chain that takes text input and returns ClassifyResult with sentiment classification.
    """

    # Define the prompt template

    output_parser = PydanticOutputParser(pydantic_object=ClassifyResult)

    template = """
    You are a sentiment classifier for movie reviews.
    Classify the following movie review into 'positive' or 'negative' sentiment.

    Text: {text}

    Output format:
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )

    # Create the chain
    classify_chain = prompt | gpt_4_1_nano | output_parser
    
    return classify_chain

sentiment_classify_chain = create_sentiment_classifier()









