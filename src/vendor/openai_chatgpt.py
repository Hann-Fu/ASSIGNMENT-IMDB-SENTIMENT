from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

gpt_4_1_nano = ChatOpenAI(model="gpt-4.1-nano")