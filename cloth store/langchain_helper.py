
from langchain_google_genai import GoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector

from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate

from langchain_community.embeddings import HuggingFaceHubEmbeddings
import os

from few_shots import few_shots
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ['api_key'])

def get_few_shot_chain():
    db_user = "root"
    db_password = "iamutsavdey1410"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    embeddings = HuggingFaceHubEmbeddings()
    to_vectorize=[" ".join(example.values()) for example in few_shots]
    vector_store=Chroma.from_texts(to_vectorize,embedding=embeddings,metadatas=few_shots)
    example_selector=SemanticSimilarityExampleSelector(
        vectorstore=vector_store,
        k=2,
    )
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return new_chain

if __name__ == "__main__":
    chain = get_few_shot_chain()
    print(chain.run("How many nike t shirts we have?"))