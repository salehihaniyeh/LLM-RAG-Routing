# Project Report: Logical & Semantic Routing Using LLMs

## Introduction
This project aims to implement an automated routing system for user queries using a Logical Language Model (LLM). The system intelligently routes user questions to the most relevant data sources, specifically targeting Python documentation (python_docs), JavaScript documentation (js_docs), and general literature (literature_docs). This report walks through the system's architecture, code implementation, and usage examples.

## System Architecture

### Components
1. *Data Model*:
   - *RouteQuery Class*: Defines the structure for routing queries using Pydantic for type validation and enforcement.
   
2. *LLM Setup*:
   - *ChatGroq*: Utilizes the ChatGroq LLM, specifically the mixtral-8x7b-32768 model, configured with a low-temperature setting for deterministic outputs.
   
3. *Prompt Design*:
   - *ChatPromptTemplate*: Creates a structured prompt template that interacts with the LLM using a defined system message.

4. *Routing Mechanism*:
   - Combines the prompt and LLM to form the routing logic.
   - Utilizes a function to dynamically choose the appropriate chain based on the routed datasource.

## Code Implementation

### Data Model
The RouteQuery class ensures that the query will be routed to one of the predefined data sources:
python
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs", "literature_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


### LLM with Function Call
The ChatGroq LLM is configured to generate structured outputs based on the RouteQuery schema:
python
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)


### Prompt Design
The prompt is defined with a system message providing context to the LLM:
python
system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


### Router Definition
Combines the prompt and structured LLM to create a router:
python
router = prompt | structured_llm


### Example Queries
Queries are routed, and the results are printed:
python
question1 = """Which paper answers the question:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result1 = router.invoke({"question": question1})
print(result1)
print(result1.datasource)

question2 = """Why is my code giving me errors:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result2 = router.invoke({"question": question2})
print(result2)
print(result2.datasource)


### Routing Logic
A choose_route function dynamically selects the appropriate chain for a given result:
python
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here


## Conclusion
This project successfully demonstrates an automated query routing system leveraging LLM capabilities. The system efficiently directs queries to appropriate data sources, showcasing the potential of LLMs in enhancing information retrieval processes. Future work could involve expanding the dataset of documentation sources and refining the prompt templates for more nuanced routing.

### Project Report: Semantic Routing for Query Handling

#### Objective

The objective of this project is to implement semantic routing to determine the most appropriate expert prompt for a given user query. This project leverages the utility of cosine similarity for embedding comparison, allowing efficient routing between prompts tailored for distinct domains — Machine Learning (ML) and Medicine.

#### Setup

The project involves several key components and steps:

1. *Prompt Creation*:
   - Two specific prompts were generated to handle queries in the fields of Machine Learning and Medicine.

2. *Embeddings*:
   - The SentenceTransformerEmbeddings from langchain_community.embeddings.sentence_transformer is used to create embeddings for both the prompts and the incoming queries.

3. *Similarity Computation*:
   - Cosine similarity computed using cosine_similarity from langchain.utils.math is utilized to measure the semantic closeness between the query and the predefined prompts.

4. *Routing Mechanism*:
   - A lambda function (RunnableLambda) routes the input query to the most semantically appropriate prompt based on similarity scores.

5. *Execution Chain*:
   - A chain of operations is created, involving:
     - RunnablePassthrough for initial handling.
     - RunnableLambda for routing the query.
     - ChatGroq to process the routed prompt.
     - StrOutputParser to parse and deliver the output.

#### Detailed Steps

1. *Prompt Creation*:
   - Two distinct prompts were defined:
     - *ML Prompt*:
       - Focus: Large Language Models (LLMs).
       - Traits: Concise, clear explanations.
     - *Medicine Prompt*:
       - Focus: Medical queries.
       - Traits: Extensive medical knowledge, experience.

python
ML_template = """You are a an expert in Machine learning..."""
medicine_template = """You are a very good physician..."""


2. *Embedding Prompts*:

python
embeddings = SentenceTransformerEmbeddings()
prompt_templates = [ML_template, medicine_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


3. *Routing Function*:
   - Embedded the input query and computed the cosine similarity between the query and the prompt embeddings.
   - Selected the prompt with the highest similarity score.

python
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using Machine Learning" if most_similar == ML_template else "Using Medicine")
    return PromptTemplate.from_template(most_similar)


4. *Execution Chain*:
   - Combined all steps into a seamless process to handle and route the query efficiently.

python
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatGroq()
    | StrOutputParser()
)


#### Demonstration

The demo shows the chain in action:

python
print(chain.invoke("How does RAG improve the ability of language models?"))


#### Conclusion

This project efficiently directs user queries to the appropriate domain expert prompt based on semantic similarity, ensuring accurate and contextually relevant responses. The robustness of SentenceTransformerEmbeddings and cosine similarity facilitates precise and scalable query handling.

The output generated by semantic routing code involves invoking the right processes for different questions to provide context-appropriate responses. When the input "How does RAG improve the ability of language models?" is processed, the code effectively uses information from the Machine Learning domain to deliver a response. It explains that Retrieval-Augmented Generation (RAG) boosts language models' capabilities by enhancing factual accuracy and coherence. This is achieved by merging traditional language models with an information retrieval system that allows access to updated and pertinent information, making the generated text more reliable and contextually appropriate.

Conversely, when the input "I've been experiencing persistent fatigue and headaches, along with unexplained weight loss and increased thirst and urination. What is the underlying cause." is processed, the code switches context to the Medicine domain. Here, the response emphasizes the importance of consulting a healthcare professional while suggesting several potential underlying causes: diabetes, thyroid disorders, anemia, or mental health issues like anxiety or depression. The code ensures that the responses in both scenarios are highly relevant and supportive, showcasing its ability to adapt based on the input context by utilizing semantic routing.
