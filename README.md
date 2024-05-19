# Project Report: Logical & Semantic Routing Using LLMs

## Introduction
This project aims to show how to implement an automated routing system for user queries using logical and semantic routing. In logical routing, the system intelligently routes user questions to the most relevant data sources. In semantic routing, the most appropriate expert prompt for a given user query is determined. This project leverages the utility of cosine similarity for embedding comparison, allowing efficient routing between prompts tailored for distinct domains. This report walks through the system's architecture, code implementation, and usage examples. 

## Logical Routing for Query Handling

### Data Model
The `RouteQuery` class ensures that the query will be routed to one of the predefined data sources, _python_docs_, _js_docs_, or _literature_docs_:

```
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["python_docs", "js_docs", "literature_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )
```

### LLM with Function Call
The `ChatGroq` LLM is configured to generate structured outputs based on the `RouteQuery` schema:

```
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
```

### Prompt Design
The prompt is defined with a system message providing context to the LLM:

```
system = """You are an expert at routing a user question to the appropriate data source.
Based on the programming language the question is referring to, route it to the relevant data source."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
```

### Router Definition
Combines the prompt and structured LLM to create a router:

```
router = prompt | structured_llm
```

### Example Queries
Queries are routed, and the results are printed:

```
question1 = """Which paper answers the question:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result1 = router.invoke({"question": question1})
print(result1)
print(result1.datasource)
```

```
datasource='literature_docs'
literature_docs
```

```
question2 = """Why is my code giving me errors:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""
result2 = router.invoke({"question": question2})
print(result2)
print(result2.datasource)
```

```
datasource='python_docs'
python_docs
```

### Routing Logic
A `choose_route` function dynamically selects the appropriate chain for a given result:

```
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here
```

## Semantic Routing for Query Handling

#### Prompt Creation
Two distinct prompts were defined:
     - *ML Prompt*:
       - Focus: Large Language Models (LLMs).
       - Traits: Concise, clear explanations.
     - *Medicine Prompt*:
       - Focus: Medical queries.
       - Traits: Extensive medical knowledge, experience.

```
ML_template = """You are a an expert in Machine learning.
You are great at answering questions about large language models (LLM) in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know."""

medicine_template = """You are a very good physician.
You are great at answering medical questions. You are so good because you have years of experience and
know human body very well and have a broad knowledge about illnesses and their symptoms. 
When you don't know the answer to a question you admit that you don't know."""
```

#### Embedding Prompts:

```
embeddings = SentenceTransformerEmbeddings()
prompt_templates = [ML_template, medicine_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)
```

#### Routing Function:
   - Embedded the input query and computed the cosine similarity between the query and the prompt embeddings.
   - Selected the prompt with the highest similarity score.

```
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using Machine Learning" if most_similar == ML_template else "Using Medicine")
    return PromptTemplate.from_template(most_similar)
```

#### Execution Chain:
   - Combined all steps into a seamless process to handle and route the query efficiently.

```
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatGroq()
    | StrOutputParser()
)
```

#### Demonstration
The output generated by semantic routing code involves invoking the right processes for different questions to provide context-appropriate responses. When the input _"How does RAG improve the ability of language models?"_ is processed, the code effectively uses information from the Machine Learning domain to deliver a response. It explains that RAG boosts language models' capabilities by enhancing factual accuracy and coherence. This is achieved by merging traditional language models with an information retrieval system that allows access to updated and pertinent information, making the generated text more reliable and contextually appropriate.

Conversely, when the input _"I've been experiencing persistent fatigue and headaches, along with unexplained weight loss and increased thirst and urination. What is the underlying cause?"_ is processed, the code switches context to the Medicine domain. Here, the response emphasizes the importance of consulting a healthcare professional while suggesting several potential underlying causes. The code ensures that the responses in both scenarios are highly relevant and supportive, showcasing its ability to adapt based on the input context by utilizing semantic routing.

```
Using Machine Learning
RAG, which stands for Retrieval-Augmented Generation, is a technique that combines a traditional language model with an information retrieval system. This approach improves the ability of language models in two main ways:

1. Factual accuracy: By incorporating an information retrieval component, RAG can access and incorporate up-to-date information from a large corpus of documents, reducing the reliance on the language model's internal knowledge, which may be outdated or factually incorrect.

2. Coherence and relevance: RAG can generate more coherent and relevant responses by conditioning the language model on the most relevant documents retrieved from the corpus. This helps ensure that the generated text is grounded in the provided context and focused on the topic at hand.

In summary, RAG improves the ability of language models by enhancing their factual accuracy, coherence, and relevance through the integration of an information retrieval system.
```

```
Using Medicine
I am an AI and while I can provide some information based on the symptoms you have described, I would highly recommend that you consult with a healthcare professional for an accurate diagnosis. The symptoms you have mentioned can be associated with several medical conditions, including:

1. Diabetes: Increased thirst and urination, along with unexplained weight loss, can be symptoms of diabetes.
2. Thyroid disorders: Fatigue and headaches can be symptoms of thyroid issues such as hypothyroidism or hyperthyroidism.
3. Anemia: Persistent fatigue and unexplained weight loss can be indicators of anemia.
4. Anxiety or depression: These conditions can cause fatigue, headaches, and increased thirst.
5. Infection: Persistent fatigue, headaches, and increased thirst and urination can be symptoms of an underlying infection.

This list is not exhaustive, and other medical conditions may also present with these symptoms. It is important to consult with a healthcare professional who can evaluate your symptoms, perform any necessary tests, and provide an accurate diagnosis and treatment plan.
```

#### Conclusion
This project successfully demonstrates an automated query routing system leveraging LLM capabilities. The system efficiently directs queries to appropriate data sources, showcasing the potential of LLMs in enhancing information retrieval processes. In the second section, it is shown how to efficiently direct user queries to the appropriate domain expert prompt based on semantic similarity, ensuring accurate and contextually relevant responses. 

