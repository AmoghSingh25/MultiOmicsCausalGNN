import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import wikipedia

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langsmith import Client

from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Mapping
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
import logging
from langgraph import config
from tqdm import tqdm


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
config.draw_graph = False
load_dotenv()


class State(TypedDict):
    question: str
    context: str
    answer: str
    context_dict: Mapping[str, List[Document]]
    vars_data: List[str]


class LLM_RAG:
    def __init__(
        self,
        model_id="meta-llama-3.1-8b-instruct",
        base_url="http://127.0.0.1:1234/v1",
        load_vector=False,
        load_path="",
        temperature=0.7,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-embeddinggemma-300m-qat",
            api_key="lm_studio",
            base_url="http://127.0.0.1:1234/v1/",
            check_embedding_ctx_length=False,
        )
        self.stored_docs = set()

        if load_vector:
            self.load_memory_vector(load_path)
        else:
            self.vector_store = InMemoryVectorStore(self.embeddings)

        self.client = Client(api_key=os.environ["LANGCHAIN_HUB_API_KEY"])
        self.prompt = self.client.pull_prompt("rlm/rag-prompt")

        self.example_messages = self.prompt.invoke(
            {"context": "(context goes here)", "question": "(question goes here)"}
        ).to_messages()

        assert len(self.example_messages) == 1

        self.llm = init_chat_model(
            model_id,
            model_provider="openai",
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature,
        )

        self.context_dict = {}

        graph_builder = StateGraph(State).add_sequence(
            [self.retrieve, self.prepare_context, self.generate]
        )
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def retrieve(self, state: State):
        return {"context_dict": self.context_dict}

    def prepare_context(self, state: State):
        docs_content = ""
        for i in state["vars_data"]:
            docs_content += "\n\n".join(
                doc.page_content for doc in state["context_dict"][i]
            )

        return {"context": docs_content}

    def generate(self, state: State):
        messages = self.prompt.invoke(
            {"question": state["question"], "context": state["context"]}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def get_wiki_page_and_store(self, desc, col_name="", text=None):
        if text is not None:
            content = text
        else:
            try:
                res = wikipedia.search(desc)
                content = wikipedia.page(res[0]).content
            except Exception:
                try:
                    res = wikipedia.search(col_name)
                    content = wikipedia.page(res[0]).content
                except Exception:
                    return
        split_content = self.text_splitter.split_text(content)
        self.vector_store.add_texts(texts=split_content)
        self.stored_docs.add(content)

    def run_rag_query(self, query, vars_data):
        # q = "What is the importance of protein - "
        for i in tqdm(vars_data):
            # retrieved_docs = self.vector_store.similarity_search(q + str(i))
            if self.context_dict.get(i) is None:
                self.context_dict[i] = self.vector_store.similarity_search(i)
            else:
                self.context_dict[i].extend(self.vector_store.similarity_search(i))

        result = self.graph.invoke({"question": query, "vars_data": vars_data})
        return result

    def run_query(self, query):
        result = self.graph.invoke({"question": query, "vars_data": []})
        return result

    def save_memory_vector(self, save_path):
        self.vector_store.dump(save_path)

    def load_memory_vector(self, save_path):
        self.vector_store = InMemoryVectorStore(self.embeddings).load(
            embedding=self.embeddings, path=save_path
        )
