import logging
import sys
import os

from llama_index.core import ComposableGraph
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.node_parser import SentenceSplitter

from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ['MONGO_URI']

reader = SimpleDirectoryReader("data/")
documents = reader.load_data()


parser = SentenceSplitter()

nodes = parser.get_nodes_from_documents(documents)

storage_context = StorageContext.from_defaults(
    docstore=MongoDocumentStore.from_uri(uri=MONGO_URI),
    index_store=MongoIndexStore.from_uri(uri=MONGO_URI),
)

storage_context.docstore.add_documents(nodes)

summary_index = SummaryIndex(nodes, storage_context=storage_context)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)