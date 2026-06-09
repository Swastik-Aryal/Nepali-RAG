from utils.retrieval import Retriever
from utils.rag import RAGChain
from utils.llm import get_llm

retriever = Retriever(collection_name="doc_a1b2c3d4e5f6")
llm = get_llm()
chain = RAGChain(retriever=retriever, llm=llm)

result = chain.run(question="नेपालको संविधानमा के लेखिएको छ?")
print(result["answer"])