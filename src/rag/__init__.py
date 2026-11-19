"""The RAG module (Task 4).

The stages, in the order data moves through them:

* :mod:`src.rag.chunker`      - cut normalised documents into passages
* :mod:`src.rag.embedder`     - embed passages locally (fastembed / ONNX)
* :mod:`src.rag.vector_store` - store and search the vectors in Chroma
* :mod:`src.rag.indexer`      - the write path: MongoDB -> chunks -> Chroma
* :mod:`src.rag.retriever`    - the retrieval function (vector/keyword/hybrid)
* :mod:`src.rag.llm`          - two providers: Claude, and a free extractive one
* :mod:`src.rag.augment`      - combine retrieved text with generated output
* :mod:`src.rag.report`       - command-line report over the index

The two entry points most callers want::

    from src.rag.retriever import retrieve
    from src.rag.augment import answer_question
"""
