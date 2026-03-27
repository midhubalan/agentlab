# Core Abstractions in LangChain


## The layering 

**LangGraph** is the *lowest* level — it's the runtime and orchestration engine. It knows about graphs, state, nodes, edges, checkpointers, and stores. It does *not* know about LLMs, prompts, or retrieval.

**LangChain** sits *above* LangGraph. It provides the LLM-specific abstractions (chat models, tools, prompts, retrievers, vector stores) and convenience functions like `create_agent()` that internally build a LangGraph graph for you. When you call `create_agent()`, you get back a compiled LangGraph `StateGraph` — you just didn't have to wire the nodes and edges yourself.

**DeepAgents** sits at the *top*. It wraps both LangChain's model/tool abstractions and LangGraph's orchestration, adding planning, filesystem context, and sub-agent spawning.

So the stack is: **LangGraph (runtime) → LangChain (LLM abstractions) → DeepAgents (agent harness)**. Each layer depends on the ones below it.

## The interfaces, organized by layer

### 1. The Runnable protocol — the universal spine

This is the single most important interface in the ecosystem. It lives in `langchain-core` but it's the contract that *everything* honors across all three layers.

A `Runnable` is anything that takes an input, does something, and returns an output. The interface defines:

- `invoke(input)` / `ainvoke(input)` — single call
- `stream(input)` / `astream(input)` — streaming output
- `batch(inputs)` / `abatch(inputs)` — parallel calls

**Why this matters across layers:** A LangChain prompt template is a Runnable. A chat model is a Runnable. A retriever is a Runnable. A compiled LangGraph `StateGraph` is a Runnable. A DeepAgents agent is a Runnable. Because they all share this interface, you can compose any of them with LCEL's pipe operator (`|`), pass any of them as a node in a LangGraph graph, or nest any of them inside a DeepAgents workflow. It's the universal plug-and-socket.

### 2. LangChain-core interfaces (the LLM abstraction layer)

These are the interfaces that provider packages implement. Each one extends `Runnable` unless noted otherwise.

**`BaseChatModel`** (extends Runnable) — The chat model interface. Input: messages. Output: an `AIMessage`. Key methods beyond the Runnable basics:
- `bind_tools(tools)` — returns a new Runnable that knows about available tools
- `with_structured_output(schema)` — returns a Runnable that parses output into a Pydantic model or TypedDict

Providers implement this: `ChatOpenAI`, `ChatAnthropic`, `ChatGroq`, etc. LangGraph doesn't care about this interface directly — it just sees a Runnable. DeepAgents' `create_deep_agent()` takes a `BaseChatModel` as its model argument.

**`Embeddings`** — Notably, this one does *not* extend Runnable. It's a simpler interface:
- `embed_query(text) → list[float]`
- `embed_documents(texts) → list[list[float]]`

Used by vector stores under the hood. Providers implement this: `OpenAIEmbeddings`, `CohereEmbeddings`, etc. You can initialize via `init_embeddings("openai:text-embedding-3-small")`.

**`VectorStore`** — Also does not extend Runnable directly. Key methods:
- `add_documents(docs)` — index documents
- `similarity_search(query, k=4)` — find similar docs
- `as_retriever(**kwargs)` → returns a `BaseRetriever` (which *is* a Runnable)

This is where the interface boundary is interesting: `VectorStore` itself isn't composable via LCEL, but the moment you call `.as_retriever()`, you get a Runnable that plugs into chains and graphs. Providers: `Chroma`, `PGVector`, `PineconeVectorStore`, etc.

**`BaseRetriever`** (extends Runnable) — Input: a string query. Output: `list[Document]`. This is the Runnable-compatible face of any retrieval system. The basic one wraps a vector store, but fancier ones exist: `MultiQueryRetriever`, `EnsembleRetriever`, `SelfQueryRetriever`, `ContextualCompressionRetriever`. These compose *other* retrievers and models. Critically, because retrievers are Runnables, you can use them as nodes in LangGraph or as tools in a DeepAgents workflow.

**`DocumentLoader`** — Not a Runnable. Simple interface:
- `load() → list[Document]`
- `lazy_load() → Iterator[Document]`

Implementations: `PyPDFLoader`, `CSVLoader`, `WebBaseLoader`, `UnstructuredHTMLLoader`, etc. These are typically used at indexing time, not at query time, so they don't need to be Runnables.

**`TextSplitter`** — Not a Runnable. Takes documents, returns smaller documents:
- `split_documents(docs) → list[Document]`
- `split_text(text) → list[str]`

Implementations: `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, etc. Again, indexing-time concern.

**`BaseTool`** (extends Runnable) — The tool interface. This one is critical because it's the bridge between LangChain and LangGraph. A tool has:
- A `name` and `description` (used by the LLM to decide when to call it)
- An `args_schema` (Pydantic model defining the input)
- `invoke(input)` — actually runs the tool

You typically create tools with the `@tool` decorator rather than subclassing. **Cross-layer relevance:** LangChain's `bind_tools()` passes these to the model. LangGraph's tool-calling nodes execute them. DeepAgents adds its own tools (`write_todos`, `task`, filesystem tools) that conform to this same interface, so they sit alongside your custom tools seamlessly.

**`BasePromptTemplate`** (extends Runnable) — Input: a dict of variables. Output: a prompt value (messages). `ChatPromptTemplate` is the main implementation you'll use. Because it's a Runnable, you can chain it directly: `prompt | model | parser`.

**`BaseOutputParser`** (extends Runnable) — Input: the raw model output. Output: parsed/structured data. Implementations: `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`. The tail end of most LCEL chains.

### 3. Core data types (shared vocabulary)

These aren't interfaces in the OOP sense, but they're the data contracts that flow between all the interfaces above.

**`Document`** — has `page_content` (str) and `metadata` (dict). This is what loaders produce, splitters consume and produce, vector stores index, and retrievers return. It flows from LangChain's retrieval layer into LangGraph state and into DeepAgents' filesystem.

**`BaseMessage`** and subclasses — `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`. These are the lingua franca for chat. LangChain models consume and produce them, LangGraph state typically stores them as `list[BaseMessage]`, and DeepAgents passes them through its agent loop.

### 4. LangGraph-level interfaces

**`BaseCheckpointSaver`** — The checkpointer interface. Implementations: `MemorySaver` (in-memory), `PostgresSaver`, etc. This is purely a LangGraph concern, but because LangChain agents compile to LangGraph graphs, you pass a checkpointer at compile time and it works transparently. DeepAgents also accepts a checkpointer at `create_deep_agent()` time.

**`BaseStore`** — The memory store interface (key-value, with namespace support). Used for long-term memory across conversations. Implementations: `InMemoryStore`, `PostgresStore`. This is a LangGraph concept, but DeepAgents specifically uses it via `StoreBackend` — that's how DeepAgents' filesystem can persist across sessions.

### 5. DeepAgents-level interfaces

**Filesystem backends** — `StateBackend`, `FilesystemBackend`, `LocalShellBackend`, `StoreBackend`, `CompositeBackend`. These are DeepAgents-specific. They share a common interface for reading/writing "files" that the agent uses as working context. The `StoreBackend` is where it bridges down to LangGraph's `BaseStore`.

### The cross-cutting picture

Here's how a single request flows through the interfaces across all three layers:

User message → **`BaseMessage`** (shared) → DeepAgents agent loop → calls **`BaseChatModel`** (LangChain) with **`BaseTool`** bindings (LangChain) → model decides to call a retrieval tool → tool calls **`BaseRetriever`** (LangChain) which queries a **`VectorStore`** (LangChain) → returns **`Document`**s (shared) → agent writes findings to filesystem backend (DeepAgents) → state is checkpointed via **`BaseCheckpointSaver`** (LangGraph) → agent continues planning with the next tool call.

