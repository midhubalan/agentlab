# LangChain Providers

**A "provider" in LangChain is essentially a vendor integration package.** Each provider corresponds to one external company or service (OpenAI, Anthropic, Google, Cohere, Pinecone, Chroma, etc.) and ships as its own installable Python package following the naming convention `langchain-{provider}` — e.g., `langchain-openai`, `langchain-anthropic`, `langchain-chroma`.

**What's inside a provider package?** Wrapper classes that implement LangChain's standard interfaces for whatever that vendor offers. So `langchain-openai` gives you a `ChatOpenAI` class (implements the `BaseChatModel` interface), an `OpenAIEmbeddings` class (implements `Embeddings`), and so on. `langchain-chroma` gives you a `Chroma` class that implements the `VectorStore` interface.

The key insight is that LangChain defines **abstract interfaces** at the core level (`BaseChatModel`, `Embeddings`, `VectorStore`, `BaseRetriever`, `DocumentLoader`, etc.), and providers are the concrete implementations. This is a classic adapter/strategy pattern — your application code programs against the interface, and the provider package handles the vendor-specific API calls, auth, serialization, and quirks.

**Why separate packages?** Two practical reasons. First, dependency isolation — you don't want to install every vendor's SDK just to use one model. Second, versioning independence — the OpenAI wrapper can ship updates when OpenAI changes their API without waiting for a core LangChain release.

**The `init_chat_model` shortcut** is where this really pays off. Instead of importing a specific provider class, you can write:

```python
from langchain.chat_models import init_chat_model

# Provider is inferred from the model name, or you can be explicit
llm = init_chat_model("gpt-4o", model_provider="openai")
# or
llm = init_chat_model("claude-sonnet-4-20250514", model_provider="anthropic")
```

Both return an object that satisfies the same `BaseChatModel` interface, so all downstream code (chains, agents, tools) works identically regardless of which provider you chose. Same idea applies with `init_embeddings` for embedding models.

**One thing to watch for:** there's a legacy `langchain-community` package that used to hold all the provider integrations in one place. That's being phased out in favor of the individual `langchain-{provider}` packages. If you see imports from `langchain_community`, that's the old pattern — prefer the dedicated provider package when one exists.

