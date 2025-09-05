import os
from retriever import hybrid_search
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Load Azure credentials
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Azure client
client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version="2024-06-01")

def build_prompt(question, context_chunks, chat_history=[], max_context_chars=8000):
    """
    Build a robust prompt that:
    - Uses as much relevant context as possible (within token limits)
    - Falls back to medical knowledge if context is insufficient
    - Encourages citations
    - Includes chat history for multi-turn conversations
    """
    context_text = ""
    for c in context_chunks:
        chunk_text = f"[Source: {c['meta']['source']} - chunk {c['meta']['chunk_id']}]\n{c['text']}\n\n"
        if len(context_text) + len(chunk_text) > max_context_chars:
            break
        context_text += chunk_text

    # Format the chat history for the prompt
    history_text = ""
    if chat_history:
        for turn in chat_history:
            if turn["role"] == "user":
                history_text += f"User: {turn['content']}\n"
            else:
                history_text += f"Assistant: {turn['content']}\n"
        history_text += "\n"

    prompt = f"""
You are an expert medical assistant specializing in hematology and oncology.
Use ONLY the following context and chat history to answer the question. If the information is not in the provided documents, state that and then provide a general medical knowledge-based answer.

Always cite sources from the context using this format: [Source: filename - chunk X].

Chat History:
{history_text}

Context:
{context_text}

Question: {question}

Answer:
"""
    return prompt

def get_answer(question, top_k=8, chat_history=[]):
    # Retrieve top chunks (semantic + keyword)
    chunks = hybrid_search(question, top_k=top_k)

    if not chunks:
        return "No relevant context found. Please check your documents.", []

    # Build optimized prompt with chat history
    prompt = build_prompt(question, chunks, chat_history)

    try:
        # Call Azure GPT-4o
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful and precise medical assistant. Use the provided context and chat history to answer questions about hematology and oncology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )

        answer = response.choices[0].message.content.strip()
        sources = [f"{c['meta']['source']} (chunk {c['meta']['chunk_id']})" for c in chunks]
        return answer, sources

    except Exception as e:
        return f"❌ Azure GPT call failed: {str(e)}", []

if __name__ == "__main__":
    question = "What are the side effects and complications of hematopoietic stem cell transplantation?"
    answer, sources = get_answer(question)
    print("\n🧠 Answer:\n", answer)
    print("\n📚 Sources:\n", sources)