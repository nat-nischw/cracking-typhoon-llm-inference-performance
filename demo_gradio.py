import gradio as gr
from txtai import Embeddings, LLM
import re
from prompt import SYSTEM_PROMPT

# Wikipedia Embeddings Index
embeddings = Embeddings()
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# LLM
llm = LLM("scb10x/llama-3-typhoon-v1.5x-8b-instruct")

# Function to process Chain of Thought reasoning
def cot(system, user, max_tokens, temperature, top_p, reflect):
    if reflect:
        system = SYSTEM_PROMPT['self_reflection'].format(SYSTEM=system)
    else:
        system = SYSTEM_PROMPT['non_self_reflection'].format(SYSTEM=system)

    response = llm(
        [
            {"role": "system", "content": system}, 
            {"role": "user", "content": user}
        ],
        maxlength=max_tokens, temperature=temperature, top_p=top_p
    )

    # Logs of thought
    logs_of_thought = f'Logs of Thought:\n{response}'
    
    # Extract the final output
    match = re.search(r"<output>(.*?)(?:</output>|$)", response, re.DOTALL)
    final_answer = match.group(1).strip() if match else response
    
    return final_answer, logs_of_thought

# Function to run CoT process using RAG
def rag(question, max_tokens, temperature, top_p, reflect):
    prompt = """
    Answer the following question using only the context below. Only include information
    specifically discussed.
    question: {question}
    context: {context}
    """

    # System prompt
    system = "You are a friendly assistant. You answer questions from users."

    # RAG context
    context = "\n".join([x["text"] for x in embeddings.search(question)])

    # RAG with CoT + Self-Reflection
    return cot(system, prompt.format(question=question, context=context), max_tokens, temperature, top_p, reflect)

# Gradio interface function
def gradio_cot_interface(question, max_tokens, temperature, top_p, reflect):
    result, logs_of_thought = rag(question, max_tokens, temperature, top_p, reflect)
    return result, logs_of_thought
    
gr.Interface(
    fn=gradio_cot_interface, 
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here"),
        gr.Slider(minimum=256, maximum=4096, step=128, value=2048, label="Max Tokens"),  # Slider for max_tokens
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Temperature"),  # Slider for temperature
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, label="Top-p"),        # Slider for top_p
        gr.Checkbox(label="Enable Self-Reflection", value=True)                          # Checkbox for reflection
    ],
    outputs=[
        gr.Textbox(label="Final Answer"),
        gr.Markdown(label="Logs of Thought")
    ],
    title="Chain of Thought + Self-Reflection + RAG",
    description="Ask any question and the model will apply Chain of Thought reasoning with self-reflection"
).launch(share=True)
