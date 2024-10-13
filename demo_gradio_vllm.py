import torch
from transformers import AutoTokenizer
import gradio as gr
from txtai import Embeddings
import re
from vllm import LLM, SamplingParams
from prompt import SYSTEM_PROMPT

# Wikipedia Embeddings Index
embeddings = Embeddings()
embeddings.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")

# LLM
MODEL_ID = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"
llm = LLM(model=MODEL_ID, dtype=torch.bfloat16)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Function to process Chain of Thought reasoning
def cot(system, user, max_tokens, temperature, top_p):
    # Run LLM inference using vLLM
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature, 
        top_p=top_p,
        stop='<|eot_id|>'
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT['self_reflection'].format(SYSTEM=system) },
        {"role": "user", "content": user},
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = llm.generate([formatted_prompt], sampling_params=sampling_params)
    generated_text = response[0].outputs[0].text  
    
    # Extract logs of thought
    logs_of_thought = f'Logs of Thought:\n{generated_text}'
    
    # Extract the final output using regex
    match = re.search(r"<output>(.*?)(?:</output>|$)", generated_text, re.DOTALL)
    final_answer = match.group(1).strip() if match else generated_text
    
    return final_answer, logs_of_thought

# Function to run CoT process using RAG
def rag(question, max_tokens, temperature, top_p):
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

    # RAG with CoT + Self-Reflection using vLLM
    return cot(system, prompt.format(question=question, context=context), max_tokens, temperature, top_p)


# Gradio interface function
def gradio_cot_interface(question, max_tokens, temperature, top_p):
    result, logs_of_thought = rag(question, max_tokens, temperature, top_p)
    return result, logs_of_thought

gr.Interface(
    fn=gradio_cot_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here"),
        gr.Slider(minimum=256, maximum=4096, step=128, value=2048, label="Max Tokens"),  
        gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.0, label="Temperature"), 
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, label="Top-p")     
    ],
    outputs=[
        gr.Textbox(label="Final Answer"),
        gr.Markdown(label="Logs of Thought")
    ],
    title="Chain of Thought + Self-Reflection + RAG",
    description="Ask any question and the model will apply Chain of Thought reasoning with self-reflection"
).launch(share=True)
