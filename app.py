USE_FLASH_ATTENTION=1

import gradio as gr
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.llms import HuggingFacePipeline
from openai import conversations
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
import os

# Loading .env file
load_dotenv()

from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

CHROMA_PATH = "chroma_database"

model_name="hiieu/halong_embedding"
model_kwargs = {
    'device': 'cuda',
    'trust_remote_code':True
}


hf = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs)

db = Chroma(persist_directory=CHROMA_PATH, embedding_function= hf)


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)
model.eval()
tokenizer.bos_token_id = 1
stop_token_ids = [0]

generation_config = dict(
        temperature=0.2,
        top_k=3,
        top_p=0.9,
        do_sample=True,
        num_beams=4,
        repetition_penalty=1.2,
        max_new_tokens=510, 
        early_stopping=True,
    )

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="cuda",
    temperature=0.2,
    top_k=1,
    top_p=0.9,
    do_sample=True,
    num_beams=4,
    repetition_penalty=1.2,
    max_new_tokens=150, 
    early_stopping=True,
)

# prompt_template = lambda query_text, context: f"""

# Bạn là một trợ lí tài chính Tiếng Việt nhiệt tình và trung thực. 
# Hãy luôn trả lời một cách hữu ích nhất có thể, 
# đồng thời giữ an toàn dựa trên thông tin bên dưới:\n

# {context}

# Query: {query_text}

# Answer:

# """


# def chat_function(message, history, system_prompt, max_new_tokens, temperature):
#     system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
#     system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
#     result = db.similarity_search_with_score(message, k=3)
#     context = ""
#     context = result[0][0].page_content
#     prompt = prompt_template(query_text=message, context=context)


#     conversation = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": prompt},
#     ]

#     chat_template = tokenizer.apply_chat_template(conversation=conversation, tokenize=False)
#     output = pipeline(chat_template, eos_token_id=tokenizer.eos_token_id,**generation_config)

#     return output[0]['generated_text'][len(chat_template):]

from prompting import context_aware_prompt, new_system_prompt

def chat_function(message, history, system_prompt, max_new_tokens, temperature):
    systemPrompt = new_system_prompt(system_prompt)
    
    res = db.similarity_search_with_score(message, k=3)
    context = "\n".join([doc[0].page_content for doc in res])
    
    conv_hist = ""
    if history:
        for turn in history[-3:]:
            conv_hist += f"User: {turn[0]}\nAssistance: {turn[1]}\n"
            
    userPrompt = context_aware_prompt(
        query_text=message,
        context=context,
        conv_history=conv_hist
    )
    
    conversation = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt},
    ]
    
    chat_template = tokenizer.apply_chat_template(conversation=conversation, tokenize=False)
    output = pipeline(chat_template, eos_token_id=tokenizer.eos_token_id,**generation_config)
    
    return output[0]['generated_text'][len(chat_template):]

with gr.Blocks() as app:
    gr.ChatInterface(
        fn=chat_function,
        textbox = gr.Textbox(placeholder="Nhập Tin Nhắn Vào Đây", container=False, scale= 8),
        chatbot = gr.Chatbot(height=400, type="messages"),
        additional_inputs=[
            gr.Textbox("Bạn là một trợ lí tài chính Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n", label = "System Prompt"),
            gr.Slider(150, 500, label="Max New Tokens"),
            gr.Slider(0,1,label="Temperature")
        ]
    )

app.launch(share=True)
