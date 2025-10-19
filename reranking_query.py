USE_FLASH_ATTENTION=1

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
import os


# Loading .env file
load_dotenv()


# API keys
from huggingface_hub import login
import cohere

login(token=os.getenv("HF_TOKEN"))
co = cohere.Client(os.getenv("COHERE_API_TOKEN"))


reranker_name = "rerank-multilingual-v2.0"


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

# model_name = "VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge"
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

prompt_template = lambda query_text, context: f"""
<s>[INST] <<SYS>>\n
Bạn là một trợ lí tài chính Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn dựa trên thông tin bên dưới:\n

{context}

Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.
Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
\n<</SYS>>\n\n
Query: {query_text}

Answer:

[/INST] 
"""




# result = qa.run()
# print(result)


def return_results(results, documents):    
    for idx, result in enumerate(results.results):
        print(f"Rank: {idx+1}") 
        print(f"Score: {result.relevance_score}")
        print(f"Document: {documents[result.index]}\n")
    


choice  = 999

while(choice):

    print("+------------------------------------------------+")
    print("|                                                |")
    print("| 1: Make Question                               |")
    print("| 0: End Conversation                            |")
    print("|                                                |")
    print("+------------------------------------------------+\n")

    choice = input()
    choice = int(choice)
    if(choice == 1):
        print("Enter query: \n")
        query_text = input()
        result = db.similarity_search_with_score(query_text, k=10)
        docs = []
        for doc in result:
            # print(doc)
            docs.append(doc[0].page_content)
        rerank_result = co.rerank(model=reranker_name, query=query_text, documents=docs, top_n=3,return_documents=True)
        context = ""
        for doc in rerank_result.results:
            context += docs[doc.index]
            context += "\n"
        
    
        prompt = prompt_template(query_text=query_text,context=context)

        print(prompt)

        

    elif(choice != 1 and choice != 0):
        print("False to process please choose again !!!\n")

