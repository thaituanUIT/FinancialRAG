USE_FLASH_ATTENTION=1

import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
import os


# Loading .env file
load_dotenv()


# API keys
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



# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

# # model_name = "VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge"
# model_name = "Viet-Mistral/Vistral-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)
# model.eval()
# tokenizer.bos_token_id = 1
# stop_token_ids = [0]

# generation_config = dict(
#         temperature=0.2,
#         top_k=3,
#         top_p=0.9,
#         do_sample=True,
#         num_beams=4,
#         repetition_penalty=1.2,
#         max_new_tokens=510, 
#         early_stopping=True,
#     )



# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="cuda",
#     temperature=0.2,
#     top_k=1,
#     top_p=0.9,
#     do_sample=True,
#     num_beams=4,
#     repetition_penalty=1.2,
#     max_new_tokens=150, 
#     early_stopping=True,
# )

prompt_template = lambda query_text, context: f"""
<s>[INST] <<SYS>>\n
You are a helpful AI assistant for courses recommendations for University Of Information Technology In VietNam
Context information is below:

{context}

Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'

\n<</SYS>>\n\n
Query: {query_text}

Answer:

[/INST] 
"""


# QA Langchain
llm = HuggingFacePipeline(pipeline=pipeline)
db_retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=db_retriever, 
    verbose=True

)



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
        result = db.similarity_search_with_score(query_text, k=3)
        
        print(result)
        # context = ""
        # if(len(result) == 0):
        #     result = [["I Apoligise I don't have any information about the question", 0]]
        # elif(result[0][1] < 0.3):
        #     result = [["I Apoligise I don't have any information about the question", 0]]

        # context = result[0][0].page_content
        # prompt = prompt_template(query_text=query_text,context=context)

        # print(prompt)
        # result = qa.invoke(prompt)
        # print(docs)
        # print(result)

        # output = pipeline(prompt, eos_token_id=tokenizer.eos_token_id,**generation_config)
        # print(output)

        # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # output = model.generate(input_ids=inputs["input_ids"], eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **generation_config)
        # print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])

        # context = "\n\n ------ \n\n".join(doc.page_content for doc, score in result)
        


    elif(choice != 1 and choice != 0):
        print("False to process please choose again !!!\n")

