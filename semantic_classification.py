from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv, find_dotenv
import json
from update_json import update_json_file, load_data

import logging
logging.basicConfig(filename='processing.log', level=logging.INFO)


load_dotenv()
my_api = os.environ.get("OPENAI_API_KEY")
with open("Prompt_Template.txt", 'r') as file:
    prompt_template = file.read()

call_chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-1106", openai_api_key = my_api)

data = load_data('heldout_op_data.json')

def process_item(item):
    try:
        title = item["op_title"]
        op_body = item["op_text"]

        positive_body = item['positive']['comments'][0]['body']
        negative_body = item['negative']['comments'][0]['body']

        positive_input = [SystemMessage(content=prompt_template),HumanMessage(content=positive_body),]
        negative_input = [SystemMessage(content=prompt_template),HumanMessage(content=negative_body),]

        positive_result = call_chat(positive_input)
        positive_result = json.loads(positive_result.content)
        negative_result = call_chat(negative_input)
        negative_result = json.loads(negative_result.content)
        return {
            title:
            {
            "op_body": op_body,
            "positive":{
                "body":positive_body,
                "data":positive_result
            },
            "negative":{
                "body":negative_body,
                "data":negative_result
            }
            }
        }
        #print(f"{positive_result}\n{negative_result}")
        
    except Exception as e:
        logging.error(f"Error processing item {title}: {e}")
        print(positive_result)
        print(negative_result)
        return None
        

for i, item in enumerate(data[:133],start=1):
    logging.info(f"Working on #{i}")
    try:
        '''
        if "op_title" not in item or "op_text" not in item:
            raise ValueError("Missing required fields in item")'''
        processed_data = process_item(item)
        if processed_data is not None:
            update_json_file("output.json", processed_data)
            logging.info(f"Successfully processed and updated item #{i}")
    except Exception as e:
        logging.error(f"Error in processing or updating item #{i}: {e}")
