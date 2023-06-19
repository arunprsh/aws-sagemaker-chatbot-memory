from boto3.dynamodb.conditions import Key
from requests.auth import HTTPBasicAuth
import requests
import logging
import boto3
import json
import os


# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('log')

# Create service clients
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Reference SageMaker JumpStart endpoints
domain_endpoint = os.environ['OS_ENDPOINT']
domain_index = os.environ['OS_INDEX_NAME']
os_username = os.environ['OS_USERNAME']
os_password = os.environ['OS_PASSWORD']

# Reference Amazon OpenSearch endpoint
URL = f'{domain_endpoint}/{domain_index}'

# Set LLM generation configs
MAX_LENGTH = 512
NUM_RETURN_SEQUENCES = 1
TOP_K = 0
TOP_P = 0.7
DO_SAMPLE = True
CONTENT_TYPE = 'application/json'
TEMPERATURE = 0.1



def lambda_handler(event: dict, context: dict) -> None:
    logger.info(f'Received event: {event}')
    logger.info(f'Received context: {context}')

    for record in event['Records']:
        if record['eventName'] == 'MODIFY':
            session_item = record['dynamodb']['NewImage']
            session_id = session_item['session_id']['S']
            end_time = session_item['end_time']['N']

            # Query the conversations table
            conversation_turns = query_conversations_table(session_id)

            # Flatten the conversation turns into a dict
            flattened_conversations = flatten_conversations(conversation_turns)

            summary = summarize_conversations(flattened_conversations)

            # Encode the dict into an embedding
            embedding = encode_conversations(summary)

            # Write the embedding to Elasticsearch
            write_to_elasticsearch(session_id, embedding, end_time, summary)

            print(f"Session {session_id} was persisted to long term memory")


def query_conversations_table(session_id: str) -> list:
    table = dynamodb.Table('conversations')
    response = table.query(KeyConditionExpression=Key('session_id').eq(session_id))
    return response['Items']


def flatten_conversations(conversation_turns: list) -> dict:
    flattened_conversations = {'conversation': ''}
    for turn in conversation_turns:
        user_message = turn['Me']
        bot_message = turn['AI']
        flattened_conversations['conversation'] += f"{user_message} {bot_message} "
    return flattened_conversations


def summarize_conversations(conversation: str) -> str:
    logger.info('Conversation: {conversation}')
    prompt = f"""Conversation==hi there! I'm doing well, thank you. what is the meaning of eminent domain? Eminent domain is the power of the government to take private property for public use, with just compensation. 
Summary==We discussed about the meaning of eminent domain and that it is the government's power to take private property for public use with just compensation. 

Conversation==Hey! I'm feeling great, how about you? Can you tell me what is the concept of due diligence? Due diligence is a comprehensive investigation or appraisal of a business or person before entering into an agreement or transaction. 
Summary==We discussed about the meaning of due diligence and that it is a comprehensive investigation or appraisal of a business or person before entering into an agreement or transaction. 

Conversation==hello! I'm good, thank you for asking. What is the definition of fiduciary duty? Fiduciary duty is a legal obligation of one party to act in the best interests of another, often in financial or legal matters. 
Summary==We talked about the meaning of fiduciary duty and that it is a legal obligation of one party to act in the best interests of another, often in financial or legal matters. 

Conversation=={conversation}
Summary==


Summarize the above Conversation as a short paragraph in 3 to 4 sentences."""
    payload = {'text_inputs': prompt,
               'max_length': MAX_LENGTH,
               'temperature': TEMPERATURE,
               'num_return_sequences': NUM_RETURN_SEQUENCES,
               'top_k': TOP_K,
               'top_p': TOP_P,
               'do_sample': DO_SAMPLE}
    payload = json.dumps(payload).encode('utf-8')
    response = sagemaker_runtime.invoke_endpoint(EndpointName=os.environ['SAGEMAKER_TEXT_GEN_ENDPOINT'],
                                                 ContentType=CONTENT_TYPE,
                                                 Body=payload)
    model_predictions = json.loads(response['Body'].read())
    generated_text = model_predictions['generated_texts'][0]
    logger.info(f'Summary: {generated_text}')
    return generated_text


def encode_conversations(summary: str) -> list:
    payload = {'text_inputs': [summary]}
    payload = json.dumps(payload).encode('utf-8')
    response = sagemaker_runtime.invoke_endpoint(EndpointName=os.environ['SAGEMAKER_TEXT_EMBED_ENDPOINT'],
                                                 ContentType='application/json',
                                                 Body=payload)
    body = json.loads(response['Body'].read())
    embedding = body['embedding'][0]
    return embedding


def write_to_elasticsearch(session_id: str, embedding: list, end_time: int, summary: str) -> None:
    document = {
        'session_id': session_id,
        'embedding': embedding,
        'created_at': end_time,
        'conversation_summary': summary
    }
    
    try:
        response = requests.post(f'{URL}/_doc/{session_id}', auth=HTTPBasicAuth(os_username, os_password),
                                 json=document)
        if response.status_code not in [200, 201]:
            logger.error(response.status_code)
            logger.error(response.text)
    except Exception as e:
        logger.error(e)
