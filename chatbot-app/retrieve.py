from requests.auth import HTTPBasicAuth
import datetime
import requests
import logging
import boto3
import yaml
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('log')


with open('./config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

os_username = config['opensearch']['credentials']['username']
os_password = config['opensearch']['credentials']['password']
domain_endpoint = config['opensearch']['domain']['endpoint']
text_embedding_model_endpoint_name = config['jumpstart']['text_embed_endpoint_name']
CONTENT_TYPE = 'application/json'

sagemaker_client = boto3.client('runtime.sagemaker')


def encode_query(query: str) -> list:
    payload = {'text_inputs': [query]}
    payload = json.dumps(payload).encode('utf-8')
    response = sagemaker_client.invoke_endpoint(EndpointName=text_embedding_model_endpoint_name,
                                                ContentType='application/json',
                                                Body=payload)
    body = json.loads(response['Body'].read())
    embedding = body['embedding'][0]
    return embedding


def get_es_query(embedding: list, k) -> dict:
    query = {
        'size': k,
        'query': {
            'knn': {
                'embedding': {
                    'vector': embedding,
                    'k': k
                }
            }
        }
    }
    return query


def retrieve_top_matching_passages(query: str, index: str) -> list:
    passages = []
    embedding = encode_query(query)
    query = get_es_query(embedding, 3)
    url = f'{domain_endpoint}/{index}/_search'
    response = requests.post(url, auth=HTTPBasicAuth(os_username, os_password), json=query)
    response_json = response.json()
    hits = response_json['hits']['hits']
    for hit in hits:
        # score = hit['_score']
        passage = hit['_source']['passage']
        doc_id = hit['_source']['doc_id']
        passage_id = hit['_source']['passage_id']
        passages.append([passage, doc_id, passage_id])
    return passages


def retrieve_top_matching_past_conversations(query: str, index: str) -> list:
    past_conversations = {}
    embedding = encode_query(query)
    query = get_es_query(embedding, 3)
    url = f'{domain_endpoint}/{index}/_search'
    response = requests.post(url, auth=HTTPBasicAuth(os_username, os_password), json=query)
    response_json = response.json()
    hits = response_json['hits']['hits']

    for hit in hits:
        # score = hit['_score']
        conversation_summary = hit['_source']['conversation_summary']
        created_at_ms = hit['_source']['created_at']
        created_at = datetime.datetime.fromtimestamp(int(created_at_ms) / 1000.0)
        created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')
        date, time = created_at.split(' ')
        # session_id = hit['_source']['session_id']
        summary = f'[{date}][{time}] {conversation_summary}'
        past_conversations[int(created_at_ms)] = summary

    sorted_past_conversations = {}
    for key in sorted(past_conversations.keys()):
        sorted_past_conversations[key] = past_conversations[key]

    sorted_conversations = list(sorted_past_conversations.values())
    sorted_conversations.reverse()
    return sorted_conversations


if __name__ == '__main__':
    matches = retrieve_top_matching_past_conversations('court defamation', 'conversations')
    logger.info(matches)
    matches = retrieve_top_matching_passages('court defamation', 'legal-passages')
    logger.info(matches)
