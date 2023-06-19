import logging
import boto3
import json
import yaml


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('log')


with open('./config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

endpoint_name = config['jumpstart']['text_gen_endpoint_name']
CONTENT_TYPE = 'application/json'

client = boto3.client('sagemaker-runtime')

NUM_RETURN_SEQUENCES = 1
TOP_K = 0
TOP_P = 0.7
DO_SAMPLE = True
TEMPERATURE = 0.1


def detect_task(query: str) -> str:
    if query.startswith('\\verified') or query.startswith('/verified'):
        return 'LTM VERIFIED SOURCES'
    elif query.startswith('\\past') or query.startswith('/past'):
        return 'LTM PAST CONVERSATIONS'
    else:
        return 'STM CHAT'


def generate(prompt: str, max_length=256) -> str:
    payload = {'text_inputs': prompt,
               'max_length': max_length,
               'num_return_sequences': NUM_RETURN_SEQUENCES,
               'top_k': TOP_K,
               'top_p': TOP_P,
               'temperature': TEMPERATURE,
               'do_sample': DO_SAMPLE}
    payload = json.dumps(payload).encode('utf-8')
    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType=CONTENT_TYPE,
                                      Body=payload)
    model_predictions = json.loads(response['Body'].read())
    generated_text = model_predictions['generated_texts'][0]
    completion = generated_text.strip()
    return completion


def summarize_passages_and_collate_answers(passages: list, query: str) -> str:
    collated_answers = []
    for passage, doc_id, passage_id in passages:
        prompt = f'Passage=={passage}\n\nQuestion=={query}\n\nAnswer==\n\nGiven a passage and a question, generate ' \
                 f'a clean answer in 2 to 3 short complete sentences. '
        answer = generate(prompt, 256)
        collated_answers.append(f'{answer}\n\n[doc = {doc_id} | passage = {passage_id}]')
    collated_answers = '\n\n'.join(collated_answers)
    logger.info(f'ANSWERS: {collated_answers}')
    return collated_answers


def generate_dialogue_response(prompt: str) -> str:
    completion = generate(prompt, 256)
    logger.info(f'DIALOGUE RESPONSE: {completion}')
    return completion


if __name__ == '__main__':
    completion_ = detect_task('definition of bribery by indian law ')
    logging.info(completion_)
