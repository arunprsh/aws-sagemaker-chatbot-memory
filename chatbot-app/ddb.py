from boto3.dynamodb.conditions import Key
import logging
import boto3
import time
import uuid


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('log')

client = boto3.resource('dynamodb')


def add_conversation_turn(table, session_id, user, bot):
    timestamp = int(time.time() * 1000)
    table.put_item(
        Item={
            'session_id': session_id,
            'timestamp': timestamp,
            'Me': user,
            'AI': bot
        }
    )


def get_conversations_by_session_id(table, session_id, descending=True):
    response = table.query(
        KeyConditionExpression=Key('session_id').eq(session_id),
        ScanIndexForward=descending
    )
    return response['Items']


def delete_conversation(table, session_id, timestamp):
    table.delete_item(
        Key={
            'session_id': session_id,
            'timestamp': timestamp
        }
    )


def create_session(table):
    session_id = str(uuid.uuid4())
    start_time = int(time.time() * 1000)
    table.put_item(
        Item={
            'session_id': session_id,
            'start_time': start_time,
            'end_time': None,
            'num_turns': 0
        }
    )
    return session_id


def end_session(table, session_id):
    end_time = int(time.time() * 1000)
    start_time_response = table.get_item(
        Key={'session_id': session_id}
    )
    start_time = start_time_response['Item']['start_time']
    num_turns = len(get_conversations_by_session_id(table, session_id))
    conversation_duration = end_time - start_time  # Compute duration in milliseconds

    table.update_item(
        Key={'session_id': session_id},
        UpdateExpression="SET end_time = :end_time, num_turns = :num_turns, conversation_duration = "
                         ":conversation_duration",
        ExpressionAttributeValues={
            ':end_time': end_time,
            ':num_turns': num_turns,
            ':conversation_duration': conversation_duration
        }
    )


if __name__ == '__main__':
    # Start a new session
    table_name = 'sessions'

    # Get the table instance
    table_ = client.Table(table_name)
    session_id_ = create_session(table_)

    # Add conversation turns
    table_name = 'conversations'
    table_ = client.Table(table_name)
    add_conversation_turn(table_, session_id_, 'hi', 'hello')
    add_conversation_turn(table_, session_id_, 'how are you?', 'i am fine')
    add_conversation_turn(table_, session_id_, 'what is the definition of court defamation?',
                          'Court defamation is a type of '
                          'civil wrong.')

    # End the session
    table_name = 'sessions'
    table_ = client.Table(table_name)
    end_session(table_, session_id_)
