from retrieve import retrieve_top_matching_past_conversations
from llm import summarize_passages_and_collate_answers
from retrieve import retrieve_top_matching_passages
from ddb import get_conversations_by_session_id
from llm import generate_dialogue_response
from ddb import add_conversation_turn
from ddb import create_session
from ddb import end_session
from llm import detect_task
import streamlit as st
import logging
import boto3


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('log')


# Set Streamlit page configuration
st.set_page_config(page_title='ai-assistant', layout='wide')

dynamodb = boto3.resource('dynamodb')

# Initialize session states
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = ''
if 'stored_session' not in st.session_state:
    st.session_state['stored_session'] = []


def get_text_input():
    """
    Get the user inputted text.
    :return: Text entered by the user
    """
    text_input = st.text_input('You: ',
                               st.session_state['input'],
                               key='input',
                               placeholder='Your AI assistant here! Ask me anything ...',
                               label_visibility='hidden')
    return text_input


def new_session():
    """
    Clears session state and starts a new session.
    """
    # End current session and update sessions table in DynamoDB
    table_name = 'sessions'
    table = dynamodb.Table(table_name)
    end_session(table, st.session_state.session_id)

    save = []
    for j in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append(f"User: {st.session_state['past'][j]}")
        save.append(f"Bot: {st.session_state['generated'][j]}")

    st.session_state['stored_session'].append(save)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ''


# Set up sidebar with various options
with st.sidebar.expander('🛠️', expanded=True):
    max_turns = st.number_input('Number of turns to remember',
                                min_value=1,
                                max_value=100)

# Set up the Streamlit app layout
st.title('🤖 AI Assistant 🧠')
st.subheader('Powered by ☁️ AWS')

# Add a button to start a new chat
st.sidebar.button('New Session', on_click=new_session, type='primary')

# Get the user input
user_input = get_text_input()

sessions_table = dynamodb.Table('sessions')
conversations_table = dynamodb.Table('conversations')


def respond_by_task(query, history):
    logger.info(f'HISTORY: {history}')
    task_type = detect_task(query)
    logger.info(f'TASK TYPE = {task_type}')
    completion = None
    if task_type == 'STM CHAT':
        if len(history) > 0:
            prompt = f"""{history}
Me: {user_input}
AI:"""
            logger.info(f'Prompt: {prompt}')
            completion = generate_dialogue_response(prompt)
        else:
            prompt = f"""Me: {user_input}
AI:"""
            logger.info(f'Prompt: {prompt}')
            completion = generate_dialogue_response(prompt)
    elif task_type == 'LTM PAST CONVERSATIONS':
        completion = retrieve_top_matching_past_conversations(user_input, 'conversations')
        completion = '\n\n'.join(completion)
    elif task_type == 'LTM VERIFIED SOURCES':
        completion = retrieve_top_matching_passages(user_input, 'passages')
        completion = summarize_passages_and_collate_answers(completion, user_input)
    return completion


def transform_ddb_past_history(history: list, num_turns=10) -> str:
    past_hist = []
    for turn in history:
        me_utterance = turn['Me']
        bot_utterance = turn['AI']
        past_hist.append(f'Me: {me_utterance}')
        past_hist.append(f'AI: {bot_utterance}')
    past_hist = past_hist[-num_turns*2:]
    past_hist_str = '\n'.join(past_hist)
    return past_hist_str


if user_input:
    user_utterance = st.session_state['input']
    ai_utterance = st.session_state['generated']
    if len(ai_utterance) == 0:
        # Start a new session
        st.session_state.session_id = create_session(sessions_table)

    past_history = get_conversations_by_session_id(conversations_table, st.session_state.session_id)
    past_history = transform_ddb_past_history(past_history, max_turns)
    output = respond_by_task(user_input, past_history)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    ai_utterance = st.session_state['generated'][-1]
    add_conversation_turn(conversations_table, st.session_state.session_id, user_utterance, ai_utterance)

# Display the conversation history using an expander, and allow the user to download it
download_str = []
with st.expander('Conversation', expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state['past'][i], icon='🧐')
        st.success(st.session_state['generated'][i], icon='🤖')
        download_str.append(f"Human: {st.session_state['past'][i]}")
        download_str.append(f"AI: {st.session_state['generated'][i]}")

    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f'Conversation Session:{i}'):
        st.write(sublist)


def del_sessions():
    del st.session_state.stored_session


# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    st.sidebar.button('Clear All', on_click=del_sessions, type='primary')
