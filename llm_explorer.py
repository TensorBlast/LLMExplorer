import streamlit as st
from streamlit_option_menu import option_menu
import replicate
from openai import OpenAI
from langchain.llms import ollama
from collections import OrderedDict, defaultdict
import os

setter = defaultdict(None)
setter['DEFAULT_SYSTEM_PROMPT'] = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
setter['MAX_TOKENS'] = 4096

os.environ['REPLICATE_API_TOKEN'] = st.secrets['replicate']['API_KEY']
OpenAI.api_key = st.secrets.openai.API_KEY
openaiclient = OpenAI()

st.title('LLM Explorer')

model_choices = dict(replicate=['Llama-2-13b-chat', 'Llama-2-70b-chat', 'CodeLlama-13b-instruct','CodeLlama-34b-instruct'])

replicatemap = dict([('Llama-2-13b-chat', "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"), \
                     ('Llama-2-70b-chat', 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3'), \
                     ('CodeLlama-13b-instruct',"meta/codellama-13b-instruct:ca8c51bf3c1aaf181f9df6f10f31768f065c9dddce4407438adc5975a59ce530"), \
                     ('CodeLlama-34b-instruct',"meta/codellama-34b-instruct:b17fdb44c843000741367ae3d73e2bb710d7428a662238ddebbf4302db2b5422")])


action_page = option_menu(None, ["Chat", "Prompt Engineer", "Settings"], 
    icons=['house', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

def prepare_prompt(messagelist: list[dict], system_prompt: str = None):
    prompt = "\n".join([f"[INST] {message['content']} [/INST]" if message['role']=='User' else message['content'] for message in messagelist])
    if system_prompt:
        prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n[\INST] {prompt}"
    return prompt

def run(messagehistory: dict[dict], provider: str = 'Replicate', llm: str = None):
    messages = list(messagehistory.values())
    if provider == 'Replicate':
        prompt = prepare_prompt(messages, system_prompt=setter['DEFAULT_SYSTEM_PROMPT'])
        resp = replicate.run(llm, {"prompt": prompt, "max_new_tokens": setter['max_new_tokens'], "temperature": setter['temperature'], "top_k": setter['top_k'], "top_p": setter['top_p']})
        return resp
    elif provider == 'OpenAI':
        resp = openaiclient.chat.completions.create(model=llm, messages= messagehistory.values(), max_tokens=setter['max_new_tokens'], temperature=setter['temperature'], top_k=setter['top_k'], top_p=setter['top_p'])
        return resp

def list_openai_models():
    models = list(openai.models.list())
    res = [model.id for model in models if 'gpt' in model.id.lower()]
    return res

def clear_all(*conv_hist, **kwargs):
    del conv_hist
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def enter_conversation(history, conv_dict):
    history[len(history)] = conv_dict

def chat():
    conversation_history = OrderedDict()
    provider = st.sidebar.selectbox('Provider', ['Replicate', 'OpenAI', 'Ollama'], on_change=clear_all, args=conversation_history)
    if provider == 'Replicate':
        model = st.sidebar.selectbox('Model', model_choices['replicate'], on_change=clear_all, args=conversation_history)
        llm = replicatemap[model]
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'OpenAI':
        modellist = list_openai_models()
        model = st.sidebar.selectbox('Model', modellist, on_change=clear_all, args=conversation_history)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')

    setter['temperature'] = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    setter['top_k'] = st.sidebar.number_input('top_k', min_value=1, max_value=10000, value=50, step=50)
    setter['top_p'] = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    setter['max_new_tokens'] = st.sidebar.slider('max_new_tokens', min_value=32, max_value=4096, value=2048, step=8)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input(">> "):
        st.session_state.messages.append({'role': 'User', 'content': prompt})
        enter_conversation(conversation_history, {'role': 'User', 'content': prompt})
        with st.chat_message("User"):
            st.markdown(prompt)
        with st.chat_message("Assistant"):
            message_placeholder = st.empty()
            full_response = ""
            print(conversation_history.values())
            for output in run(conversation_history, provider, llm):
                full_response += output
                message_placeholder.markdown(full_response+"▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({'role': 'Assistant', 'content': full_response})
        enter_conversation(conversation_history, {'role': 'Assistant', 'content': full_response})

def prompting():
    pass

def settings():
    prompt= st.text_area('Default System Prompt:', value=setter['DEFAULT_SYSTEM_PROMPT'], height=200)
    max_tokens= st.number_input('Max Tokens: ', value=setter['MAX_TOKENS'])
    setter['DEFAULT_SYSTEM_PROMPT'] = prompt
    setter['MAX_TOKENS'] = max_tokens

page_names_to_funcs = {
    "Chat": chat,
    "Prompt Engineer": prompting,
    "Settings": settings
}
page_names_to_funcs[action_page]()