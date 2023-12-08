import streamlit as st
from streamlit_option_menu import option_menu
import replicate
from openai import OpenAI
from langchain.llms import ollama
from collections import defaultdict
import os
from uuid import uuid4 as v4

setter = defaultdict(None)
setter['DEFAULT_SYSTEM_PROMPT'] = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
setter['MAX_TOKENS'] = 4096


os.environ['REPLICATE_API_TOKEN'] = st.secrets['replicate']['API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets.openai.API_KEY


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
    prompt = "\n".join([f"[INST] {message['content']} [/INST]" if message['role']=='user' else message['content'] for message in messagelist])
    if system_prompt:
        prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n[\INST]ß {prompt}"
    return prompt

def run(provider: str, llm: str, conversation_id):
    messages = list(st.session_state.conversations[conversation_id])
    if provider == 'Replicate':
        prompt = prepare_prompt(messages, system_prompt=setter['DEFAULT_SYSTEM_PROMPT'])
        resp = replicate.run(llm, {"prompt": prompt, "max_new_tokens": st.session_state.max_new_tokens, "temperature": st.session_state.temperature, "top_k": st.session_state.top_k, "top_p": st.session_state.top_p})
        return resp
    elif provider == 'OpenAI':
        client = OpenAI()
        resp = client.chat.completions.create(model=llm, messages= messages, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
        return resp.choices[0].message.content

def list_openai_models():
    client = OpenAI()
    models = list(client.models.list())
    res = [model.id for model in models if 'gpt' in model.id.lower()]
    return res

def clear_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def create_new_conversation():
    conversation_id = v4()
    if "conversations" not in st.session_state:
        st.session_state['conversations'] = {}
    st.session_state.conversations[conversation_id] = []
    st.session_state['current_conversation'] = conversation_id
    return conversation_id

def select_convo(key):
    st.session_state['current_conversation'] = key

def delconv(key):
    del st.session_state.conversations[key]
    if len(st.session_state.conversations) > 0:
        st.session_state['current_conversation'] = list(st.session_state.conversations.keys())[0]
    else:
        del st.session_state['current_conversation']
    
    
def generate_buttons():
    for key,convo in st.session_state.conversations.items():
        try:
            st.sidebar.button(f"{convo[0]['content'][:50]}...", key=key, on_click=select_convo, args=(key,), use_container_width=True)
        except IndexError:
            st.sidebar.button(f"New Conversation...", key=key, on_click=select_convo, args=(key,), use_container_width=True)
def draw_sidebar():
    provider = st.sidebar.selectbox('Provider', ['Replicate', 'OpenAI', 'Ollama'])
    if provider == 'Replicate':
        model = st.sidebar.selectbox('Model', model_choices['replicate'])
        llm = replicatemap[model]
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'OpenAI':
        modellist = list_openai_models()
        model = st.sidebar.selectbox('Model', modellist)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')

    st.session_state.temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    st.session_state.top_k = st.sidebar.number_input('top_k', min_value=1, max_value=10000, value=50, step=50)
    st.session_state.top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state.max_new_tokens = st.sidebar.slider('max_new_tokens', min_value=32, max_value=4096, value=2048, step=8)
    st.session_state.provider = provider   
    st.session_state.llm = llm

def chat():
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    with placeholder1.container():
        draw_sidebar()
        st.sidebar.button("Begin New Conversation", on_click=create_new_conversation)

    

    if "conversations" not in st.session_state or len(st.session_state.conversations) == 0:
        current_convo = create_new_conversation()
    elif "current_conversation" in st.session_state and st.session_state.current_conversation:
        current_convo = st.session_state['current_conversation']
    else:
        current_convo = list(st.session_state.conversations.keys())[0]
        st.session_state['current_conversation'] = current_convo

    for message in st.session_state.conversations[current_convo]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input(">> "):
        st.session_state.conversations[current_convo].append({'role': 'user', 'content': prompt})
        with st.chat_message("User"):
            st.markdown(prompt)
        with st.chat_message("Assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for output in run(st.session_state.provider, st.session_state.llm, current_convo):
                full_response += output
                message_placeholder.markdown(full_response+"▌")
            message_placeholder.markdown(full_response)

        st.session_state.conversations[current_convo].append({'role': 'assistant', 'content': full_response})

    with placeholder2.container():
        st.sidebar.markdown("#### Conversations")
        generate_buttons()
        st.sidebar.button("Delete Conversation", on_click=delconv, args=(current_convo,), use_container_width=True)
    if st.session_state['current_conversation'] != current_convo:
        current_convo = st.session_state['current_conversation']


def createnewkey():
    id = v4()
    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = defaultdict(dict)
    st.session_state.placeholders[id] = {'name': '', 'value': ''}
    st.session_state.current_key = id
    st.session_state.keynum += 1

def delkey(key):
    del st.session_state.placeholders[key]
    if len(st.session_state.placeholders) > 0:
        st.session_state.current_key = list(st.session_state.placeholders.keys())[0]
    else:
        del st.session_state['current_key']

def drawkeys():
    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = defaultdict(dict)
        createnewkey()
    for i, (id, val) in enumerate(st.session_state.placeholders.items()):
        # if id not in st.session_state.entered_keys:
        st.session_state.placeholders[id]['name'] = st.sidebar.text_input(f"Key {i}", value=f"Key {i}", key=id)
        st.session_state.entered_keys.append(id)
    st.sidebar.button("Add Key", on_click=createnewkey, use_container_width=True)

def gen_prompt():
    prompt = st.session_state.sys_prompt
    for id, val in st.session_state.placeholders.items():
        prompt = prompt.replace(f"{{{val['name']}}}", val['value'])
    return prompt

def generate(t):
    prompt = gen_prompt()
    full_response = ""
    if st.session_state.provider == 'Replicate':
        for resp in replicate.run(st.session_state.llm, {"prompt": prompt, "max_new_tokens": st.session_state.max_new_tokens, "temperature": st.session_state.temperature, "top_k": st.session_state.top_k, "top_p": st.session_state.top_p}):
            full_response += resp
            st.session_state.generation = full_response+"▌"
        st.session_state.generation = full_response
    elif st.session_state.provider == 'OpenAI':
        client = OpenAI()
        prompt = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=st.session_state.llm, messages=prompt, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
        st.session_state.generation = resp.choices[0].message.content
    print(st.session_state.generation)

def dynamic(t):
    t.empty()
    return generate(t)

def prompting():
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    st.session_state.entered_keys = []
    st.session_state.keynum = 0
    st.session_state.generation = ""

    with placeholder1.container():
        draw_sidebar()

    with placeholder2.container():
        st.sidebar.markdown("#### Add Keys")
        drawkeys()
        st.sidebar.button("Delete Keys", on_click=delkey, args=(st.session_state.current_key if 'current_key' in st.session_state else None,), use_container_width=True)

    
    with st.container():
        st.markdown('#### Prompt Engineer')

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sys_prompt = "Enter a prompt here, using {Key 0} as placeholders"
            st.session_state.sys_prompt = st.text_area('System Prompt', value=st.session_state.sys_prompt, height=200)         
            for i, (id, val) in enumerate(st.session_state.placeholders.items()):
                st.session_state.placeholders[id]['value'] = st.text_area(f"{val['name'] if len(val['name'])>0 else f'Key {i}'}", value=val['value'], key=str(id)+'value')

        with col2:
            st.markdown('##### Generation')
            textarea = st.empty()
            textarea.markdown(st.session_state.generation)
            

        st.button("Generate", on_click=generate, args=(textarea,))

    
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