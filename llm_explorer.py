import streamlit as st
import litellm
from streamlit_option_menu import option_menu
import replicate
from openai import OpenAI
from langchain.llms import ollama
from collections import defaultdict
import os
from uuid import uuid4 as v4
import yaml
from io import StringIO
import requests
from litellm import completion
import json

st.session_state.sys_prompt = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being concise. Please ensure that your responses are socially unbiased and positive in nature. Please also make the response as concise as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
st.session_state.max_tokens = 4096


os.environ['REPLICATE_API_TOKEN'] = st.secrets['replicate']['API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets.openai.API_KEY
os.environ['HUGGINGFACE_API_KEY'] = st.secrets.huggingface.API_KEY


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
        prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n[\INST] {prompt}"
    return prompt

def apply_prompt_template(messagelist: list[dict], system_prompt: str = None, bos_token: str = "", eos_token: str = ""):
    print("Preparing custom prompt from messages!")
    prompt = bos_token + st.session_state.prompt_template['initial_prompt_value'] + "\n"
    bos_open = True

    for message in messagelist:
        role = message['role']

        if role in ['system','user'] and not bos_open:
            prompt += bos_token
            bos_open = True

        prompt += st.session_state.prompt_template['roles'][role]['pre_message'] + message['content'] + st.session_state.prompt_template['roles'][role]['post_message']

        if role == 'assistant':
            prompt += eos_token
            bos_token = False
        
    prompt += st.session_state.prompt_template['final_prompt_value']
    return prompt

def run(provider: str, llm: str, conversation_id):
    messages = list(st.session_state.conversations[conversation_id])
    if provider == 'Replicate':
        if 'custom_prompt' in st.session_state and st.session_state.custom_prompt:
            prompt = apply_prompt_template(messages, system_prompt=st.session_state.sys_prompt, bos_token="<s>", eos_token="</s>")
            print(prompt)
        else:
            prompt = prepare_prompt(messages, system_prompt=st.session_state.sys_prompt)
        resp = replicate.run(llm, {"prompt": prompt, "max_new_tokens": st.session_state.max_new_tokens, "temperature": st.session_state.temperature, "top_k": st.session_state.top_k, "top_p": st.session_state.top_p})
        return resp
    elif provider == 'OpenAI':
        client = OpenAI()
        resp = client.chat.completions.create(model=llm, messages= messages, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p, presence_penalty=st.session_state.presence_penalty, frequency_penalty=st.session_state.frequency_penalty)
        return resp.choices[0].message.content
    elif provider == 'Ollama':
        litellm.drop_params = True
        resp = completion(model='ollama/'+llm, messages=messages, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p, presence_penalty=st.session_state.presence_penalty, frequency_penalty=st.session_state.frequency_penalty)
        return resp.choices[0].message.content
    elif provider == 'Huggingface':
        if 'llama' in llm.lower():
            litellm.register_prompt_template(llm, st.session_state.prompt_template['roles'])
        litellm.drop_params = True
        resp = completion(model='huggingface/'+llm, messages=messages, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p, presence_penalty=st.session_state.presence_penalty, frequency_penalty=st.session_state.frequency_penalty)
        print(resp)
        return resp

def list_openai_models():
    client = OpenAI()
    models = list(client.models.list())
    res = [model.id for model in models if 'gpt' in model.id.lower()]
    return res

def list_ollama_models():
    resp = requests.get('http://localhost:11434/api/tags')
    models = [x['name'] for x in resp.json()['models']]
    return models

def list_hfi_models():
    from huggingface_hub import HfApi, ModelFilter
    api = HfApi()
    models = api.list_models(filter=ModelFilter(task='text-generation', ))
    models = [x.id for x in models]
    return models

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
    provider = st.sidebar.selectbox('Provider', ['Replicate', 'OpenAI', 'Ollama', 'Custom'])
    if provider == 'Replicate':
        model = st.sidebar.selectbox('Model', model_choices['replicate'])
        llm = replicatemap[model]
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'OpenAI':
        modellist = list_openai_models()
        model = st.sidebar.selectbox('Model', modellist)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Ollama':
        modellist = list_ollama_models()
        model = st.sidebar.selectbox('Model', modellist)
        llm = model
        st.markdown(f'##### Chosen Model: 🦙💬 {model}')
    elif provider == 'Custom':
        st.sidebar.markdown(f'###### *Customize endpoing settings in settings menu*')
        llm = 'Custom'
        st.markdown(f'##### Chosen Model: 🤗 ')

    st.session_state.temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.75, step=0.01)
    st.session_state.top_k = st.sidebar.number_input('top_k', min_value=1, max_value=10000, value=50, step=50)
    st.session_state.top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    st.session_state.frequency_penalty = st.sidebar.slider('frequency_penalty', min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    st.session_state.presence_penalty = st.sidebar.slider('presence_penalty', min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
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

def generate():
    prompt = gen_prompt()
    full_response = ""
    if st.session_state.provider == 'Replicate':
        for resp in replicate.run(st.session_state.llm, {"prompt": prompt, "max_new_tokens": st.session_state.max_new_tokens, "temperature": st.session_state.temperature, "top_k": st.session_state.top_k, "top_p": st.session_state.top_p}):
            full_response += resp
            st.session_state.generation = full_response+"▌"
        return full_response
    elif st.session_state.provider == 'OpenAI':
        client = OpenAI()
        prompt = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=st.session_state.llm, messages=prompt, max_tokens=st.session_state.max_new_tokens, temperature=st.session_state.temperature, top_p=st.session_state.top_p)
        return resp.choices[0].message.content
    
def prompting():
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    st.session_state.entered_keys = []
    st.session_state.keynum = 0

    with placeholder1.container():
        draw_sidebar()

    with placeholder2.container():
        st.sidebar.markdown("#### Add Keys")
        drawkeys()
        st.sidebar.button("Delete Keys", on_click=delkey, args=(st.session_state.current_key if 'current_key' in st.session_state else None,), use_container_width=True)

    
    
    st.markdown('#### Prompt Engineer')

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sys_prompt = "Enter a prompt here, using {Key 0} as placeholders"
        st.session_state.sys_prompt = st.text_area('System Prompt', value=st.session_state.sys_prompt, height=200)         
        for i, (id, val) in enumerate(st.session_state.placeholders.items()):
            st.session_state.placeholders[id]['value'] = st.text_area(f"{val['name'] if len(val['name'])>0 else f'Key {i}'}", value=val['value'], key=str(id)+'value')

    with col2:
        st.markdown("###### Model Output")
        textarea = st.empty()
            

    if st.button("Generate", use_container_width=True):
        textarea.markdown(generate())

def build_prompt_template():
    st.session_state.prompt_template = {
        "initial_prompt_value": st.session_state.initial_prompt,
        "roles": {
            "system": {
                "pre_message": st.session_state.sys_prefix,
                "post_message": st.session_state.sys_suffix
            },
            "user": {
                "pre_message": st.session_state.user_prefix,
                "post_message": st.session_state.user_suffix
            },
            "assistant": {
                "pre_message": st.session_state.assistant_prefix,
                "post_message": st.session_state.assistant_suffix
            }
        },
        "final_prompt_value": st.session_state.final_prompt
    }
    return st.session_state.prompt_template

def promptformat():
    if 'sys_prefix' not in st.session_state:
        st.session_state.sys_prefix = ""
    if 'sys_suffix' not in st.session_state:
        st.session_state.sys_suffix = ""
    if 'user_prefix' not in st.session_state:
        st.session_state.user_prefix = ""
    if 'user_suffix' not in st.session_state:
        st.session_state.user_suffix = ""
    if 'assistant_prefix' not in st.session_state:
        st.session_state.assistant_prefix = ""
    if 'assistant_suffix' not in st.session_state:
        st.session_state.assistant_suffix = ""
    if 'initial_prompt' not in st.session_state:
        st.session_state.initial_prompt = ""
    if 'final_prompt' not in st.session_state:
        st.session_state.final_prompt = ""

    st.markdown('#### Prompt Format')
    msgformat = f"{st.session_state.initial_prompt}\n{st.session_state.sys_prefix} [System Message] {st.session_state.sys_suffix}"\
    f" {st.session_state.user_prefix} [User Message] {st.session_state.user_suffix} {st.session_state.assistant_prefix} [Assistant Message] {st.session_state.assistant_suffix}\n{st.session_state.final_prompt}"
    preview = st.empty()
    preview.text_area("Message Format Preview", value= f"{msgformat}", height=200)
    
    st.session_state.initial_prompt = st.text_area("Initial Prompt", value=st.session_state.initial_prompt, height=100)
    st.session_state.sys_prefix = st.text_input("System Message Prefix", value=st.session_state.sys_prefix)
    st.session_state.sys_suffix = st.text_input("System Message Suffix", value=st.session_state.sys_suffix)
    st.session_state.user_prefix = st.text_input("User Message Prefix", value=st.session_state.user_prefix)
    st.session_state.user_suffix = st.text_input("User Message Suffix", value=st.session_state.user_suffix)
    st.session_state.assistant_prefix = st.text_input("Assistant Message Prefix", value=st.session_state.assistant_prefix)
    st.session_state.assistant_suffix = st.text_input("Assistant Message Suffix", value=st.session_state.assistant_suffix)
    st.session_state.final_prompt = st.text_area("Final Prompt", value=st.session_state.final_prompt, height=100)


    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply", use_container_width=True):
            build_prompt_template()
            st.session_state.custom_prompt = True
            st.rerun()
    with col2:
        datadict = dict([('initial_prompt', st.session_state.initial_prompt), ('sys_prefix', st.session_state.sys_prefix), ('sys_suffix', st.session_state.sys_suffix), ('user_prefix', st.session_state.user_prefix), ('user_suffix', st.session_state.user_suffix), ('assistant_prefix', st.session_state.assistant_prefix), ('assistant_suffix', st.session_state.assistant_suffix), ('final_prompt', st.session_state.final_prompt)])
        st.download_button("Save", use_container_width=True, data=yaml.dump(datadict), mime="text/yaml")
        
        
        x = st.file_uploader("Load from file", type=['yaml'])
        if x is not None:
            data = yaml.safe_load(x.read())
            st.session_state.sys_prefix = data.get('sys_prefix', st.session_state.sys_prefix)
            st.session_state.sys_suffix = data.get('sys_suffix', st.session_state.sys_suffix)
            st.session_state.user_prefix = data.get('user_prefix', st.session_state.user_prefix)
            st.session_state.user_suffix = data.get('user_suffix', st.session_state.user_suffix)
            st.session_state.assistant_prefix = data.get('assistant_prefix', st.session_state.assistant_prefix)
            st.session_state.assistant_suffix = data.get('assistant_suffix', st.session_state.assistant_suffix)
            st.session_state.initial_prompt = data.get('initial_prompt', st.session_state.initial_prompt)
            st.session_state.final_prompt = data.get('final_prompt', st.session_state.final_prompt)
            build_prompt_template()
            st.session_state.custom_prompt = True
            st.rerun()


def endpoint():
    st.markdown('#### Custom Endpoint')
    st.session_state.endpoint_url = st.text_input("Endpoint URL", value="https://api.openai.com/v1/engines/davinci/completions")
    st.session_state.endpoint_type = st.selectbox("Endpoint Type", ["Huggingface", "vLLM", "Other"])
    if st.session_state.endpoint_type == "Huggingface":
        st.session_state.endpoint_model = st.text_input("Model ID", value="mistralai/Mistral-7B-Instruct-v0.1")
        st.session_state.endpoint_token = st.text_input("API Token", value="", type="password")
    elif st.session_state.endpoint_type == "vLLM":
        st.session_state.endpoint_model = st.text_input("Model ID", value="llama-2")
        st.session_state.endpoint_token = st.text_input("API Token", value="", type="password")
    elif st.session_state.endpoint_type == "Other":
        st.markdown("Ensure that the custom endpoint accepts 'prompt' and 'max_tokens' as parameters, and returns a JSON object with a list of objects with a 'text' field.\nFor other fields (temperature, top_p, etc), please specify them in the 'Custom Parameters' field below.")
        fields = {'prompt':'str', 'max_tokens': 'int', 'temperature': 'float', 'top_p': 'float', 'top_k': 'int', 'presence_penalty': 'float', 'frequency_penalty': 'float'}
        fieldstr = json.dumps(fields, indent=4)
        st.session_state.endpoint_schema = st.text_area("Endpoint Schema", value=fieldstr, height=200)
        st.button("Apply", use_container_width=True, on_click=read_schema)

def read_schema():
    st.session_state.endpoint_request_payload = json.loads(st.session_state.endpoint_schema)
    for key in st.session_state.endpoint_request_payload.keys():
        st.session_state.endpoint_request_payload[key] = st.text_input(key, value=st.session_state.endpoint_request_payload[key])
    print(st.session_state.endpoint_request_payload)

def settings_master():
    with st.sidebar:
        settingpage = option_menu("Settings", ["General", "Prompt Format", "Custom Endpoint"],
                                icons=['gear', 'list-task', 'code'], 
            menu_icon="cast", default_index=0, orientation="vertical")
    setting_page_to_funcs = {
    "General": settings,
    "Prompt Format": promptformat,
    "Custom Endpoint": endpoint
    }
    setting_page_to_funcs[settingpage]()

def settings():
    st.session_state.sys_prompt = st.text_area('Default System Prompt:', value=st.session_state.sys_prompt, height=200)
    st.session_state.max_tokens = st.number_input('Max Tokens: ', value=st.session_state.max_tokens)



page_names_to_funcs = {
    "Chat": chat,
    "Prompt Engineer": prompting,
    "Settings": settings_master
}
page_names_to_funcs[action_page]()