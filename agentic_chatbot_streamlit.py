from agentic_chatbot_backend import workflow
from langchain_core.messages import HumanMessage
import streamlit as st

thread_id = "1"

st.title("Agentic Chatbot")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id

CONFIG = {"configurable":{"thread_id":st.session_state.thread_id}}

if "messages" not in st.session_state:
    st.session_state.messages = []                 # used for persistence, stored messages are only meant to be just displayed in the frontend, without it previous messages get replaced by new one.   session_state doesn't store data permanently, only survives reruns.
    state = workflow.get_state(CONFIG)
    if state.values.get("messages"):
        for msg in state.values["messages"]:
            role = "user" if msg.type=="human" else "assistant"
            st.session_state.messages.append({"role":role,"content":msg.content})            # loadind data from checkpointer into session_state

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    def stream_response(user_input):
        for chunk,metadata in workflow.stream(
            {"messages":[HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        ):
            content = chunk.content
            if isinstance(content,str):
                yield content
            elif isinstance(content,list):
                for block in content:
                    if isinstance(block,dict):
                        text = block.get("text","")
                        if text:
                            yield text
                    elif isinstance(block,str):
                        yield block

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.write_stream(stream_response(user_input))

    st.session_state.messages.append({"role":"assistant","content":response})