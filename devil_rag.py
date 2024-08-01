from openai import AsyncOpenAI
from typing import Optional, List, Callable, override, Sequence
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from sqlalchemy.orm import Session
from fastapi import Depends

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

from database import get_db
# from secret import openai_api_key
import dotenv
from devil_base import DevilBase
import crud
from constants import critique_system_message

from os import environ as env
# from models import SecretDm
import schemas

dotenv.load_dotenv()
openai_api_key = env['openai_api_key']


"""
I think Development of the AI should be regulated. AI is taking away jobs and people are getting fired

I agree


I can't agree with these guys. Rather than taking away, AI creates jobs!
"""


class RagDevil(DevilBase):
    def __init__(self):
        self.__client = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=256
            )
        self.history: Sequence[MessageLikeRepresentation] = []
        self.__system_prompt = critique_system_message
        self.__counter = 0
        self.enabled = True
        

    def create_message_param(self):
        chat_history_string = "[대화 내역]\n"

        for msg in self.history:
            chat_history_string += msg.content + "\n"

        _messages = [
            SystemMessage(content=self.__system_prompt),
            HumanMessage(content="{context}"),
            HumanMessage(content=chat_history_string)
        ]

        return _messages

    def __get_opposing_opinions(self):
        """
        For Condition B, C, used Field aims to remove dms used in first round experiment
        For Condition A, it aims to avoid same secret dm appear again
        """
        dms = crud.get_unused_secret_dms(db=next(get_db()))
        # dms = crud.get_all_secret_dms(db=next(get_db()))

        opinions = "[Comment Box]\n"
        ids = []

        for dm in dms:
            opinions += dm.content + "\n"
            ids.append(dm.id)

        if len(dms) <= 0:
            opinions += "Empty"

        opinion_docs = [Document(page_content=opinions)]

        return opinion_docs, ids

    def __mark_as_used(self, ids: List[int]):
        crud.mark_dm_as_used(db=next(get_db()), ids=ids)

    async def __get_rag_chain(self):

        _messages = self.create_message_param()
        print(_messages)
        prompt = ChatPromptTemplate.from_messages(_messages)

        docs, ids = self.__get_opposing_opinions()
        # self.__mark_as_used(ids=ids) # For Condition A

        print(docs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = await Chroma.afrom_documents(
            documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain: RunnableSerializable = (
            {"context": retriever | format_docs}
            | prompt
            | self.__client
            | StrOutputParser()
        )

        return rag_chain


    ## Stream
    @override
    async def __get_stream(self):
        rag_chain = await self.__get_rag_chain()
        stream = rag_chain.astream("")

        return stream

    @override
    async def get_streamed_content(self, streamHandler: Callable[[str, bool], None],
                                   completionHandler: Callable[[str], None]):
        stream = await self.__get_stream()
        completion_buffer = ""

        isFirstChunk = True

        async for chunk in stream:
            chunk_content = chunk
            if chunk_content is not None:

                completion_buffer += chunk_content
                await streamHandler(completion_buffer, isFirstChunk)

                if isFirstChunk:
                    isFirstChunk = False

        self.__add_history(AIMessage(content=completion_buffer))
        completionHandler(completion_buffer)

    @override
    def __add_history(self, chat: MessageLikeRepresentation):
        self.history.append(chat)

    @override
    def add_user_message(self, sender: str, message: str):
        self.__add_history(
            HumanMessage(content=f"{sender}: {message}")
        )
        self.increase_counter()

    @override
    def get_counter(self):
        return self.__counter

    @override
    def reset_counter(self):
        self.__counter = 0

    @override
    def increase_counter(self):
        self.__counter += 1

    # For admin
    @override
    def get_history(self):
        return self.history

    @override
    def reset_history(self):
        self.history = []
        self.reset_counter()

    def set_system_prompt(self, content: str):
        self.__system_prompt = content

    def get_system_prompt(self):
        return self.__system_prompt

    def enable(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            self.reset_counter()

    def is_enabled(self):
        return self.enabled