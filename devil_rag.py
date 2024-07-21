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

from os import environ as env

dotenv.load_dotenv()
openai_api_key = env['openai_api_key']


"""
I think Development of the AI should be regulated. AI is taking away jobs and people are getting fired

I agree


I can't agree with these guys. Rather than taking away, AI creates jobs!
"""


class RagDevil(DevilBase):
    def __init__(self):
        self.__client = ChatOpenAI(api_key=openai_api_key)
        self.history: Sequence[MessageLikeRepresentation] = [
            SystemMessage(content="""You are the devil's advocate.
             You listen to the stories of people (from documents)
             who oppose public opinion but are unable to express their dissent
             due to social awareness, and based on this, you present arguments against the public opinion.
             Respond in 2-3 sentences
         """)
        ]
        self.__counter = 0

    def __get_opposing_opinions(self):
        dms = crud.get_unused_secret_dms(db=next(get_db()))
        opinions = "opinions:\n"
        ids = []

        for dm in dms:
            opinions += dm.content + "\n"
            ids.append(dm.id)

        opinion_docs = [Document(page_content=opinions)]

        return opinion_docs, ids

    def __mark_as_used(self, ids: List[int]):
        crud.mark_dm_as_used(db=next(get_db()), ids=ids)

    async def __get_rag_chain(self):
    
      prompt = ChatPromptTemplate.from_messages(self.history + [SystemMessage(content="{question}\n{context}")])

      docs, ids = self.__get_opposing_opinions()
      self.__mark_as_used(ids=ids)

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
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | self.__client
        | StrOutputParser()
      )
      
      return rag_chain

    @override
    async def __get_stream(self, messages: Sequence[MessageLikeRepresentation]):
        print(messages)
        rag_chain = await self.__get_rag_chain()
        stream = rag_chain.astream("Express opinion against the public opinion")

        return stream

    @override
    async def get_streamed_content(self, streamHandler: Callable[[str, bool], None],
                                   completionHandler: Callable[[str], None]):
        stream = await self.__get_stream(self.history)
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

    @override
    def get_history(self):
        return self.history
    
    @override
    def reset_history(self):
        self.history = []


# devil = DevilManager()
devil = RagDevil()