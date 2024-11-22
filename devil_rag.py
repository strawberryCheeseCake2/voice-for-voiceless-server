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

from sentence_transformers import SentenceTransformer, util

from database import get_db
# from secret import openai_api_key
import dotenv
from devil_base import DevilBase
import crud
from constants import critique_system_message, emp_info, summary_system_message

from os import environ as env
# from models import SecretDm
import schemas

dotenv.load_dotenv()
openai_api_key = env['openai_api_key']


"""
I think Development of the AI should be regulated. AI is taking away jobs and people are getting fired
 bert
I agree


I can't agree with these guys. Rather than taking away, AI creates jobs!
"""


class RagDevil(DevilBase):
    def __init__(self):
        self.__client = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o",
            temperature=1.1,
            max_tokens=256,
            model_kwargs={
                "frequency_penalty": 0.7,
            }
        )
        self.__summary_client = AsyncOpenAI(api_key=openai_api_key)
        self.current_summary = ""
        self.bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")

        self.history: Sequence[MessageLikeRepresentation] = []
        self.__system_prompt = critique_system_message
        self.__counter = 0
        self.enabled = True
        

    def __create_critique_param(self):

        _cur_summary = "[Target]\n" + self.current_summary + "\n"

        chat_history_string = "[대화 내역]\n"

        for msg in self.history:
            chat_history_string += msg.content + "\n"


        _messages = [
            SystemMessage(content=self.__system_prompt),
            HumanMessage(content=_cur_summary), # 여론
            HumanMessage(content=emp_info), # 직원정보
            HumanMessage(content="{context}"),  # Secret DM
            HumanMessage(content=chat_history_string),  # Chat History
            HumanMessage(content="2-3문장으로 짧게 답해")
        ]

        #TODO: Remove
        print("Critique Param Created, Used Info:")
        print("1. sys_prompt, 2.emp_info")
        print(f"3. current chat history summary: {_cur_summary}")
        print(f"4.chat history (last chat of history={self.history[-1]})")



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

    async def __get_critique_chain(self):

        await self.__update_summary()

        _messages = self.__create_critique_param()
        # print(_messages)
        prompt = ChatPromptTemplate.from_messages(_messages)

        docs, ids = self.__get_opposing_opinions()
        # self.__mark_as_used(ids=ids) # For Condition A

        print("---------------------------------")
        print("5.Secret DM used in this request:")
        print(docs)
        print("-----------------------------------")
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

    def calculate_completion_similarity(self, target_sentence, candidate_sentences):
        # Encode the target sentence once
        target_embedding = self.bert_model.encode(target_sentence, convert_to_tensor=True).cuda()
        # Encode all candidate sentences
        candidate_embeddings = self.bert_model.encode(candidate_sentences, convert_to_tensor=True).cuda()
        # Compute cosine similarity between the target and each candidate sentence
        cosine_scores = util.pytorch_cos_sim(target_embedding, candidate_embeddings)
        # Convert tensor to list of similarity scores
        similarity_scores = cosine_scores.squeeze().tolist()
        return similarity_scores


    async def get_critique(self):
        rag_chain = await self.__get_critique_chain()
        completion = await rag_chain.ainvoke("")


        # Check Duplicate
        prev_ai_messeges = list(filter(lambda x: type(x) is AIMessage, self.history))
        prev_ai_str_messages = list(map(lambda x: x.content, prev_ai_messeges))
        print("AI MESSGAE")
        print(prev_ai_messeges)
        print("AI MESSAGE_STRING")
        print(prev_ai_str_messages)
        
        if len(prev_ai_messeges) >= 2:
            print("Content")
            print(prev_ai_messeges[0].content)
            print(prev_ai_messeges[1].content)
            scores = self.calculate_completion_similarity(completion, prev_ai_str_messages)

            for idx, score in enumerate(scores):
                if score > 0.8:
                    print(prev_ai_messeges[idx])
                    return None


        self.__add_history(AIMessage(content=completion))
        return completion


    def __create_summary_param(self):

        chat_history_string = "[대화 내역]\n"

        for msg in self.history:
            chat_history_string += msg.content + "\n"

        br = "\n"
        return [
            {"role": "system", "content": summary_system_message},
            {"role": "user", "content": f"""
            {chat_history_string}
        """},
        ]

    async def __update_summary(self):
        res = await self.__summary_client.chat.completions.create(
            model="gpt-4o",
            messages=self.__create_summary_param(),
            stream=False,
            max_tokens=256,
            temperature=0.3
        )

        summary = res.choices[0].message.content

        print(summary)
        self.current_summary = summary
        
        return summary

    async def remove_used_dm(self, ai_completion: str):
        # Calculate similarity between AI Answer and Secret DM
        # Delete Secret Dm if similar 

        dms = crud.get_unused_secret_dms(db=next(get_db()))
        dms_str_list = list(map(lambda x: x.content, dms))

        if len(dms_str_list) <= 0:
            return
        res = self.calculate_completion_similarity(ai_completion, dms_str_list)
        if len(res) <= 1 or type(res) is float:
            res = [res]

        similar_dm_list = []
        threshold = 0.5
        print("------------------------")
        print("Remove Duplicate DM START")
        for idx, sim in enumerate(res):
            if sim > threshold:
                similar_dm_list.append(dms[idx].id)
        print("Similar DM LIST START, they will be removed")
        print(similar_dm_list)
        print("--------------------")
        self.__mark_as_used(similar_dm_list)


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
