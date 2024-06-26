from openai import AsyncOpenAI
from typing import Optional, List, Callable
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from secrets import openai_api_key
import asyncio


class DevilManager:
    def __init__(self):
        self.__client = AsyncOpenAI(api_key=openai_api_key)
        self.history: List[ChatCompletionMessageParam] = []
        self.session_chats: List[ChatCompletionMessageParam] = []
        self.completion_buffer = ""

    async def __get_stream(self, messages: List[ChatCompletionMessageParam]):
        return await self.__client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )

    async def get_streamed_content(self, streamHandler: Callable):
        stream = self.__get_stream(self.session_chats)
        async for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                self.completion_buffer += chunk_content
                streamHandler(self.completion_buffer)

    def clear_buffer(self):
        self.completion_buffer = ""

    def add_session_chat(self, chat: ChatCompletionMessageParam):
        self.session_chats += chat
        return

    def add_history(self, chat: ChatCompletionMessageParam):
        self.history += chat
        return


# client = AsyncOpenAI(api_key=openai_api_key)


# async def test():
#     stream = await client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "너는 악마의 대변인이야. 근거를 들어서 여론에 반박해."},
#             {"role": "user", "content": "User 1: 우리 회사는 비대면 근무를 도입해야 합니다. 비대면 근무를 하면 사원들이 출퇴근 시간을 절약할 수 있습니다."},
#             {"role": "user", "content": "User 2: 동의합니다"}
#         ],
#         stream=True,
#     )

#     async for chunk in stream:
#         chunk_content = chunk.choices[0].delta.content
#         if chunk_content is not None:
#             # 합쳐서 broadcast
#             print(chunk_content, end="")

async def test2():
    devil = DevilManager()
    devil.add_session_chat({"role": "system", "content": "너는 악마의 대변인이야. 근거를 들어서 여론에 반박해."})
    devil.add_session_chat({"role": "user", "content": "윤이: 우리 회사는 비대면 근무를 도입해야 합니다. 비대면 근무를 하면 사원들이 출퇴근 시간을 절약할 수 있습니다."})
    devil.add_session_chat({"role": "user", "content": "철수: 동의합니다."})
    
    devil.get_streamed_content()




# asyncio.run(test())
