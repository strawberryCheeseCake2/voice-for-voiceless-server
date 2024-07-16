# from openai import AsyncOpenAI
# from typing import Optional, List, Callable, override
# from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# from .secrets import openai_api_key
# from .devil_base import DevilBase
# import asyncio


# class DevilManager(DevilBase):
#     def __init__(self):
#         self.__client = AsyncOpenAI(api_key=openai_api_key)
#         self.history: List[ChatCompletionMessageParam] = [
#             {"role": "system", "content": "너는 악마의 대변인이야. 근거를 들어서 여론에 반박해. 세 문장 이내로 말해."}
#         ]
#         self.__counter = 0

#     @override
#     async def __get_stream(self, messages: List[ChatCompletionMessageParam]):
#         print(messages)
#         stream = await self.__client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             stream=True,
#         )

#         return stream
    
#     @override
#     async def get_streamed_content(self, streamHandler: Callable[[str, bool], None],
#                                     completionHandler: Callable[[str], None]):
#         stream = await self.__get_stream(self.history)
#         completion_buffer = ""

#         isFirstChunk = True

#         async for chunk in stream:
#             chunk_content = chunk.choices[0].delta.content
#             if chunk_content is not None:
               
#                 completion_buffer += chunk_content
#                 await streamHandler(completion_buffer, isFirstChunk)

#                 if isFirstChunk:
#                     isFirstChunk = False

#         self.__add_history({"role": "assistant", "content": completion_buffer})
#         completionHandler(completion_buffer)

#     @override
#     def __add_history(self, chat: ChatCompletionMessageParam):
#         self.history.append(chat)
    
#     @override
#     def add_user_message(self, sender: str ,message: str):
#         self.__add_history({"role": "user", "content": f"{sender}: {message}"})
#         self.increase_counter()

#     @override  
#     def get_counter(self):
#         return self.__counter
    
#     @override
#     def reset_counter(self):
#         self.__counter = 0

#     @override
#     def increase_counter(self):
#         self.__counter += 1


# # async def test2():
# #     devil = DevilManager()
# #     devil.add_history({"role": "system", "content": "너는 악마의 대변인이야. 근거를 들어서 여론에 반박해. 세 문장 이내로 말해."})
# #     devil.add_history({"role": "user", "content": "윤이: 우리 회사는 비대면 근무를 도입해야 합니다. 비대면 근무를 하면 사원들이 출퇴근 시간을 절약할 수 있습니다."})
# #     devil.add_history({"role": "user", "content": "철수: 동의합니다."})
    
# #     def broadcast_chunks(content: str):
# #       streamed_message = {"sentTime": "",
# #                        "username": "Devil", "message": content}
# #       print(content)
# #     await devil.get_streamed_content(streamHandler=broadcast_chunks)




# # asyncio.run(test2())
