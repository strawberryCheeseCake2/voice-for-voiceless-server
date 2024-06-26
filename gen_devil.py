from openai import OpenAI, AsyncOpenAI
from openai.types.beta.threads.message import Message
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional, List
from time import time

from secrets import openai_api_key
import asyncio


class GenDevilManager:
    def __init__(self):
        self.__client = AsyncOpenAI(api_key=openai_api_key)

    async def run_devill(self):
        response: ChatCompletion = await self.__client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a devil's advocate. You have to express a contentious opinion with reasonable reason"},
                {"role": "user", "content": "User 1: I think abortion should be legal. Making it illegal can infringe on the rights of pregnant women. User2: I agree with you. no one can hinder one's decision"},
                # {"role": "user", "content": "I agree with you. no one can hinder one's decision"},
            ],
            max_tokens=200
        )
        
        if response.choices[0]:
            return response.choices[0].message.content
        else:
            return None


devil = GenDevilManager()

async def test():
    prev = time()
    resp = await devil.run_devill()
    print(resp)
    print(time() - prev)

asyncio.run(test())

