from openai import OpenAI, AsyncOpenAI
from openai.types.beta.threads.message import Message
from openai.pagination import SyncCursorPage
from typing import Optional, List

from .secrets import openai_api_key
import asyncio


class GenDevilManager:
    def __init__(self):
        self.__client = AsyncOpenAI(api_key=openai_api_key)

    async def run_devill(self):
        response = await self.__client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a  assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant",
                    "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )

        if True:
            return None
        else:
            return None
