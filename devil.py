from openai import OpenAI, AsyncOpenAI
from openai.types.beta.threads.message import Message
from openai.pagination import SyncCursorPage
from typing import Optional, List

from .secrets import openai_api_key
import asyncio


class DevilManager:
  def __init__(self):
    self.__client = AsyncOpenAI(api_key=openai_api_key)
    
  
  async def setup_devil(self):
    print("start devil setup")
    self.__assistant = await self.__client.beta.assistants.create(
      name="Devil",
      instructions="You are a devil's advocate. You have to express a contentious opinion with reasonable reason",
      tools=[],
      model="gpt-4o",
    )

    self.__thread = await self.__client.beta.threads.create()
    print("Done Setup")

  
  async def run_devill(self):
    print("before createpoll")
    run = await self.__client.beta.threads.runs.create_and_poll(
      thread_id=self.__thread.id,
      assistant_id=self.__assistant.id,
      instructions="You are a devil's advocate. You have to express a contentious opinion with reasonable reason"
    )

    if run.status == 'completed': 
      messages: SyncCursorPage[Message] = await self.__client.beta.threads.messages.list(
        thread_id=self.__thread.id
      )
      print(messages)
      message = messages.data[0]
      response = message.content[0].text.value

      return response
    else:
      return None


  async def add_user_message(self, message: str):
    print("start add message")
    await self.__client.beta.threads.messages.create(
      thread_id=self.__thread.id,
      role="user",
      # content="I think abortion should be legal. Making it illegal can infringe on the rights of pregnant women"
      content=message
    )
    print("finish")
    


# manager = DevilManager()

# async def test():
#   manager.setup_devil()
#   print("m1 before")
#   await manager.add_user_message("I think abortion should be legal. Making it illegal can infringe on the rights of pregnant women")
#   print("m1 after")
#   print("m2 before")
#   await manager.add_user_message("I agree with you. no one can hinder one's decision")
#   print("m2 after")
#   msgs = await manager.run_devill()
#   print(msgs)

# asyncio.run(test())




