from openai import OpenAI
from openai.types.beta.threads.message import Message
from openai.pagination import SyncCursorPage
from typing import Optional, List


class DevilManager:
  def __init__(self):
    self.client = OpenAI(api_key='')
    self.assistant = self.client.beta.assistants.create(
      name="Devil",
      instructions="You are a devil's advocate. You have to express a contentious opinion with reasonable reason",
      tools=[],
      model="gpt-4o",
    )
    self.thread = self.client.beta.threads.create()
  
  def run_devill(self):
    print("before createpoll")
    run = self.client.beta.threads.runs.create_and_poll(
      thread_id=self.thread.id,
      assistant_id=self.assistant.id,
      instructions="You are a devil's advocate. You have to express a contentious opinion with reasonable reason"
    )
    print("after")
    if run.status == 'completed': 
      messages: SyncCursorPage[Message] = self.client.beta.threads.messages.list(
        thread_id=self.thread.id
      )

      message = messages.data[0]
      response = message.content[0].text.value

      return response
    else:
      return None


  def add_user_message(self, message: str):
    self.client.beta.threads.messages.create(
      thread_id=self.thread.id,
      role="user",
      # content="I think abortion should be legal. Making it illegal can infringe on the rights of pregnant women"
      content=message
    )
    

# manager = DevilManager()

# print("m2after")
# manager.add_user_message("I think abortion should be legal. Making it illegal can infringe on the rights of pregnant women")
# print("m1 after")
# manager.add_user_message("I agree with you. no one can hinder one's decision")
# print("m2 after")

# msgs = manager.run_devill()
