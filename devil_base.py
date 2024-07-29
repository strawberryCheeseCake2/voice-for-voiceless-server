from abc import ABCMeta, abstractmethod
from typing import Union, List, Sequence, Callable

from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from langchain_core.prompts.chat import MessageLikeRepresentation


class DevilBase(metaclass=ABCMeta):

  @abstractmethod
  async def get_streamed_content(self, streamHandler: Callable, completionHandler: Callable):
        pass

  @abstractmethod
  def add_user_message(self, sender: str, message: str):
    pass

  @abstractmethod
  def get_counter(self):
    pass

  @abstractmethod
  def reset_counter(self):
    pass

  @abstractmethod
  def increase_counter(self):
    pass
  
  @abstractmethod
  def get_history(self):
        pass
  
  @abstractmethod
  def reset_history(self):
    pass

