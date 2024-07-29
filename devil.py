# from devil_raw import 
from devil_rag import RagDevil

shared_devil = RagDevil()

def get_devil():
  return shared_devil