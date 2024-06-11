import asyncio
from opcua import Client
import logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

class OPC:
	def __init__(self,client) -> None:
		self.client = client

	def connect_client(self):
		pass

	def activate_tag(self):
		pass

	def deactivate_tag(self):
		pass