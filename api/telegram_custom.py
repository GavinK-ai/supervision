# import the necessary packages
from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
class Telegram:
	def __init__(self):
		custom_token = "6171542378:AAE9ZNf3JY1aAAd3I3OBUa5v7kv3eqnfn8k"
		self.application = Application.builder().token(custom_token).connect_timeout(60).read_timeout(60).write_timeout(60).build()
		self.chat_SSIP_alert = -832936404
		self.chat_SSIP_daily_log = -921382569
		self.chat_SSIP_workshop_alert = -961695618
		self.chat_SSIP_alert_security = -4048682772
		self.thread = None
		print(f"Telegram API Initialized")

		# self.encoded_image = None
		# self.caption = None
		# self.message = None

	async def send_image(self,encoded_image,caption):
		await self.application.bot.send_photo(self.chat_SSIP_alert,encoded_image,caption)

	async def send_message(self,message):
		await self.application.bot.send_message(self.chat_SSIP_alert,message)

	async def send_video(self,video_path,caption):
		await self.application.bot.send_video(self.chat_SSIP_alert,video_path,caption=caption,write_timeout=None,supports_streaming=False)		

	async def send_log(self,message):
		await self.application.bot.send_message(self.chat_SSIP_daily_log,message)

	async def send_video2(self,video_path,caption):
		await self.application.bot.send_video(self.chat_SSIP_workshop_alert,video_path,caption=caption,write_timeout=None,supports_streaming=False)

	async def send_video_security(self,video_path,caption):
		await self.application.bot.send_video(self.chat_SSIP_alert_security,video_path,caption=caption,write_timeout=None,supports_streaming=False)