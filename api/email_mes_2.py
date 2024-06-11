import requests
import os
import asyncio

# Replace with your actual API endpoint
apiurl = "http://spl-vmamesdev:808/MasterDataProvider/api/MailCenter/SendEmail"
headers = {
        'Authorization': '59A2171B-2244-4F6D-8368-2424C5075DD9'
    }
name = 'IFMS'
address = 'ifms@shimano.com.sg'
recipient = 'gavinlim@shimano.com.sg'
dev_recipient = 'gavinlim@shimano.com.sg;jareltan@shimano.com.sg'
#recipient = 'gavinlim@shimano.com.sg;limteckzhi@shimano.com.sg'
copyRecipient = ''
replyTo = 'limteckzhi@shimano.com.sg'
subject = 'SSIP IFMS ALERT: C-L1-XX TRESPASSING/SPEEDIND/ DETECTED!'
body = 'Dear Recipient,<br /><br />A Trespassing/Speeding/Overtime Parking has been detected in the area monitored by C-L1-XX.<br /><br />Attachement:Link<br /><br /><i>This is a system generated email. Please do not reply</i>'
bodyFormat = 'HTML'
attachment = '0'
body_string = ''
data = {
        'FromName': "IFMS",
        'FromAddress': "SENDER@SHIMANO.COM.SG",
        'Recipient': "GAVINLIM@SHIMANO.COM.SG",
        'Subject': "Mail Without Attachment From Python", 
        'Body': body_string,
        'CreateUser': "SPLC2022",
        'CopyRecipient': "LIMTECKZHI@SHIMANO.COM.SG;KELBIN@SHIMANO.COM.SG"
        # Add other parameters as needed
    }

class MESEmail:
	def __init__(self) -> None:
		self.apiurl = apiurl
		self.headers = headers

	def send_email_without_attachment(self,stream_id,violation,link):
		try:
			if stream_id == 10:
				floor = 'B1'
			else:
				floor = 'L1'

			body_string = f'''
							Dear Recipient,
							<br /><br />{violation} has been detected in the area monitored by C-{floor}-{stream_id}.
							<br />Flie Location:  {link}
							<br /><br /><i>This is a system generated email. Please do not reply.</i>
							'''
			data = {
				'FromName': "IFMS",
				'FromAddress': "SENDER@SHIMANO.COM.SG",
				'Recipient': "GAVINLIM@SHIMANO.COM.SG",
				'Subject': "Mail Without Attachment From Python", 
				'Body': body_string,
				'CreateUser': "SPLC2022",
				'CopyRecipient': "LIMTECKZHI@SHIMANO.COM.SG;KELBIN@SHIMANO.COM.SG"
				# Add other parameters as needed
			}
			response = requests.post(self.apiurl,headers=self.headers,data=data)
			# LOGGER.debug(f'Send Email without attachment sucessful\nInfo: C-{floor}-{stream_id} {violation}\nFile: {link}')
		except:
			print(f'Send Email without attachment failed')

	def send_email_with_attachment():
		pass

if __name__ == "__main__":
	pass


