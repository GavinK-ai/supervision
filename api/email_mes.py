import pymssql
import pandas as pd
from datetime import date,datetime
import logging

server = 'spl-vmsqlqa19a'
user = 'SPL-MESService'
password = 'MESService'
database ='MailCenterQA'
MessageHeaderId = ''

attachment_server_folder = '\\172.30.20.13\AUTOMAIL\QA'
attachment_server_user = 'shimanoace\spl_btsservice'
attachment_server_password = 'gloBTS'


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

class MESEmail:
    def __init__(self, host=server, user=user, password=password, database=database, port = 1433, name=name, address=address, recipient=recipient, 
                 copyRecipient=copyRecipient, dev_recipient=dev_recipient, replyTo=replyTo, subject=subject, body=body, bodyFormat=bodyFormat, attachment=attachment,
                 attachment_server_folder=attachment_server_folder, attachment_server_user=attachment_server_user, attachment_server_password= attachment_server_password):
        
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.cursor = None

        self.attachment_server_folder = attachment_server_folder
        self.attachment_server_user = attachment_server_user
        self.attachment_server_password = attachment_server_password

        self.messageHeaderId = None
        self.newHeaderId = None

        self.name = name
        self.address = address
        self.recipient = recipient
        self.dev_recipient = dev_recipient
        self.copyRecipient = copyRecipient
        self.replyTo = replyTo
        self.subject = subject
        self.body = body
        self.bodyFormat = bodyFormat
        self.attachment = attachment

    def connect(self):
        self.connection = pymssql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )
        self.cursor = self.connection.cursor()
    
    #def connect_attachment_server(self):
    #    self.connection = pymssql.connect(
    #        host = self.host
    #        user = self.attachment_server_user
    #        password = self.attachment_server_password
    #        database = self.database
    #    )

    def execute_query(self, query, values=None):
        try:
            self.connect()
            self.cursor.execute(query, values)
            self.connection.commit()
            return self.cursor.fetchall()
        except pymssql.Error as e:
            print(f"Error: {e}")
            self.connection.rollback()
        finally:
            self.close()
            
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def getMessageId(self):
        try:
            sql_getId = "SELECT MessageHeaderId FROM [MailCenterQA].[dbo].[ctMessageHeader]"
            #all_id = self.execute_query(sql_getId,('41415'))
            self.connect()
            self.cursor = self.connection.cursor()
            self.cursor.execute(sql_getId,('41415',))
            all_id = self.cursor.fetchall()
            df_id = pd.DataFrame(all_id,index = None)
            latest_id = str(df_id.iloc[-1][0])
            # print(f'Last Id {latest_id}')
            date_str = latest_id[0:8]
            latest_id_date = datetime.strptime(date_str,"%Y%m%d").date()
            date_now = date.today()
            # print(f'Last Date: {latest_id_date}\nToday Date: {date_now}')
            if latest_id_date==date_now:
                new_msg_id = int(latest_id[9:13])+1
                self.newHeaderId = latest_id[0:9]+str(new_msg_id).zfill(4)
            else:
                self.newHeaderId = str(date_now).replace('-','')+'A0001'
            logging.debug(f'Get Header Successful {self.newHeaderId}')
        except Exception as e:
            logging.exception(f'Get Last Header Id Error {e}')

    def getControlNumber (self):
        try:
            self.connect()
            self.cursor = self.connection.cursor()
            self.cursor.execute('DECLARE @Out AS VARCHAR(50)')
            # self.cursor.callproc('dbo.spGetControlNumber',('1','MessageHeaderId','','','','','@Out'))
            self.cursor.execute('dbo.spGetControlNumber',('1','MessageHeaderId','','','','','@Out'))
            self.cursor.execute('SELECT @Out AS OutputValue')
            result = self.cursor.fetchall()
            print(f'Result: {result}')
        except Exception as e:
            logging.exception(f'Call ctMessageHeader Fail {e}')
        finally:
            self.close()

    def sendAlertEmail(self, stream_id, violation, attachment_id, link): #Send Alert Email
        pass
        # try:
        #     # self.newHeaderId = self.getMessageId()
        #     self.getMessageId()
        #     if stream_id == 10:
        #         floor = 'B1'
        #     else:
        #         floor = 'L1'
        #     self.subject = f'SSIP IFMS ALERT: C-{floor}-{stream_id} {violation} DETECTED'  
        #     # self.subject = f'TESTING AUTOMATED EMAIL WITH VIDEO {stream_id} {violation}'
        #     self.Body = f'''Dear Recipient,
        #                 <br /><br />{violation} has been detected in the area monitored by C-{floor}-{stream_id}.
        #                 <br /><br />Attachment:  {attachment_id}
        #                 <br />Flie Location:  {link}
        #                 <br /><br /><i>This is a system generated email. Please do not reply.</i>'''

        #     sql_sendLogEmail = ("INSERT INTO ctMessageHeader (MessageHeaderId, FromName, FromAddress, Recipient, CopyRecipient, ReplyTo, Subject, Body, BodyFormat) "
        #                         "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
            
        #     values = (self.newHeaderId, self.name, self.address, self.dev_recipient, self.copyRecipient, self.replyTo, self.subject, self.Body, self.bodyFormat)
        #     self.execute_query(sql_sendLogEmail, values)
        #     logging.info(f'Send Alert Email Successful')

        # except Exception as e:
        #     logging.exception(f'Send Alert Email Failed')

    def sendLogEmail(self,stream_id,attachment_id,link): #Send Log Email
        try:
            self.getMessageId()
            if stream_id == 10:
                floor = 'B1'
            else:
                floor = 'L1'
            self.subject = f'SSIP IFMS LOG: C-{floor}-{stream_id}'
            self.Body = f'''Dear Recipient,
                        <br /><br />Logging Data from C-{floor}-{stream_id}.
                        <br /><br />Attachment:  {attachment_id}
                        <br />Flie Location:  {link}
                        <br /><br /><i>This is a system generated email. Please do not reply.</i>'''
            sql_sendAlertEmail = ("INSERT INTO ctMessageHeader (MessageHeaderId, FromName, FromAddress, Recipient, CopyRecipient, ReplyTo, Subject, Body, BodyFormat) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
            values = (self.newHeaderId, self.name, self.address, self.recipient, self.copyRecipient, self.replyTo, self.subject, self.Body, self.bodyFormat)
            self.execute_query(sql_sendAlertEmail, values)
            logging.info(f'Send Alert Email Successful')
        except Exception as e:
            logging.exception(f'Send Log Email Failed')

    def sendTroubleshootEmail(self): #Send Troubleshoot Email
        pass

    def copyAttachment(self, attachment_file_link):
        pass



    

if __name__ == "__main__":
    # Create an instance of MESEmail
    m = MESEmail()

    # Call the sendAlertEmail method
    #(Stream Id, Violation, Img Attachment Id, Video Share Folder Link)
    # m.sendAlertEmail('01','Testing','Img Attachment Id','Video Saved Test Link')
    m.getControlNumber()
        
