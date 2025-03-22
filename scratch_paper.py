
import smtplib
from email.message import EmailMessage
 
EMAIL_ADDRESS = 'zhatz'
EMAIL_PASSWORD = 'replace with yellow box password'
 
msg = EmailMessage()
msg['Subject'] = 'Subject of mail sent by Python code'
msg['From'] = EMAIL_ADDRESS
msg['To'] = EMAIL_ADDRESS
msg.set_content('Content of mail sent by Python code')
 
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    smtp.send_message(msg)

