import sys, os
import uuid
import cgi
import cgitb
import json
import time
import numpy as np
import subprocess
import getpass 
# the following 4 modules are for sending e-mail w/ pdf attached 
import smtplib
import mimetypes
import email
import email.mime.application


contacts_file = 'contacts.txt'
solicit_template = 'solicitation_template.tex'

def send_email_to_client(client_email, invoice_pdf):
    """
    sends the e-mail to client 
    """
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(airoptions_gmail_acct, airoptions_gmail_pass)
    # plain text part of the message 
    msg = email.mime.Multipart.MIMEMultipart()
    msg['Subject'] = 'AirOptions purchase'
    msg['From'] = 'airoptions.llc@gmail.com'
    msg['To'] = client_email
    body = email.mime.Text.MIMEText("""Thank you for your purchase with AirOptions. 
    Your receipt is attached to this e-mail.""")
    msg.attach(body)
    # pdf attachment
    fp = open(invoice_pdf, 'rb')
    att = email.mime.application.MIMEApplication(fp.read(), _subtype="pdf")
    fp.close()
    att.add_header('Content-Disposition', 'attachment', filename='invoice.pdf')
    msg.attach(att)
    if '@' in client_email:
        server.sendmail('airoptions.llc@gmail.com', client_email, msg.as_string())
    # otherwise dont send mail 
    server.close()


#
# main file, presenting everything 
# 
# lt = time.localtime()  # local-time
# as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
#        str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
# lt_slash = str(lt.tm_mon) + '/' + str(lt.tm_mday) + '/' +  str(lt.tm_year)
# lt_str = ds.convert_dateslash_str(lt_slash)

# writing all files, etc 
base_dir = '/home/brumen/work/ao/doc/letters/' # base dir


def generate_pdf(company, contact_person, address, city, zipcode):
    print "WRINTING", company
    solicitation_fo = open(solicit_template, 'r')
    contact_fo = open('pdfs/' + company + '.tex', 'w')
    for line in solicitation_fo:
        if "COMPANY" in line:
            contact_fo.write(line.replace('COMPANY', company))
        elif "CONTACTPERSON" in line:
            contact_fo.write(line.replace('CONTACTPERSON', contact_person))
        elif "ADDRESS1" in line:
            contact_fo.write(line.replace('ADDRESS1', address))
        elif "CITY" in line:
            contact_fo.write(line.replace('CITY', city))
        elif "ZIPCODE" in line:
            contact_fo.write(line.replace('ZIPCODE', zipcode))
        else:
            contact_fo.write(line)
    contact_fo.close()
    solicitation_fo.close()

    print "CONVERTING"
    # check if the file invoice_tex is alfanumeric
    cmd = 'pdflatex -output-directory pdfs/ ' + ' ' + '\"pdfs/' + company + '.tex\" > /dev/null'
    print "CMD: ", cmd
    os.system(cmd)
        

contacts_fo = open(contacts_file, 'r')    
for entry in contacts_fo:
    company, contact_person, address, city, zipcode = entry.split(',')
    generate_pdf(company, contact_person, address, city, zipcode)
contacts_fo.close()

