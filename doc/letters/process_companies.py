import sys, os

# the following 4 modules are for sending e-mail w/ pdf attached
import smtplib
import email
import email.mime.application

from ao_codes import airoptions_gmail_acct, airoptions_gmail_pass

contacts_file    = 'contacts.txt'
solicit_template = 'solicitation_template.tex'
# writing all files, etc
base_dir = '/home/brumen/work/ao/doc/letters/' # base dir


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
    with open(invoice_pdf, 'rb') as fp:
        att = email.mime.application.MIMEApplication(fp.read(), _subtype="pdf")

    att.add_header('Content-Disposition', 'attachment', filename='invoice.pdf')
    msg.attach(att)
    if '@' in client_email:
        server.sendmail('airoptions.llc@gmail.com', client_email, msg.as_string())

    # otherwise dont send mail
    server.close()


def generate_pdf(company, contact_person, address, city, zipcode):
    """ Generate pdf from TODO: FINISH THIS

    """
    print("WRINTING", company)
    with open(solicit_template, 'r') as solicitation_fo, open('pdfs/' + company + '.tex', 'w') as contact_fo:
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

    # check if the file invoice_tex is alfanumeric
    cmd = 'pdflatex -output-directory pdfs/ ' + ' ' + '\"pdfs/' + company + '.tex\" > /dev/null'
    os.system(cmd)
        

with open(contacts_file, 'r') as contacts_fo:
    for entry in contacts_fo:
        company, contact_person, address, city, zipcode = entry.split(',')
        generate_pdf(company, contact_person, address, city, zipcode)
