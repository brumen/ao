#!/usr/bin/env python
import sys
import cgi
import cgitb
import json
import time 
cgitb.enable(display=0, logdir="/home/brumen/work/zo/inquiry/")  # for troubleshooting

# my local modules 
sys.path.append('/home/brumen/work/zo/')
sys.path.append('/home/brumen/public_html/cgi-bin/')
import config


# complete web generation
# form = cgi.FieldStorage()
f1 = open('/home/brumen/work/zo/inquiry/help.txt', 'a')
f1.write('help\n')
f1.close()

form = json.load(sys.stdin)  # post form is a dict 
lt = time.localtime()
as_of = str(lt.tm_year) + '_' + str(lt.tm_mon) + '_' + str(lt.tm_mday) + '_' + \
        str(lt.tm_hour) + '_' + str(lt.tm_min) + '_' + str(lt.tm_sec)
inquiry_dir = '/home/brumen/work/zo/inquiry/inquiry_solo/'
inquiry_file = inquiry_dir + 'inquiry_' + as_of + '.inq'
# inquiry_pdf = inquiry_dir + 'inquiry_' + as_of + '.pdf'
fo = open(inquiry_file, 'w')
fo.write(json.dumps(form))
fo.close()

# print reutrn 
body = json.dumps({'valid': True})
print "Content-Type: application/json"
print "Length:", len(body)
print ""
print body
