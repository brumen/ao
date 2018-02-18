##
## Makefile
##  
## Made by brumen
## Login   <brumen/prasic>
##


compile:
#	python pricers_setup.py build_ext --inplace


# backups the public_html & ao folders 
backup:
	tar czvf /home/brumen/archive/ao/backups/ao_all-`date +%F`.tar.gz /home/brumen/work/ao

# copies the backed-up files to stick
copy_stick:
	cp /home/brumen/archive/ao/ao/public_html_archive/public_html-`date +%F`.tar.gz /media/brumen/backup/backup/
	cp /home/brumen/archive/ao/ao/ao_archive/work_ao-`date +%F`.tar.gz /media/brumen/backup/backup/


backup_db:
	mysqldump --user=brumen --host=localhost --protocol=tcp --port=3306 --default-character-set=utf8 --routines --result-file=/home/brumen/archive/ao/ao/db_archive/ao_mysql-`date +%F`.sql  "ao" 
	gzip /home/brumen/archive/ao/ao/db_archive/ao_mysql-`date +%F`.sql


copy_db_stick:
	cp /home/brumen/archive/ao/ao/db_archive/ao_mysql-`date +%F`.sql.gz /media/brumen/backup/backup/
