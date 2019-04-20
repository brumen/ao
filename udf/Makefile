INCLUDE_DIR2=-I/usr/include/mysql -I/usr/include/mysql/mysql -I/usr/include/mysql/psi -I/usr/include/mysql/mysql/psi
INCLUDE_DIR=-I/usr/include/mysql
SPEED_UP=-O3 

# mnemonic for compute partial drift/vol
comp_pdv:
	g++ -std=c++11 $(INCLUDE_DIR) $(SPEED_UP) -fabi-version=2 -fno-omit-frame-pointer -shared -fPIC -o comp_pdv.so comp_pdv.c

comp_pdv-g:
	g++ -std=c++11 -g -c $(INCLUDE_DIR) -fabi-version=2 -fno-omit-frame-pointer -o comp_pdv.o comp_pdv.c

# add calibration functions to the ao library 
add_fcts_to_lib:
	mysql -u brumen -p ao < comp_pdf.sql 

