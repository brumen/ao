INCLUDE_DIR=-I/usr/include/mysql
SPEED_UP=-O3 
GCC=(g++)
GCC_OPTIONS=-std=c++11 -fabi-version=2 -fno-omit-frame-pointer

# mnemonic for compute partial drift/vol
comp_pdv: comp_pdv.c
	$(GCC) $(GCC_OPTIONS) $(INCLUDE_DIR) $(SPEED_UP) -shared -fPIC -o comp_pdv.so comp_pdv.c

comp_pdv-debug: comp_pdv.c
	$(GCC) $(GCC_OPTIONS) -g -c $(INCLUDE_DIR)  -o comp_pdv.o comp_pdv.c

# add calibration functions to the ao library 
add_fcts_to_lib:
	mysql -u brumen -p ao < comp_pdf.sql 

