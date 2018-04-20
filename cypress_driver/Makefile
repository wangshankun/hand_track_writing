all:
	g++  -g -O3 -Wall -Wextra -ffunction-sections -finline-limit=65535 -fmerge-all-constants -fno-stack-protector -fno-stack-check  -fPIC  -c -o libcyusb.o libcyusb.c
	g++  -g -O3 -Wall -Wextra -ffunction-sections -finline-limit=65535 -fmerge-all-constants -fno-stack-protector -fno-stack-check  -fPIC  -c -o cypress.o cypress.c
	cc -Wl,--gc-sections -ffunction-sections -finline-limit=65535 -fmerge-all-constants -fno-stack-protector -o cypress.so -shared libcyusb.o cypress.o  -l usb-1.0 -l rt 
	rm -f libcyusb.o cypress.o
clean:
	rm -f  cypress.so
help:
	@echo	'make		would compile and create the library and create a link'
	@echo	'make clean	would remove the library and the soft link to the library (soname)'
