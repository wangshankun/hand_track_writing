#!/bin/bash
cp -f cyusb.conf /etc/
cp -f 88-cyusb.rules /etc/udev/rules.d/

make

# Remove stale versions of the libcyusb library

# Copy the libcyusb library into the system library folders.
rm  -r /usr/lib/cypress.so
cp -f cypress.so /usr/lib

rm -f cypress.so
