#! /bin/bash

START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make test CONF=ini/weatheriid-allone/windF/windF/windF-test.ini RUN=$i PREF=wind-alloneF
    # make test CONF=ini/weatheriid-allone/windT/windT/windT-test.ini RUN=$i PREF=wind-alloneT
done
