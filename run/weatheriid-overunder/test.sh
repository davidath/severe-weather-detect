#! /bin/bash
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make atest CONF=ini/weatheriid-over/tt/tt.ini CONF2=ini/weatheriid-over/tt/ttsae.ini RUN=$i PREF=over
    # make atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under
    # make atest CONF=ini/weatheriid-combine/tt/tt.ini CONF2=ini/weatheriid-combine/tt/ttsae.ini RUN=$i PREF=combine
done
