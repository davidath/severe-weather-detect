#! /bin/bash
# make init DATASET=weatheriid-over RUN=0
# make init DATASET=weatheriid-under RUN=0
make init DATASET=weatheriid-combine RUN=0 PREF=combine
# make run_cmd CONF=ini/weatheriid-over/tt/tt.ini CONF2=ini/weatheriid-over/tt/ttsae.ini RUN=0 PREF=over
# make run_cmd CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=0 PREF=under
make run_cmd CONF=ini/weatheriid-combine/tt/tt.ini CONF2=ini/weatheriid-combine/tt/ttsae.ini RUN=0 PREF=combine
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    # make run_cmd atest CONF=ini/weatheriid-over/tt/tt.ini CONF2=ini/weatheriid-over/tt/ttsae.ini RUN=$i PREF=over
    # make run_cmd atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under
    make run_cmd atest CONF=ini/weatheriid-combine/tt/tt.ini CONF2=ini/weatheriid-combine/tt/ttsae.ini RUN=$i PREF=combine
done
