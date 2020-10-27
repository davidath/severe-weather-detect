#! /bin/bash
make init DATASET=weatheriid-under RUN=0 PREF=under4k
make run_cmd CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=0 PREF=under4k
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make run_cmd atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under4k
done

bash ini/undersamplereplace.sh ini/weatheriid-under/ under4k under10k
bash ini/undersamplereplace.sh ini/weatheriid-under/ data_undersample4k.npz data_undersample10k.npz
bash ini/undersamplereplace.sh ini/weatheriid-under/ evi_undersample4k.npz evi_undersample10k.npz

make init DATASET=weatheriid-under RUN=0 PREF=under10k
make run_cmd CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=0 PREF=under10k
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make run_cmd atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under10k
done

bash ini/undersamplereplace.sh ini/weatheriid-under/ under10k under20k
bash ini/undersamplereplace.sh ini/weatheriid-under/ data_undersample10k.npz data_undersample20k.npz
bash ini/undersamplereplace.sh ini/weatheriid-under/ evi_undersample10k.npz evi_undersample20k.npz

make init DATASET=weatheriid-under RUN=0 PREF=under20k
make run_cmd CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=0 PREF=under20k
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make run_cmd atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under20k
done

bash ini/undersamplereplace.sh ini/weatheriid-under/ under20k under30k
bash ini/undersamplereplace.sh ini/weatheriid-under/ data_undersample20k.npz data_undersample30k.npz
bash ini/undersamplereplace.sh ini/weatheriid-under/ evi_undersample20k.npz evi_undersample30k.npz

make init DATASET=weatheriid-under RUN=0 PREF=under30k
make run_cmd CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=0 PREF=under30k
START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do 
    make run_cmd atest CONF=ini/weatheriid-under/tt/tt.ini CONF2=ini/weatheriid-under/tt/ttsae.ini RUN=$i PREF=under30k
done
