#! /bin/bash
# make init DATASET=weatheriid-allone/windF RUN=0 PREF=wind-alloneF
# make init DATASET=weatheriid-allone/windT RUN=0 PREF=wind-alloneT

# make init DATASET=weatheriid-allone/floodW RUN=0 PREF=flood-alloneW
# make init DATASET=weatheriid-allone/floodT RUN=0 PREF=flood-alloneT

# make init DATASET=weatheriid-allone/tornadoW RUN=0 PREF=tornado-alloneW
# make init DATASET=weatheriid-allone/tornadoF RUN=0 PREF=tornado-alloneF

# make run_cmd CONF=ini/weatheriid-allone/windF/windF/windF.ini CONF2=ini/weatheriid-allone/windF/windF/windFsae.ini RUN=0 PREF=wind-alloneF
# make run_cmd CONF=ini/weatheriid-allone/windT/windT/windT.ini CONF2=ini/weatheriid-allone/windT/windT/windTsae.ini RUN=0 PREF=wind-alloneT

# make run_cmd CONF=ini/weatheriid-allone/floodW/floodW/floodW.ini CONF2=ini/weatheriid-allone/floodW/floodW/floodWsae.ini RUN=0 PREF=flood-alloneW
# make run_cmd CONF=ini/weatheriid-allone/floodT/floodT/floodT.ini CONF2=ini/weatheriid-allone/floodT/floodT/floodTsae.ini RUN=0 PREF=flood-alloneT

# make run_cmd CONF=ini/weatheriid-allone/tornadoW/tornadoW/tornadoW.ini CONF2=ini/weatheriid-allone/tornadoW/tornadoW/tornadoWsae.ini RUN=0 PREF=tornado-alloneW
# make run_cmd CONF=ini/weatheriid-allone/tornadoF/tornadoF/tornadoF.ini CONF2=ini/weatheriid-allone/tornadoF/tornadoF/tornadoFsae.ini RUN=0 PREF=tornado-alloneF

START=0
let END=START+3
for (( i=$START; i<=$END; i++ ));do
    # make run_cmd CONF=ini/weatheriid-allone/windF/windF/windF.ini CONF2=ini/weatheriid-allone/windF/windF/windFsae.ini RUN=$i PREF=wind-alloneF
    # make test CONF=ini/weatheriid-allone/windF/windF/windF-test.ini RUN=$i PREF=wind-alloneF
    # make run_cmd CONF=ini/weatheriid-allone/windT/windT/windT.ini CONF2=ini/weatheriid-allone/windT/windT/windTsae.ini RUN=$i PREF=wind-alloneT
    # make test CONF=ini/weatheriid-allone/windT/windT/windT-test.ini RUN=$i PREF=wind-alloneT

    # make run_cmd CONF=ini/weatheriid-allone/floodW/floodW/floodW.ini CONF2=ini/weatheriid-allone/floodW/floodW/floodWsae.ini RUN=$i PREF=flood-alloneW
    # make test CONF=ini/weatheriid-allone/floodW/floodW/floodW-test.ini RUN=$i PREF=flood-alloneW
    # make run_cmd CONF=ini/weatheriid-allone/floodT/floodT/floodT.ini CONF2=ini/weatheriid-allone/floodT/floodT/floodTsae.ini RUN=$i PREF=flood-alloneT
    # make test CONF=ini/weatheriid-allone/floodT/floodT/floodT-test.ini RUN=$i PREF=flood-alloneT

    # make run_cmd CONF=ini/weatheriid-allone/tornadoW/tornadoW/tornadoW.ini CONF2=ini/weatheriid-allone/tornadoW/tornadoW/tornadoWsae.ini RUN=$i PREF=tornado-alloneW
    # make test CONF=ini/weatheriid-allone/tornadoW/tornadoW/tornadoW-test.ini RUN=$i PREF=tornado-alloneW
    # make run_cmd CONF=ini/weatheriid-allone/tornadoF/tornadoF/tornadoF.ini CONF2=ini/weatheriid-allone/tornadoF/tornadoF/tornadoFsae.ini RUN=$i PREF=tornado-alloneF
    # make test CONF=ini/weatheriid-allone/tornadoF/tornadoF/tornadoF-test.ini RUN=$i PREF=tornado-alloneF

done
