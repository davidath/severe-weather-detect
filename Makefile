CONF=
RUN=
EXP=
PREF=
CONF2= 
START=
DATASET=

clean::
	rm -f *.npy
	rm -f *.txt

run::
	./train.py $(CONF) $(RUN) $(CONF2) | tee $(PREF)train$(RUN).txt

test::
	./test.py $(CONF) $(RUN) | tee $(PREF)test$(RUN).txt

run_cmd::
	 ./train.py $(CONF) $(RUN) $(CONF2)

rename_test::
	 for i in `ls TEST_$(PREF)*`;do mv $$i `echo $$i | awk -F "TEST_" '{print $$2}'`;done

gather::
	mkdir $(PREF) && mv *$(PREF)* $(PREF)/

init::
	./train.py ini/$(DATASET)/px.ini $(RUN) | tee $(PREF)pxtrain$(RUN).txt

atest::
	./anomaly_test.py $(CONF) $(RUN) | tee $(PREF)test$(RUN).txt
