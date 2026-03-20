.PHONY: install run demo clean

install:
	python3 -m pip install -r requirements.txt

run:
	python3 dp_governor_poc.py

demo:
	python3 dp_governor_poc.py --demo

clean:
	rm -f assets/plot*.png assets/table*.csv
