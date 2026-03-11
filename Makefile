.PHONY: install run demo clean

install:
	pip install -r requirements.txt

run:
	python dp_governor_poc.py

demo:
	python dp_governor_poc.py --demo

clean:
	rm -f assets/plot*.png assets/table*.csv
