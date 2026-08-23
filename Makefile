.PHONY: test process bandwidth dashboard

test:
	python -m pytest

process:
	python orchestrator.py

bandwidth:
	python bandwidth_demo.py

dashboard:
	streamlit run dashboard.py
