.PHONY: test process bandwidth dashboard edge-replay

test:
	python -m pytest

process:
	python orchestrator.py

bandwidth:
	python bandwidth_demo.py

dashboard:
	streamlit run dashboard.py

edge-replay:
	python edge_agent.py --source input_video.mp4 --gps-csv gps_data.csv \
		--gps-source-type synthetic_demo --profile desktop \
		--output-dir artifacts/edge_live/latest
