# record.py
import sys                                                                                        
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
 
import yaml
from src.orchestrator.recording_orchestrator import RecordingOrchestrator

config = yaml.safe_load(open("config/record_data_binance.yaml"))
orchestrator = RecordingOrchestrator(config)
orchestrator.run()  # Ctrl+C to stop
