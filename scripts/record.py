# record.py
import sys                                                                                        
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
 
import yaml
from src.orchestrator.recording_orchestrator import RecordingOrchestrator



config = yaml.safe_load(open("config/record_data_binance.yaml"))

# If the file already exists check whether the user wants to overwrite it or not
if os.path.exists(config["recording"]["filepath"]):
    response = input(f"File {config['recording']['filepath']} already exists. Do you want to overwrite it? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without overwriting the file.")
        sys.exit(0)

orchestrator = RecordingOrchestrator(config)
orchestrator.run()  # Ctrl+C to stop
