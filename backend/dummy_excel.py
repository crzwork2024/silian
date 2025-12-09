import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def create_dummy_excel(file_path: str):
    data = {
        config.EXCEL_COL_PRODUCT_MODEL: ["Model A", "Model B", "Model A", "Model C", "Model B"],
        config.EXCEL_COL_PRODUCT_NUMBER: ["PN001", "PN002", "PN003", "PN004", "PN005"],
        config.EXCEL_COL_FAULT_LOCATION: ["Engine", "Wheel", "Engine", "Brakes", "Transmission"],
        config.EXCEL_COL_FAULT_MODE: ["Overheating", "Flat", "Knocking", "Squeaking", "Slipping"],
        config.EXCEL_COL_FAULT_DESCRIPTION: [
            "Engine runs hot after 30 minutes of operation.",
            "Tire went flat on a bumpy road, very sudden.",
            "Loud knocking sound from engine at idle, especially when cold.",
            "Brakes make a high-pitched squeaking noise when applied gently.",
            "Transmission slips when shifting from 2nd to 3rd gear."
        ],
        config.EXCEL_COL_SOLUTION: [
            "Check coolant levels and inspect radiator for blockages. Consider thermostat replacement.",
            "Replace the tire and check rim for damage. Advise customer on proper tire pressure.",
            "Inspect valve clearances and crankshaft bearings. Might require engine overhaul.",
            "Clean brake calipers and rotors. Apply anti-squeal compound. Check pad wear.",
            "Inspect transmission fluid level and quality. Consider solenoid replacement or full rebuild."
        ]
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.to_excel(file_path, index=False)
    print(f"Dummy Excel file created at {file_path}")

if __name__ == "__main__":
    os.makedirs(config.DATA_DIR, exist_ok=True)
    create_dummy_excel(config.EXCEL_FILE_PATH)
