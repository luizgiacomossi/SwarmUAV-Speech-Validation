import random
import json
import os

# Configuration based on Section 4.1 "Experimental Architecture"
# The paper dictates N=10,000 for statistical significance
NUM_COMMANDS = 10000
OUTPUT_FILE = "base_commands.json"

# PCFG Lexical Definitions
INTENTS = {
    "CRITICAL": [
        ("abort mission", "ABORT"), 
        ("cancel operation", "ABORT"), 
        ("emergency land", "LAND"), 
        ("return to base", "RTL"),
        ("stop immediately", "STOP")
    ],
    "NAV": [
        ("go to", "GOTO"), 
        ("proceed to", "GOTO"), 
        ("fly to", "GOTO"),
        ("change altitude to", "ALTITUDE"),
        ("track", "TRACK")
    ],
    "QUERY": [
        ("report status", "STATUS"), 
        ("what is battery of", "BATTERY"), 
        ("show camera", "VIDEO"),
        ("show feed", "VIDEO")
    ]
}

TARGETS = [
    ("drone one", "D1"), 
    ("drone two", "D2"), 
    ("drone three", "D3"), 
    ("drone four", "D4"), 
    ("drone five", "D5"), 
    ("all drones", "ALL"),
    ("team Red", "RED"),
    ("team Blue", "BLUE"),
    ("team Green", "GREEN")
]

# Actionable parameters depending on the intent
PARAMS = {
    "GOTO": [("waypoint one", "WP1"), ("waypoint two", "WP2"), ("point alpha", "PT_A")],
    "ALTITUDE": [("ten meters", "10M"), ("twenty meters", "20M"), ("fifty meters", "50M"), ("one hundred meters", "100M")],
    "TRACK": [("target alpha", "T_ALPHA"), ("target bravo", "T_BRAVO")]
}

def generate_synthetic_dataset(n_samples: int, output_path: str):
    """
    Generates a ground-truth algorithmic dataset mapped by a Probabilistic Context-Free Grammar.
    Allows exact verification of ASR corruption and contextual inference equations.
    """
    dataset = []
    
    for i in range(n_samples):
        # 1. Severity Class Selection (Unbalanced distribution: 50% Query, 40% Nav, 10% Critical)
        severity = random.choices(["QUERY", "NAV", "CRITICAL"], weights=[0.5, 0.4, 0.1])[0]
        
        # 2. Specific Intent and Ground Truth Selection
        text_intent, true_intent = random.choice(INTENTS[severity])
        
        # 3. Referential Omission (15% probability)
        # Specifically designed to validate the Contextual Interpretation policy (Eq. 29)
        if random.random() < 0.15:
            text_target, true_target = ("", "NULL")
        else:
            text_target, true_target = random.choice(TARGETS)
        
        # 4. Extract parameters if intent allows for them
        text_param, true_param = "", "NULL"
        if severity == "NAV" and true_intent in PARAMS:
            text_param, true_param = random.choice(PARAMS[true_intent])
            
        # 5. PCFG Combination logic:
        # Default structure: <TARGET> <ACTION> <PARAM> (e.g., "drone one go to waypoint one")
        parts = [text_target, text_intent, text_param]
        
        # Natural language adjustment: Queries usually lead with the action
        # e.g., "report status drone one" instead of "drone one report status"
        if severity == "QUERY":
            parts = [text_intent, text_target, text_param]
            
        utterance_words = [p for p in parts if p.strip() != ""]
        utterance = " ".join(utterance_words).strip()
        
        # Initial clear speech confidence before environmental Gaussian noise injection
        base_confidence = round(random.uniform(0.92, 0.99), 3)
        
        # 6. Export exact state mapping to validate CFR, SER, IRR logic downstream
        dataset.append({
            "id": i,
            "raw_text": utterance,
            "ground_truth": {
                "intent": true_intent,
                "target": true_target,
                "parameters": true_param,
                "base_confidence": base_confidence,
                "severity": severity
            }
        })
        
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Dataset successfully generated with {n_samples} commands at {output_path}.")

if __name__ == "__main__":
    # Deterministic temporal seed guarantees scientific reproducibility (Criterion 7)
    random.seed(42)
    generate_synthetic_dataset(NUM_COMMANDS, OUTPUT_FILE)
