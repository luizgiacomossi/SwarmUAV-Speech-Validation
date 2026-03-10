import json
import random
import math
import argparse
import jellyfish
import pronouncing

def get_phonemes(text):
    """
    Converts a text string into a list of ARPABET phonemes using CMUdict.
    If a word is not found, it falls back to a crude character-based approximation 
    just to keep the pipeline from failing, but standard commands should be in the dict.
    """
    words = text.lower().split()
    phonemes = []
    for w in words:
        p = pronouncing.phones_for_word(w)
        if p:
            # Use the first pronunciation variant
            phonemes.extend(p[0].split())
        else:
            # Fallback for out-of-vocabulary (unlikely for these commands)
            phonemes.extend(list(w))
    return phonemes

def phonetic_distance(intent1, intent2):
    """
    Calculates the Levenshtein distance between the phonetic representations 
    of two commands.
    """
    p1 = get_phonemes(intent1)
    p2 = get_phonemes(intent2)
    
    # We join them as space-separated strings and calculate Levenshtein
    # Jellyfish's levenshtein_distance works on strings.
    s1 = " ".join(p1)
    s2 = " ".join(p2)
    return jellyfish.levenshtein_distance(s1, s2)

def build_confusion_matrix(intents_dict):
    """
    Builds a confusion matrix where the probability of substituting intent A for intent B
    is inversely proportional to their phonetic Levenshtein distance.
    """
    # Flatten all intent texts into a single list
    all_intent_texts = []
    for category, items in intents_dict.items():
        for text, _ in items:
            all_intent_texts.append(text)
            
    matrix = {}
    for i1 in all_intent_texts:
        matrix[i1] = {}
        total_weight = 0
        for i2 in all_intent_texts:
            if i1 == i2:
                continue
            
            dist = phonetic_distance(i1, i2)
            # Prevent division by zero. If distance is 0 but strings are different (rare), 
            # treat it as highly confusable.
            if dist == 0:
                weight = 100.0
            else:
                # Inversely proportional weight. Squaring makes dissimilar ones even less likely.
                weight = 1.0 / (dist ** 2)
            
            matrix[i1][i2] = weight
            total_weight += weight
            
        # Normalize weights into probabilities
        for i2 in matrix[i1]:
            matrix[i1][i2] /= total_weight
            
    return matrix


def inject_faults(input_file, output_file, noise_sigma=0.3):
    """
    Simulates ASR degradation purely mathematically (Computational Pipeline).
    Validates equations from Section 4.2 of the paper.
    """
    print(f"Loading '{input_file}'...")
    with open(input_file, "r") as f:
        dataset = json.load(f)
        
    print(f"Building Phonetic Confusion Matrix (sigma={noise_sigma})...")
    # Need to reconstruct the INTENTS dictionary from generate_dataset to build the matrix
    # Hardcoded here for simplicity, but in a real module they'd share a config.
    from generate_dataset import INTENTS
    
    # We need a reverse mapping to get the 'true_intent' if the text gets corrupted
    text_to_label = {}
    for cat, items in INTENTS.items():
         for text, label in items:
             text_to_label[text] = label
             
    confusion_matrix = build_confusion_matrix(INTENTS)
    
    corrupted_count = 0
    degraded_dataset = []
    
    for item in dataset:
        c_conf = item["ground_truth"]["base_confidence"]
        original_text = item["raw_text"]
        
        # 1. Bounded Confidence Degradation (Validation of Section 4.2 logic)
        # Equation: c_conf_hat = max(0, min(1, c_conf - |N(0, sigma^2)|))
        noise = abs(random.gauss(0, noise_sigma))
        c_conf_hat = max(0.0, min(1.0, c_conf - noise))
        
        # 2. Phonetic Intent Corruption
        # If the environment is noisy enough, what the user actually said (Intent A) 
        # might be transcribed as another command (Intent B) with probability 
        # based on phonetic distance.
        # We increase probability of corruption based on the noise applied.
        
        # The greater the noise degradation, the higher the chance of ASR misclassification
        degradation_amount = c_conf - c_conf_hat
        
        corrupted_text = original_text
        corrupted_label = item["ground_truth"]["intent"]
        
        # Extract the original action text from the utterance for confusion matching
        # This is a bit tricky because the utterance could be <TARGET> <ACTION> <PARAM>
        # For simplicity in this simulation, we'll try to find which intent text is in the original text
        original_action_text = None
        for text in text_to_label.keys():
             if text in original_text:
                 original_action_text = text
                 break
                 
        if original_action_text and degradation_amount > 0.4: # Arbitrary threshold for "severe noise"
             # Pick a new intent text based on the confusion matrix probabilities
             possible_corruptions = list(confusion_matrix[original_action_text].keys())
             probabilities = list(confusion_matrix[original_action_text].values())
             
             # Randomly decide if corruption actually happens, scaled by noise
             if random.random() < degradation_amount:
                 new_action_text = random.choices(possible_corruptions, weights=probabilities)[0]
                 corrupted_text = original_text.replace(original_action_text, new_action_text)
                 corrupted_label = text_to_label[new_action_text]
                 corrupted_count += 1
        
        # Save the simulation output
        degraded_dataset.append({
            "id": item["id"],
            "asr_transcription": corrupted_text,
            "asr_confidence": round(c_conf_hat, 3),
            "inferred_intent": corrupted_label, # The label the system *thinks* it heard
            "ground_truth": item["ground_truth"] # Immutable, for calculating CFR/SER later
        })

    with open(output_file, "w") as f:
        json.dump(degraded_dataset, f, indent=4)
        
    print(f"Dataset successfully corrupted and saved to '{output_file}'.")
    print(f"Total Intents Phonetically Corrupted: {corrupted_count} / {len(dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic Fault Injection (Phonetic Distance)")
    parser.add_argument("--input", default="base_commands.json", help="Clean dataset JSON")
    parser.add_argument("--output", default="noisy_commands.json", help="Corrupted dataset JSON")
    parser.add_argument("--sigma", type=float, default=0.3, help="Noise standard deviation")
    
    args = parser.parse_args()
    
    # Deterministic seed for scientific reproducibility
    random.seed(42)
    inject_faults(args.input, args.output, args.sigma)
