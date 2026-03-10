# Synthetic Dataset Generator for Speech Input Policy Evaluation

This repository module contains the algorithmic generation pipeline (`generate_dataset.py`) for the quantitative evaluation of the Speech Input Policy, specifically designed to bypass the limitations of human adaptation effects in operational scenarios.

## Overview

The generator algorithm produces a "Ground Truth" syntactic baseline of $N=10,000$ formally structured voice commands. By deterministically binding intent, targets, parameters, and environmental variables, it maps perfectly to the architectural definitions outlined in the mathematical constraints of the paper ($c = (intent, target, parameters, confidence)$).

This dataset emulates realistic cognitive limitations and conversational operations under the harsh field constraints of swarm drone control.

## Theoretical Justification

This module aims to prove that the architecture’s resilience against Automatic Speech Recognition (ASR) uncertainty and referential ambiguity is grounded in **formal programmatic constraints** rather than user compensation (i.e., operators subconsciously articulating commands clearer when a system fails).

To validate properties like **Catastrophic Failure Rate (CFR)** and test equations defining the **Contextual Interpretation policy** , the dataset explicitly contains edge cases, grammatical unbalance, and purposeful intent malformations (refer to *Stochastic Properties* below) from the ground truth origin, prior to any acoustic or phonetic perturbation methodologies.

## Lexical Grammar definitions (BNF)

The sentences are mapped explicitly via Backus-Naur Form (BNF). All output strings fundamentally derive from this defined structure:

```bnf
<COMMAND> ::= <TARGET> <ACTION> <PARAM> | <ACTION> <TARGET> | <ACTION> <PARAM>
<TARGET>  ::= "drone one" | "drone two" | "drone three" | "drone four" | "drone five" | "all drones" | "team alpha" | "team bravo" | "team charlie" | ""
<ACTION>  ::= <CRITICAL_INTENT> | <NAV_INTENT> | <QUERY_INTENT>

<CRITICAL_INTENT> ::= "abort mission" | "cancel operation" | "emergency land" | "return to base" | "stop immediately"
<NAV_INTENT>      ::= "go to" | "proceed to" | "fly to" | "change altitude to" | "track"
<QUERY_INTENT>    ::= "report status" | "what is battery of" | "show camera" | "show feed"

<PARAM>   ::= "waypoint one" | "waypoint two" | "point alpha" | "ten meters" | "twenty meters" | "fifty meters" | "one hundred meters" | "target alpha" | "target bravo" | ""
```

## Stochastic Properties & Simulation Realism

The synthetic commands are not uniform (i.e., not all commands are equally likely). The PCFG (Probabilistic Context-Free Grammar) generator actively unbalances generation probabilities to validate mathematical claims from the paper under practical operational realities:

### 1. Realistic Operational Frequencies (Severity Unbalance)
When generating $10,000$ iterations, the generator biases intents:
*   **50% Informational Commands (QUERY)**: Highly frequent interactions (e.g., getting a status update or showing a camera feed) carrying minimal operational risk.
*   **40% Navigational Commands (NAV)**: Mid-level frequency operations altering states (e.g., changing altitude or waypoint navigation).
*   **10% Critical Commands (CRITICAL)**: Rare but fundamental assertions. These intents directly test the system against catastrophic failures (e.g., "abort mission"); they map to the strictest validation safeguards.

### 2. Referential Omissions (Cognitive Load Emulation)
To formally exercise the Mathematical logic defining the **Contextual Inference set** layer ($T(c)$), the parameter `<TARGET>` is strategically dropped (omitted as an empty string `""` / mapped as `"NULL"` in ground truth) in precisely **$15\%$** of the simulated commands ($P_{omit} = 0.15$).

This mimics human cognitive overload, ensuring the ambiguity resolution layer of the architecture is aggressively, mathematically triggered during empirical evaluations.

## Implementation Data Structure

Running `python3 generate_dataset.py` outputs a structured JSON artifact (`base_commands.json`) containing an array of objects mapping the raw string with its canonical parameters. 

**Structure Example:**
```json
{
    "id": 16,
    "raw_text": "drone one return to base",
    "ground_truth": {
        "intent": "RTL",
        "target": "D1",
        "parameters": "NULL",
        "base_confidence": 0.941,
        "severity": "CRITICAL"
    }
}
```

This traceability prevents execution ambiguity in latter testing procedures. The generated output creates a baseline dataset required for the subsequent phonetic logic perturbations.

---

## 3. Option A: Full Acoustic Simulation Pipeline (`run_acoustic_pipeline.py`)

As the "Gold Standard" IEEE methodology, this script bypasses mathematical approximations by forcing a real Automatic Speech Recognition (ASR) engine to fail under physical acoustic stress.

The pipeline executes the following sequence:
1. **TTS Synthesis:** Converts the generated ground truth text into pristine `.mp3` audio files using `gTTS`.
2. **Physical Noise Injection:** Synthesizes realistic 120Hz quadcopter rotor hum using `pydub` low-pass and high-pass filters on White Noise.
3. **SNR Mixing:** Mixes the speech and the rotor noise at specific Signal-to-Noise Ratio (SNR) decibel thresholds (`--snr`).
4. **ASR Transcription:** Feeds the noisy `.wav` file into a real transcription engine (`SpeechRecognition` wrapper) to capture genuine phonetic misunderstandings.

### Execution Arguments
*   `--snr` (float): Defines the Signal to Noise ratio in decibels. The lower the number, the harsher the acoustic environment.
*   `--engine` (string): 
    *   `google` (default): Uses Google Web Speech API for rapid pipeline testing without downloading heavy models.
    *   `whisper`: Uses OpenAI's **Whisper** base model running strictly local/offline. This proves Edge-Computing deployment viability for the IEEE review criteria.
*   `--samples` (int): Number of commands to process. Defaults to 10 for quick testing (the TTS generation implies API overhead). For the final paper results, set to `10000`.

Running `python3 run_acoustic_pipeline.py --engine whisper --snr -5.0` will process commands and output `acoustic_commands.json` containing the real, flawed transcriber outputs mapped against the original `ground_truth`.

---

## 4. Option B: Computational Fault Injection (`inject_faults.py`)

If running 10,000 synthesized audio files is computationally prohibitive, the evaluation demands simulating physical Acoustic Signal-to-Noise Ratio (SNR) degradation mathematically without introducing ad-hoc synthetic biases. This is achieved via the `inject_faults.py` execution.

### Phonetic Levenshtein Confusion Matrix
Instead of applying naive $5\%$ random semantic corruptions, the script extracts ARPABET phonemes (ARPABET is a phonetic alphabet used to represent the sounds of words) for every vocabulary word defined in the grammar using `pronouncing` (python library for CMUdict - Carnegie Mellon University Pronouncing Dictionary) library.

The probability of the ASR (Automatic Speech Recognition) confusing **Command A** for **Command B** is mapped as inversely proportional to the **Phonetic Levenshtein Distance** (a metric measuring the minimum number of single-phoneme edits—insertions, deletions, or substitutions—required to change one word's pronunciation into another) calculated between their sound arrays via the `jellyfish` library (https://github.com/jamesturk/jellyfish - Library for phonetic algorithms). For example, "Report Status" and "Abort Mission" possess a higher statistical likelihood of interchanging under noise than completely phonetically dissimilar intents.

### Confidence Degradation Formulation
The ASR confidence output ($c_{conf}$) is degraded corresponding to the theoretical model established in Section 4.2 of the paper:
$$\hat{c}_{conf} = \max(0, \min(1, c_{conf} - |\mathcal{N}(0, \sigma^2)|))$$

### Fault Injection Payload
Running `python3 inject_faults.py --sigma 0.3` produces `noisy_commands.json`. The output dataset encapsulates the corrupted string the ASR simulated, the degraded confidence factor, the erroneously mapped intent, and precisely preserves the original immutable `ground_truth` block.
This dataset is the final payload injected into the Speech Input Policy module to calculate the absolute **Catastrophic Failure Rate (CFR)** and **Spurious Execution Rate (SER)** performance boundaries.
