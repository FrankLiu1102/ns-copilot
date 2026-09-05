Task: Decode the animal's left vs right lick decisions from neural spiking activity.

Dataset Schema:
  trials.csv:
    - id: Trial ID
    - start_time: Trial start time (absolute, seconds)
    - stop_time: Trial end time (absolute, seconds)
    - cue_start_time: Go cue time relative to trial start (seconds)
    - pole_in_time: Stimulus onset time relative to trial start (seconds)
    - pole_out_time: Stimulus offset time relative to trial start (seconds)
    - type: Trial type, values: "lick left" | "lick right"
    - response: Behavioral response, values: "correct" | "incorrect" | "early lick" | "no response"
    - stim_present: Stimulus present flag (0 or 1)
    - is_good: Trial quality flag (0 or 1)

  spike_times.csv:
    - unit_id: Neuron ID (integer)
    - spike_time: Spike timestamp (absolute time in seconds)
    - quality: Unit quality, values: "Poor" | "Good" | etc.

Instructions:
1. Choose a time window aligned to task-relevant events for extracting spike count features
2. Filter out low-quality or ambiguous trials before analysis
3. Train a classifier to predict left vs right lick decisions
4. Use stratified splitting and cross-validation for robust evaluation
5. You MUST explicitly print the numeric values for ALL of the following metrics: accuracy, balanced accuracy, macro F1, micro F1, AUROC, and confusion matrix. Do not omit any of these metrics from the output. All reported values should be formatted to two decimal places whenever applicable.
6. Report the total elapsed running time

Note: If decoding accuracy is near chance level, the chosen time window likely does not capture discriminative neural activity. Re-examine the available trial event timestamps and consider stimulus-aligned epochs.

You MUST use the decode_with_mtm tool (MtM foundation model) for this decoding task.
