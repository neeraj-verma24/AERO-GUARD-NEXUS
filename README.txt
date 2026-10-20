AERO-GUARD NEXUS — STEP 68 RUL OUTPUT CAP CONSISTENCY

Purpose
- Keep inference outputs aligned with the existing 125-cycle capped RUL target.
- Clip baseline and each ensemble member prediction to [0, 125] before calculating the ensemble mean, spread, and p10/p90 values.
- Do not retrain or overwrite saved models.

Important limits
- This is an output-domain consistency change, not a model accuracy improvement by itself.
- Ensemble disagreement and p10/p90 remain descriptive and are NOT calibrated uncertainty intervals.
- NASA C-MAPSS FD001 is simulated benchmark data, not actual aircraft sensor telemetry.
- This package does not grant maintenance, scheduling, or operational clearance.

Install
1. Extract this ZIP into C:\Users\verma\aero-guard-nexus (merge folders if prompted).
2. Ensure add_step68_rul_output_cap.py is in the project root.
3. In CMD:
   cd /d "%USERPROFILE%\aero-guard-nexus"
   .venv\Scripts\activate
   python add_step68_rul_output_cap.py
   python -m py_compile backend\ai_engine.py backend\main.py
   python verify_step68_rul_output_cap.py

Runtime verification
- Keep/restart Uvicorn as needed (with --reload, saving the module normally reloads it).
- Run:
  curl http://127.0.0.1:8000/api/v1/ai/rul-predictions
- Check predicted_rul_cycles, baseline_rul_cycles, member_prediction_p10_cycles, and member_prediction_p90_cycles are all within [0, 125].
- Confirm top-level uncertainty_method still says ensemble spread is not a calibrated confidence interval.

Rollback
- Stop Uvicorn, then restore the timestamped backend\ai_engine.py.backup-step68-* file over backend\ai_engine.py.
