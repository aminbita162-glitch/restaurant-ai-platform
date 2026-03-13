# Golden Test Pack – Phase 1

This document defines the baseline tests that must pass before any production changes.

If any of these tests fail, the system must be considered unstable.

---

# Test 1 — Full Pipeline Execution

Purpose:
Verify that the entire AI pipeline runs successfully.

URL:

/api/v1/pipeline/run?execute=1&confirm=yes&restaurant_id=restaurant_001&location_id=location_001&steps=1_data_ingestion,2_data_warehouse,3_feature_engineering,4_feature_store_sync,model_registry,5_ml_prediction,gpt_insight,6_optimization,8_dashboard_update

Expected Result:

- status = ok
- summary.error_count = 0
- summary.ok_count = 9

---

# Test 2 — Prediction Engine

Purpose:
Verify ML prediction works independently.

Steps:

5_ml_prediction

Expected Result:

- forecast array returned
- horizon = 7

---

# Test 3 — GPT Insight Generation

Purpose:
Verify AI business insights generation.

Steps:

gpt_insight

Expected Result:

- insight_json exists
- risk_level exists
- actions list exists

---

# Test 4 — Dashboard Update

Purpose:
Verify dashboard payload generation.

Steps:

8_dashboard_update

Expected Result:

- dashboard_update_status = ok
- forecast returned
- staffing_plan returned
- inventory_plan returned

---

# Test 5 — Pipeline Health

Purpose:
Ensure pipeline orchestration works.

Expected:

- run_id generated
- steps executed sequentially
- duration_ms returned

---

# Baseline Status

If all tests pass, the system is considered **production stable (Phase 1)**.