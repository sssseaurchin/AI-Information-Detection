import hashlib
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd


def get_repo_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def get_default_governance_config_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "config", "governance.json")


def get_default_access_log_path() -> str:
    return os.path.join(get_repo_root(), "governance", "access_log.jsonl")


def get_default_baseline_path() -> str:
    return os.path.join(get_repo_root(), "governance", "baseline.json")


def compute_file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_governance_config(config_path: str | None = None) -> dict[str, Any]:
    path = config_path or get_default_governance_config_path()
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = {
        "selection_metric",
        "tuning_split_preference",
        "held_out_splits",
        "max_heldout_evals_per_day",
        "require_manifest",
        "require_label_mapping_hash",
        "require_split_policy_frozen",
        "allow_subset_tuning",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Governance config missing keys: {', '.join(sorted(missing))}")
    return config


def select_tuning_split_from_policy(manifest: pd.DataFrame, tuning_split_preference: list[str], eval_split: str) -> str:
    available = set(manifest["split"].astype(str).tolist())
    for split_name in tuning_split_preference:
        if split_name in available:
            return split_name
    if eval_split in available and eval_split not in set(tuning_split_preference):
        return eval_split
    raise ValueError("No governance-approved tuning split is available in the manifest.")


def requires_tuning_split(threshold_policy: str, calibrate_mode: str) -> bool:
    """Return whether evaluation needs a separate tuning/calibration split."""
    return not (str(threshold_policy).lower() == "fixed" and str(calibrate_mode).lower() == "none")


def _read_access_log(log_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(log_path):
        return []

    records: list[dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _count_allowed_heldout_today(records: list[dict[str, Any]], held_out_splits: set[str], day_key: str) -> int:
    return sum(
        1
        for record in records
        if record.get("eval_split") in held_out_splits
        and str(record.get("utc_timestamp", "")).startswith(day_key)
        and bool(record.get("allowed"))
    )


def _load_baseline(baseline_path: str) -> dict[str, Any] | None:
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_baseline(baseline_path: str, baseline: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return baseline_path


def _build_baseline_hashes(
    governance_config_hash: str,
    manifest_hash: str,
    model_hash: str,
    label_mapping_hash: str | None,
) -> dict[str, Any]:
    return {
        "governance_config_hash": governance_config_hash,
        "manifest_hash": manifest_hash,
        "model_hash": model_hash,
        "label_mapping_hash": label_mapping_hash,
    }


def _detect_hash_drift(current_hashes: dict[str, Any], baseline: dict[str, Any] | None) -> list[str]:
    if not baseline:
        return []

    mismatches: list[str] = []
    baseline_hashes = baseline.get("hashes", {})
    file_labels = {
        "governance_config_hash": "governance config",
        "manifest_hash": "manifest",
        "model_hash": "model",
        "label_mapping_hash": "label mapping",
    }
    for key, label in file_labels.items():
        baseline_value = baseline_hashes.get(key)
        current_value = current_hashes.get(key)
        if baseline_value and current_value and baseline_value != current_value:
            mismatches.append(
                f"{label} hash mismatch (baseline={baseline_value}, current={current_value})"
            )
    return mismatches


def build_access_record(
    user_id: str,
    eval_split: str,
    tuning_split: str,
    model_path: str,
    manifest_path: str,
    governance_config_path: str,
    subset: dict[str, Any],
    threshold_policy: str,
    calibrate_mode: str,
    forbid_test_tuning: bool,
    override_used: bool,
    governance_config_hash: str,
    manifest_hash: str,
    model_hash: str,
    label_mapping_hash: str | None,
    baseline_hashes: dict[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.utcnow()
    timestamp = now.isoformat(timespec="seconds") + "Z"
    default_run_id = f"blocked_{now.strftime('%Y%m%dT%H%M%SZ')}"
    return {
        "timestamp_utc": timestamp,
        "utc_timestamp": timestamp,
        "user_id": user_id,
        "run_id": default_run_id,
        "eval_split": eval_split,
        "tuning_split": tuning_split,
        "model_path": os.path.abspath(model_path),
        "manifest_path": os.path.abspath(manifest_path) if manifest_path else "",
        "governance_config_path": os.path.abspath(governance_config_path),
        "governance_config_hash": governance_config_hash,
        "manifest_hash": manifest_hash,
        "model_hash": model_hash,
        "label_mapping_hash": label_mapping_hash,
        "baseline_hashes": baseline_hashes,
        "subset": {
            "dataset_id": subset.get("dataset_id"),
            "domain": subset.get("domain"),
        },
        "threshold_policy": threshold_policy,
        "calibrate_mode": calibrate_mode,
        "forbid_test_tuning": bool(forbid_test_tuning),
        "allowed": False,
        "valid_for_thesis": False,
        "reason": "pending",
        "override_used": bool(override_used),
    }


def evaluate_governance_request(
    governance_config: dict[str, Any],
    governance_config_path: str,
    manifest_path: str,
    label_mapping_hash: str | None,
    eval_split: str,
    tuning_split: str,
    model_path: str,
    user_id: str,
    subset: dict[str, Any],
    threshold_policy: str,
    calibrate_mode: str,
    forbid_test_tuning: bool,
    override_governance: bool,
    access_log_path: str | None = None,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    held_out_splits = set(governance_config["held_out_splits"])
    log_path = access_log_path or get_default_access_log_path()
    resolved_baseline_path = baseline_path or get_default_baseline_path()
    day_key = datetime.utcnow().strftime("%Y-%m-%d")

    if not os.path.exists(governance_config_path):
        raise FileNotFoundError(f"Governance config not found: {governance_config_path}")
    if governance_config.get("require_manifest", True) and not manifest_path:
        governance_config_hash = compute_file_sha256(governance_config_path)
        manifest_hash = ""
        model_hash = compute_file_sha256(model_path)
        baseline = _load_baseline(resolved_baseline_path)
        baseline_hashes = baseline.get("hashes") if baseline else None
        record = build_access_record(
            user_id=user_id,
            eval_split=eval_split,
            tuning_split=tuning_split,
            model_path=model_path,
            manifest_path=manifest_path,
            governance_config_path=governance_config_path,
            subset=subset,
            threshold_policy=threshold_policy,
            calibrate_mode=calibrate_mode,
            forbid_test_tuning=forbid_test_tuning,
            override_used=override_governance,
            governance_config_hash=governance_config_hash,
            manifest_hash=manifest_hash,
            model_hash=model_hash,
            label_mapping_hash=label_mapping_hash,
            baseline_hashes=baseline_hashes,
        )
        record["reason"] = "manifest_required"
        return {
            "allowed": False,
            "reason": "manifest_required",
            "override_used": bool(override_governance),
            "valid_for_thesis": False,
            "access_log_path": os.path.abspath(log_path),
            "baseline_path": os.path.abspath(resolved_baseline_path),
            "selection_metric": governance_config["selection_metric"],
            "record": record,
            "hashes": {
                "governance_config_hash": governance_config_hash,
                "manifest_hash": manifest_hash,
                "model_hash": model_hash,
                "label_mapping_hash": label_mapping_hash,
            },
            "baseline_hashes": baseline_hashes,
            "baseline_pending_write": False,
        }

    if governance_config.get("require_manifest", True) and not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    governance_config_hash = compute_file_sha256(governance_config_path)
    manifest_hash = compute_file_sha256(manifest_path)
    model_hash = compute_file_sha256(model_path)
    baseline = _load_baseline(resolved_baseline_path)
    baseline_hashes = baseline.get("hashes") if baseline else None
    current_hashes = _build_baseline_hashes(
        governance_config_hash=governance_config_hash,
        manifest_hash=manifest_hash,
        model_hash=model_hash,
        label_mapping_hash=label_mapping_hash,
    )

    record = build_access_record(
        user_id=user_id,
        eval_split=eval_split,
        tuning_split=tuning_split,
        model_path=model_path,
        manifest_path=manifest_path,
        governance_config_path=governance_config_path,
        subset=subset,
        threshold_policy=threshold_policy,
        calibrate_mode=calibrate_mode,
        forbid_test_tuning=forbid_test_tuning,
        override_used=override_governance,
        governance_config_hash=governance_config_hash,
        manifest_hash=manifest_hash,
        model_hash=model_hash,
        label_mapping_hash=label_mapping_hash,
        baseline_hashes=baseline_hashes,
    )

    allowed = True
    valid_for_thesis = True
    reason = "allowed"

    if governance_config.get("require_label_mapping_hash", True) and not label_mapping_hash:
        allowed = False
        valid_for_thesis = False
        reason = "label_mapping_hash_required"
    elif governance_config.get("require_split_policy_frozen", True) and not manifest_path:
        allowed = False
        valid_for_thesis = False
        reason = "split_policy_not_frozen"
    elif forbid_test_tuning and eval_split in held_out_splits and tuning_split in held_out_splits:
        if override_governance:
            allowed = True
            valid_for_thesis = False
            reason = "override_used_hash_drift"
        else:
            allowed = False
            valid_for_thesis = False
            reason = "held_out_tuning_forbidden"
    else:
        drift_messages = _detect_hash_drift(current_hashes, baseline)
        if drift_messages:
            if override_governance:
                allowed = True
                valid_for_thesis = False
                reason = "override_used_hash_drift"
            else:
                allowed = False
                valid_for_thesis = False
                reason = "hash_drift_blocked: " + "; ".join(drift_messages)

        records = _read_access_log(log_path)
        held_out_today = _count_allowed_heldout_today(records, held_out_splits, day_key)
        if allowed and eval_split in held_out_splits and held_out_today >= int(governance_config["max_heldout_evals_per_day"]):
            if override_governance:
                allowed = True
                valid_for_thesis = False
                reason = "override_daily_heldout_limit"
            else:
                allowed = False
                valid_for_thesis = False
                reason = "daily_heldout_limit_exceeded"
        elif allowed and override_governance and reason == "allowed":
            valid_for_thesis = False
            reason = "override_used_hash_drift"

    resolved_baseline_hashes = baseline_hashes or (
        current_hashes if allowed and valid_for_thesis and baseline is None and not override_governance else None
    )
    record["allowed"] = bool(allowed)
    record["valid_for_thesis"] = bool(valid_for_thesis)
    record["reason"] = reason
    record["baseline_hashes"] = resolved_baseline_hashes
    return {
        "allowed": bool(allowed),
        "reason": reason,
        "override_used": bool(override_governance),
        "valid_for_thesis": bool(valid_for_thesis),
        "access_log_path": os.path.abspath(log_path),
        "baseline_path": os.path.abspath(resolved_baseline_path),
        "selection_metric": governance_config["selection_metric"],
        "record": record,
        "hashes": current_hashes,
        "baseline_hashes": resolved_baseline_hashes,
        "baseline_pending_write": bool(allowed and valid_for_thesis and baseline is None and not override_governance),
        "governance_config_path": os.path.abspath(governance_config_path),
    }


def finalize_access_record(
    governance_result: dict[str, Any],
    user_id: str,
    run_id: str,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(governance_result["record"])
    record["run_id"] = run_id
    if artifacts:
        record["artifacts"] = artifacts

    if governance_result.get("baseline_pending_write"):
        baseline_payload = {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "created_by_user_id": user_id,
            "run_id": run_id,
            "hashes": governance_result["hashes"],
        }
        _write_baseline(governance_result["baseline_path"], baseline_payload)
        record["baseline_created"] = True
        record["baseline_path"] = governance_result["baseline_path"]

    return record


def append_access_log(record: dict[str, Any], access_log_path: str | None = None) -> str:
    log_path = access_log_path or get_default_access_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return log_path
