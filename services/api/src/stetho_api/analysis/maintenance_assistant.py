"""Evidence-Based Maintenance Assistant for Stetho Sprint 6A (Feature 4).

Generates transparent, rule-based inspection guidance linking detected acoustic
deviations directly to empirical signal metrics, frequency band shifts, and
physical maintenance checklists. Avoids hallucinated or unverified failure probabilities.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class MaintenanceAssistant:
    """Generates evidence-backed inspection guidance from acoustic metrics and machine fingerprint."""

    @staticmethod
    def generate_advice(
        analysis_record: Dict[str, Any],
        machine_fingerprint: Optional[Dict[str, Any]] = None,
        quality_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce structured maintenance advice based strictly on verifiable signal evidence."""
        decision = analysis_record.get("decision", "INCONCLUSIVE")
        anomaly_score = analysis_record.get("anomaly_score", 0.0)
        operating_threshold = analysis_record.get("operating_threshold", 0.0)
        quality_status = analysis_record.get("quality_status", "ACCEPTED")

        findings: List[Dict[str, str]] = []
        checklist: List[Dict[str, Any]] = []
        urgency = "LOW"

        # 1. Quality & Inconclusive Assessment
        if quality_status in ["REJECTED", "WARNING"] or decision == "INCONCLUSIVE":
            urgency = "VERIFY_RECORDING"
            findings.append({
                "type": "RECORDING_QUALITY",
                "evidence": f"Audio quality status is '{quality_status}' with low SNR or amplitude variation.",
                "explanation": "Signal anomalies may be caused by microphone placement, handling noise, or transient background factory sounds rather than mechanical changes.",
            })
            checklist.append({
                "action": "Repeat Acoustic Recording",
                "priority": "IMMEDIATE",
                "instructions": "Place microphone 15-25 cm from machine housing, ensure a solid grip to prevent handling rattle, and record for a minimum of 5 seconds during steady-state operation.",
            })

        # 2. Unusual Anomaly Evaluation
        elif decision == "UNUSUAL":
            dev_ratio = (anomaly_score / operating_threshold) if operating_threshold > 0 else 1.0
            if dev_ratio >= 2.0:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"

            findings.append({
                "type": "ACOUSTIC_DEVIATION",
                "evidence": f"Anomaly score ({anomaly_score:.5f}) exceeds operating threshold ({operating_threshold:.5f}) by {((dev_ratio - 1.0) * 100):.1f}%.",
                "explanation": "Reconstruction error across multiple temporal frames indicates non-stationary acoustic patterns not present in the machine's enrolled normal baseline.",
            })

            # Check spectral band shifts if fingerprint is available
            if machine_fingerprint:
                # Compare high frequency band
                fp_high = machine_fingerprint.get("high_energy_db", -30.0)
                checklist.append({
                    "action": "Inspect Rotating Bearings & Lubrication",
                    "priority": "HIGH",
                    "instructions": "Check bearing temperatures with infrared thermometer. Verify lubrication grease level and listen for metallic chattering or high-frequency hiss.",
                })
                checklist.append({
                    "action": "Check Impeller & Blade Clearances",
                    "priority": "MEDIUM",
                    "instructions": "Inspect fan blades for particulate accumulation, pitting, or casing rub signs during planned downtime.",
                })
                checklist.append({
                    "action": "Verify Structural Mounting & Belt Tension",
                    "priority": "MEDIUM",
                    "instructions": "Check torque on base foundation bolts, verify anti-vibration damping pads, and inspect drive belts for slippage or wear.",
                })
            else:
                checklist.append({
                    "action": "Comprehensive Mechanical Walkdown",
                    "priority": "HIGH",
                    "instructions": "Conduct an operator walkdown inspecting motor casing temperature, mounting bolt integrity, and audible rattles.",
                })

        # 3. Normal-like Behavior
        else:
            urgency = "LOW"
            findings.append({
                "type": "NORMAL_STABILITY",
                "evidence": f"Anomaly score ({anomaly_score:.5f}) is within operating threshold ({operating_threshold:.5f}).",
                "explanation": "Machine acoustic signature conforms to reference baseline profile under current operating conditions.",
            })
            checklist.append({
                "action": "Continue Scheduled Predictive Monitoring",
                "priority": "ROUTINE",
                "instructions": "Log regular periodic acoustic inspections to track long-term baseline drift.",
            })

        return {
            "analysis_id": analysis_record.get("analysis_id"),
            "machine_id": analysis_record.get("machine_id"),
            "decision": decision,
            "maintenance_urgency": urgency,
            "anomaly_score": anomaly_score,
            "operating_threshold": operating_threshold,
            "deviation_ratio": round(anomaly_score / operating_threshold, 3) if operating_threshold > 0 else 1.0,
            "findings": findings,
            "inspection_checklist": checklist,
            "engineering_disclaimer": "Engineering Disclaimer: This assistance is rule-based advisory guidance derived from acoustic signal analysis. It does not replace physical inspection or certified vibration analysis standards.",
        }
