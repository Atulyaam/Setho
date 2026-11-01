"""Model Router and Policy Dispatcher for Stetho (Sprint 7).

Enforces explicit routing across deployment profiles (edge real-time, cloud high-accuracy,
zero-shot general, few-shot calibrated) and strictly enforces asset-category physical boundaries
to prevent dangerous cross-category transfer errors.
"""

import logging
from typing import Optional, Set, Tuple, Union

from stetho_api.db.asset_repo import Asset
from stetho_api.exceptions import ValidationException
from stetho_api.inference.model_registry import ModelType

logger = logging.getLogger("stetho.model_router")

# Category support matrix
CATEGORY_COMPATIBILITY = {
    ModelType.COMPACT_STUDENT_INT8: {"fan", "pump", "valve", "universal"},
    ModelType.COMPACT_STUDENT: {"fan", "pump", "valve", "universal"},
    ModelType.MULTITASK_RESNET_AE: {"fan", "pump", "valve", "universal"},
    ModelType.RESNET_AE: {"fan"},
    ModelType.CASCADE_ENSEMBLE: {"fan"},
    ModelType.STETHO_MSAE: {"fan"},
    ModelType.DENSE_AUTOENCODER: {"fan"},
}


class ModelRouter:
    """Enforces deployment profile dispatching and strict mechanical category bounds."""

    @classmethod
    def resolve_model(
        cls,
        asset: Asset,
        profile_override: Optional[str] = None,
        allow_cross_category: bool = False,
    ) -> Tuple[ModelType, str]:
        """Resolve the optimal model type and version for an asset inspection.
        
        Returns:
            Tuple[ModelType, version_str]
        """
        category = asset.category.lower().strip()
        assigned = asset.assigned_model.lower().strip()
        version = asset.assigned_model_version or "1.0.0"

        # 1. Profile override mapping
        if profile_override:
            p_lower = profile_override.lower().strip()
            if p_lower in ["edge", "edge_real_time", "student"]:
                target_model = ModelType.COMPACT_STUDENT_INT8
            elif p_lower in ["cloud", "cloud_high_accuracy", "resnet"]:
                target_model = ModelType.RESNET_AE
            elif p_lower in ["cascade", "hybrid_cascade", "ensemble"]:
                target_model = ModelType.CASCADE_ENSEMBLE
            elif p_lower in ["zero_shot", "multitask"]:
                target_model = ModelType.MULTITASK_RESNET_AE
            elif p_lower in ["stetho_msae", "legacy_production"]:
                target_model = ModelType.STETHO_MSAE
            elif p_lower in ["dense_autoencoder", "legacy_baseline"]:
                target_model = ModelType.DENSE_AUTOENCODER
            else:
                try:
                    target_model = ModelType(p_lower)
                except ValueError:
                    raise ValidationException(f"Unknown deployment profile override: '{profile_override}'")
        else:
            # 2. Derive from asset configuration
            try:
                target_model = ModelType(assigned)
            except ValueError:
                target_model = ModelType.CASCADE_ENSEMBLE

        # 3. Strict Asset-Category Restriction Check
        allowed_categories: Set[str] = CATEGORY_COMPATIBILITY.get(target_model, {"fan"})
        if category not in allowed_categories and "universal" not in allowed_categories:
            if not allow_cross_category:
                raise ValidationException(
                    f"Category routing error: Model '{target_model.value}' is restricted to "
                    f"{sorted(list(allowed_categories))} equipment and cannot be evaluated on '{category}' "
                    f"due to acoustic regime mismatch (Sprint 6D bound). Set allow_cross_category=true to override."
                )
            else:
                logger.warning(
                    f"Cross-category execution forced: Model '{target_model.value}' applied to '{category}' "
                    f"for asset '{asset.asset_id}'."
                )

        # 4. Handle uncalibrated assets when no explicit override requested
        if asset.calibration_status == "uncalibrated" and not profile_override:
            if target_model in [ModelType.RESNET_AE, ModelType.CASCADE_ENSEMBLE]:
                logger.info(
                    f"Asset '{asset.asset_id}' is uncalibrated. Routing to zero-shot MultiTaskResNetAE "
                    f"to prevent false alarm skew."
                )
                target_model = ModelType.MULTITASK_RESNET_AE

        logger.debug(
            f"Routed asset '{asset.asset_id}' ({category}) to model '{target_model.value}' (v{version})"
        )
        return target_model, version
