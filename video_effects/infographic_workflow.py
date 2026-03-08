"""Temporal child workflow: InfographicGeneratorWorkflow.

Runs in parallel with MG planning + video rendering.
For each infographic spec: generate TSX code, validate, retry up to N times,
fall back to existing templates on failure.
"""

import math
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.config import settings
    from video_effects.schemas.infographic import FALLBACK_MAP, InfographicType


@workflow.defn(name="InfographicGeneratorWorkflow")
class InfographicGeneratorWorkflow:

    @workflow.run
    async def run(self, input: dict) -> dict:
        """Generate custom infographic components from transcript analysis.

        Input: {
            "spatial_context": dict,
            "transcript": str,
            "segments": list[dict],
            "style_config": dict | None,
            "video_fps": int,
            "video_info": dict,
            "workflow_prefix": str,   # short ID for uniqueness
        }
        Output: {
            "generated_components": list[dict],  # Remotion ComponentSpec dicts
            "fallback_components": list[dict],    # template-based fallbacks
            "registry_path": str,
        }
        """
        spatial_context = input["spatial_context"]
        transcript = input.get("transcript", "")
        segments = input.get("segments", [])
        style_config = input.get("style_config")
        fps = input.get("video_fps", 30)
        video_info = input.get("video_info", {})
        prefix = input.get("workflow_prefix", "ig")

        activity_timeout = timedelta(minutes=5)
        max_retries = getattr(settings, "INFOGRAPHIC_MAX_RETRIES", 3)

        # Clean up generated/ directory from previous runs
        await workflow.execute_activity(
            "vfx_cleanup_generated",
            {},
            start_to_close_timeout=timedelta(seconds=30),
        )

        # ── A1: Plan infographics ──
        plan_result = await workflow.execute_activity(
            "vfx_plan_infographics",
            {
                "spatial_context": spatial_context,
                "transcript": transcript,
                "segments": segments,
                "style_config": style_config,
                "video_fps": fps,
            },
            start_to_close_timeout=activity_timeout,
        )

        specs = plan_result.get("infographics", [])
        if not specs:
            workflow.logger.info("No infographics planned, skipping code generation")
            return {"generated_components": [], "fallback_components": [], "registry_path": ""}

        workflow.logger.info("Planning complete: %d infographic(s) to generate", len(specs))

        # ── A2+A3: Generate + validate each infographic ──
        successful = []  # [{component_id, export_name, spec, props}]
        fallbacks = []   # [{template, spec}]

        for i, spec in enumerate(specs):
            # Prefix component ID for uniqueness
            spec["id"] = f"{prefix}_{spec.get('id', f'infographic_{i}')}"

            generated = await self._generate_with_retries(
                spec=spec,
                style_config=style_config,
                video_info=video_info,
                max_retries=max_retries,
                activity_timeout=activity_timeout,
            )

            if generated is not None:
                successful.append(generated)
            else:
                # Fall back to existing template
                fallback = self._build_fallback(spec, fps)
                if fallback:
                    fallbacks.append(fallback)
                    workflow.logger.info(
                        "Fell back to template '%s' for %s",
                        fallback["template"], spec["id"],
                    )

        # ── A4: Build registry for successful components ──
        registry_result = {"components": [], "registry_path": ""}
        if successful:
            registry_result = await workflow.execute_activity(
                "vfx_build_generated_registry",
                {
                    "generated_components": successful,
                    "video_fps": fps,
                    "style_config": style_config,
                },
                start_to_close_timeout=timedelta(seconds=30),
            )

        workflow.logger.info(
            "Infographic generation complete: %d generated, %d fallback",
            len(successful), len(fallbacks),
        )

        return {
            "generated_components": registry_result.get("components", []),
            "fallback_components": fallbacks,
            "registry_path": registry_result.get("registry_path", ""),
        }

    async def _generate_with_retries(
        self,
        spec: dict,
        style_config: dict | None,
        video_info: dict,
        max_retries: int,
        activity_timeout: timedelta,
    ) -> dict | None:
        """Try to generate + validate a component, retrying on failure.

        Returns component dict on success, None on exhausted retries.
        """
        previous_errors: list[str] = []
        previous_code = ""

        for attempt in range(1, max_retries + 1):
            workflow.logger.info(
                "Generating %s (attempt %d/%d)", spec["id"], attempt, max_retries
            )

            # A2: Generate code
            gen_result = await workflow.execute_activity(
                "vfx_generate_infographic_code",
                {
                    "spec": spec,
                    "style_config": style_config,
                    "video_info": video_info,
                    "attempt": attempt,
                    "previous_errors": previous_errors,
                    "previous_code": previous_code,
                },
                start_to_close_timeout=activity_timeout,
            )

            tsx_code = gen_result["tsx_code"]
            export_name = gen_result["export_name"]
            component_id = gen_result["component_id"]

            # A3: Validate
            val_result = await workflow.execute_activity(
                "vfx_validate_infographic",
                {
                    "component_id": component_id,
                    "tsx_code": tsx_code,
                    "export_name": export_name,
                },
                start_to_close_timeout=activity_timeout,
            )

            if val_result["valid"]:
                workflow.logger.info("Validated %s on attempt %d", component_id, attempt)
                return {
                    "component_id": component_id,
                    "export_name": export_name,
                    "spec": spec,
                    "props": gen_result.get("props", {}),
                }

            # Failed — collect errors for retry
            previous_errors = val_result.get("errors", [])
            previous_code = tsx_code
            workflow.logger.warning(
                "Validation failed for %s (attempt %d): %s",
                component_id, attempt, previous_errors[:3],
            )

        workflow.logger.warning(
            "All %d attempts failed for %s, falling back to template",
            max_retries, spec["id"],
        )
        return None

    @staticmethod
    def _build_fallback(spec: dict, fps: int) -> dict | None:
        """Map a failed infographic spec to an existing template component."""
        infographic_type = spec.get("type", "custom")
        try:
            ig_type = InfographicType(infographic_type)
        except ValueError:
            ig_type = InfographicType.CUSTOM

        template_name, default_style = FALLBACK_MAP.get(
            ig_type, ("animated_title", "fade")
        )

        start_frame = round(spec.get("start_time", 0) * fps)
        end_time = spec.get("end_time", spec.get("start_time", 0) + 3)
        end_frame = round(end_time * fps)
        duration_frames = max(1, end_frame - start_frame)

        data = spec.get("data", {})
        props: dict = {}

        if template_name == "data_animation":
            items = data.get("items", data.get("stats", []))
            if items and isinstance(items, list) and len(items) > 0:
                first = items[0]
                props = {
                    "style": default_style,
                    "value": first.get("value", 0),
                    "label": first.get("label", spec.get("title", "")),
                    "suffix": data.get("unit", ""),
                    "items": items[:4],
                }
            else:
                props = {
                    "style": "stat-callout",
                    "value": 0,
                    "label": spec.get("title", ""),
                }
        elif template_name == "listicle":
            items = data.get("steps", data.get("events", data.get("items", [])))
            str_items = []
            for item in items[:6]:
                if isinstance(item, str):
                    str_items.append(item)
                elif isinstance(item, dict):
                    str_items.append(item.get("title", item.get("label", item.get("text", str(item)))))
            props = {
                "items": str_items or [spec.get("title", "")],
                "style": "slide",
                "listStyle": "numbered",
            }
        elif template_name == "animated_title":
            props = {
                "text": spec.get("title", "Infographic"),
                "style": default_style,
            }

        return {
            "template": template_name,
            "startFrame": start_frame,
            "durationInFrames": duration_frames,
            "props": props,
            "bounds": spec.get("bounds", {"x": 0.1, "y": 0.1, "w": 0.35, "h": 0.3}),
            "zIndex": 5,
            "anchor": spec.get("anchor", "static"),
        }
