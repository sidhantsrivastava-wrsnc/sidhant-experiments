"""Temporal workflow: VideoEffectsWorkflow.

Pipeline (7+2 groups, G6 split into 3 sub-activities):
  G1: Extract video info + audio
  G2: Transcribe audio
  G3: LLM: parse effect cues from transcript  ─┐
  G4: Validate timeline + resolve conflicts     │ loops on rejection
  G5: CLI approval                             ─┘
  G6a: Prepare render plan (probe, intervals)
  G6b: Setup processors (face tracking, cache)
  G6c: Render video (decode → process → encode)
  G7: Final composition + audio mux
  G8e: Render Remotion motion graphics overlay   (if enable_motion_graphics)
  G9:  FFmpeg composite overlay onto base        (if enable_motion_graphics)
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.schemas.styles import StyleConfig, get_style
    from video_effects.schemas.workflow import VideoEffectsInput, VideoEffectsOutput

MAX_RETRIES = 5


@workflow.defn(name="VideoEffectsWorkflow")
class VideoEffectsWorkflow:

    def __init__(self):
        self._approval_decision: bool | None = None  # None = pending, True = approved, False = rejected
        self._rejection_feedback: str = ""
        self._timeline_data: dict | None = None
        # Motion graphics approval state
        self._mg_plan_data: dict | None = None
        self._mg_approval_decision: bool | None = None
        self._mg_rejection_feedback: str = ""

    @workflow.signal
    async def approve_timeline(self, args: list) -> None:
        """Signal from CLI to approve or reject the effect timeline.

        Args is [approved: bool, feedback: str].
        """
        self._approval_decision = args[0]
        self._rejection_feedback = args[1] if len(args) > 1 else ""

    @workflow.signal
    async def approve_mg_plan(self, args: list) -> None:
        """Signal from CLI to approve or reject the motion graphics plan.

        Args is [approved: bool, feedback: str].
        """
        self._mg_approval_decision = args[0]
        self._mg_rejection_feedback = args[1] if len(args) > 1 else ""

    @workflow.query
    def get_timeline(self) -> dict | None:
        """Query to get the current timeline for CLI display."""
        return self._timeline_data

    @workflow.query
    def get_mg_plan(self) -> dict | None:
        """Query to get the current motion graphics plan for CLI display."""
        return self._mg_plan_data

    @workflow.run
    async def run(self, input: VideoEffectsInput) -> VideoEffectsOutput:
        video_path = input.input_video
        output_path = input.output_video
        temp_dir = f"/tmp/video_effects/{workflow.info().workflow_id}"

        activity_timeout = timedelta(minutes=10)
        long_timeout = timedelta(minutes=30)

        # ── G1: Extract video info + audio (parallel) ──
        video_info_task = workflow.execute_activity(
            "vfx_get_video_info",
            video_path,
            start_to_close_timeout=activity_timeout,
        )
        extract_audio_task = workflow.execute_activity(
            "vfx_extract_audio",
            {"video_path": video_path, "output_dir": temp_dir},
            start_to_close_timeout=activity_timeout,
        )

        video_info, audio_result = await asyncio.gather(
            video_info_task, extract_audio_task
        )
        audio_path = audio_result["audio_path"]
        original_audio_path = audio_result["original_audio_path"]

        # ── G2: Transcribe audio (+ optional jump cut detection in parallel) ──
        transcribe_task = workflow.execute_activity(
            "vfx_transcribe_audio",
            {"audio_path": audio_path},
            start_to_close_timeout=activity_timeout,
        )

        jump_cut_result = None
        if input.smooth_jump_cuts:
            jump_cut_task = workflow.execute_activity(
                "vfx_detect_jump_cuts",
                {"video_path": video_path, "video_info": video_info},
                start_to_close_timeout=activity_timeout,
                heartbeat_timeout=timedelta(minutes=5),
            )
            transcript_result, jump_cut_result = await asyncio.gather(
                transcribe_task, jump_cut_task
            )
        else:
            transcript_result = await transcribe_task

        # ── Creative Designer: auto-detect or apply style ──
        if input.style:
            style_config = get_style(input.style).config.model_dump()
            style_preset_name = input.style
            workflow.logger.info("Using explicit style: %s", input.style)
        else:
            creative_result = await workflow.execute_child_workflow(
                "CreativeDesignerWorkflow",
                {
                    "transcript": transcript_result["transcript"],
                    "video_duration": video_info["duration"],
                    "video_fps": video_info.get("fps", 30),
                },
                id=f"{workflow.info().workflow_id}/creative-designer",
            )
            style_config = creative_result["config"]
            style_preset_name = creative_result["preset_name"]
            workflow.logger.info("Creative designer picked style: %s", style_preset_name)

        # ── G3-G5 loop: parse → validate → approve (retries on rejection) ──
        rejection_feedback = ""
        for attempt in range(MAX_RETRIES):
            # G3: LLM parse effect cues
            parse_input = {
                "transcript": transcript_result["transcript"],
                "segments": transcript_result["segments"],
                "duration": video_info["duration"],
                "style_config": style_config,
                "style_preset_name": style_preset_name,
                "dev_mode": input.dev_mode,
            }
            if rejection_feedback:
                parse_input["feedback"] = rejection_feedback

            parse_result = await workflow.execute_activity(
                "vfx_parse_effect_cues",
                parse_input,
                start_to_close_timeout=activity_timeout,
            )

            # G4: Validate timeline
            timeline = await workflow.execute_activity(
                "vfx_validate_timeline",
                {
                    "effects": parse_result["effects"],
                    "duration": video_info["duration"],
                },
                start_to_close_timeout=activity_timeout,
            )
            self._timeline_data = timeline

            # G5: CLI approval
            if input.auto_approve:
                break

            # Reset decision and wait for signal
            self._approval_decision = None
            self._rejection_feedback = ""

            try:
                await workflow.wait_condition(
                    lambda: self._approval_decision is not None,
                    timeout=timedelta(minutes=10),
                )
            except asyncio.TimeoutError:
                return VideoEffectsOutput(
                    output_video="",
                    error="Timeline approval timed out after 10 minutes",
                )

            if self._approval_decision:
                break  # Approved

            # Rejected — loop back to G3 with feedback
            rejection_feedback = self._rejection_feedback or "User rejected the timeline. Try different effect choices."
            workflow.logger.info(
                f"Timeline rejected (attempt {attempt + 1}/{MAX_RETRIES}): {rejection_feedback}"
            )
        else:
            return VideoEffectsOutput(
                output_video="",
                error=f"Timeline rejected {MAX_RETRIES} times, giving up",
            )

        effects = timeline.get("effects", [])

        # Inject synthetic zoom cues for jump cut smoothing
        if input.smooth_jump_cuts and jump_cut_result:
            jump_cuts = jump_cut_result.get("jump_cuts", [])
            for jc in jump_cuts:
                effects.append({
                    "effect_type": "zoom",
                    "start_time": jc["time"] - 0.15,
                    "end_time": jc["time"] + 0.35,
                    "verbal_cue": "jump cut smoothing",
                    "confidence": jc["confidence"],
                    "zoom_params": {
                        "tracking": "face",
                        "zoom_level": 1.15,
                        "easing": "smooth",
                        "action": "bounce",
                        "motion_blur": 0.3,
                    },
                })
            if jump_cuts:
                workflow.logger.info(
                    "Injected %d synthetic zoom cues for jump cut smoothing", len(jump_cuts)
                )

        # Inject color grading from style if applicable
        style_preset = get_style(style_preset_name)
        grading_preset = style_preset.color_grading_preset
        grading_intensity = style_preset.color_grading_intensity

        if grading_preset and grading_intensity > 0:
            duration = video_info.get("duration", 0)
            effects.append({
                "effect_type": "color_change",
                "start_time": 0,
                "end_time": duration,
                "confidence": 1.0,
                "verbal_cue": f"Style: {grading_preset} color grading",
                "color_params": {
                    "preset": grading_preset,
                    "intensity": grading_intensity,
                },
            })
            workflow.logger.info(
                "Injected %s color grading at %.0f%% intensity from style",
                grading_preset, grading_intensity * 100,
            )

        if not effects:
            return VideoEffectsOutput(
                output_video=video_path,
                effects_applied=0,
                transcript_length=len(transcript_result["transcript"]),
            )

        # ── G6a: Prepare render plan ──
        render_plan = await workflow.execute_activity(
            "vfx_prepare_render",
            {"video_path": video_path, "effects": effects, "video_info": video_info},
            start_to_close_timeout=activity_timeout,
        )

        if not render_plan.get("has_effects"):
            return VideoEffectsOutput(
                output_video=video_path,
                effects_applied=len(effects),
                transcript_length=len(transcript_result["transcript"]),
                phases_executed=0,
            )

        # ── G8a-G8b: MG planning + approval (parallel with G6b-G7) ──
        mg_plan_task = None
        if input.enable_motion_graphics:
            mg_plan_task = self._plan_motion_graphics(
                video_info=video_info,
                transcript=transcript_result["transcript"],
                segments=transcript_result["segments"],
                effects=effects,
                style_hint=input.style,
                style_config=style_config,
                style_preset_name=style_preset_name,
                temp_dir=temp_dir,
                auto_approve=input.auto_approve,
                activity_timeout=activity_timeout,
            )

        # ── G6b: Setup processors (face tracking, etc.) ──
        setup_result = await workflow.execute_activity(
            "vfx_setup_processors",
            {
                "video_path": video_path, "effects": effects,
                "video_info": video_info, "cache_dir": temp_dir,
            },
            start_to_close_timeout=activity_timeout,
            heartbeat_timeout=timedelta(minutes=5),
        )

        # ── G6c: Render video frames ──
        render_task = workflow.execute_activity(
            "vfx_render_video",
            {
                "video_path": video_path, "output_dir": temp_dir,
                "effects": effects, "video_info": video_info,
                "render_plan": render_plan, "cache_dir": temp_dir,
            },
            start_to_close_timeout=long_timeout,
            heartbeat_timeout=timedelta(minutes=2),
        )

        # Wait for render + MG plan in parallel
        if mg_plan_task is not None:
            apply_result, mg_plan = await asyncio.gather(render_task, mg_plan_task)
        else:
            apply_result = await render_task
            mg_plan = None

        # ── G7: Final composition + audio mux ──
        compose_result = await workflow.execute_activity(
            "vfx_compose_final",
            {
                "processed_video": apply_result["processed_video"],
                "audio_path": original_audio_path,
                "output_path": output_path,
                "has_audio": video_info.get("has_audio", True),
            },
            start_to_close_timeout=activity_timeout,
        )

        base_output = compose_result["output_video"]
        mg_applied = 0

        # ── G8e + G9: Render overlay + composite (needs base video + approved plan) ──
        if mg_plan is not None and mg_plan.get("components"):
            mg_applied = await self._render_and_composite_mg(
                mg_plan=mg_plan,
                base_video=base_output,
                output_path=output_path,
                video_info=video_info,
                temp_dir=temp_dir,
                activity_timeout=activity_timeout,
                long_timeout=long_timeout,
            )

        return VideoEffectsOutput(
            output_video=base_output,
            effects_applied=len(effects),
            transcript_length=len(transcript_result["transcript"]),
            phases_executed=apply_result["phases_executed"],
            motion_graphics_applied=mg_applied,
        )

    async def _plan_motion_graphics(
        self,
        *,
        video_info: dict,
        transcript: str,
        segments: list,
        effects: list,
        style_hint: str,
        style_config: dict | None = None,
        style_preset_name: str = "",
        temp_dir: str,
        auto_approve: bool,
        activity_timeout: timedelta,
    ) -> dict | None:
        """Run G8a-G8b: build context, LLM plan, approval loop.

        Returns approved composition plan dict, or None if skipped/rejected.
        Runs in parallel with G6b-G7 (no dependency on rendered video).
        """
        fps = int(video_info.get("fps", 30))

        # ── G8a: Build Remotion context ──
        spatial_context = await workflow.execute_activity(
            "vfx_build_remotion_context",
            {
                "video_info": video_info,
                "transcript": transcript,
                "segments": segments,
                "effects": effects,
                "cache_dir": temp_dir,
            },
            start_to_close_timeout=activity_timeout,
        )

        # ── G8b: LLM plan motion graphics (with approval loop) ──
        mg_feedback = ""
        plan = {}
        for attempt in range(MAX_RETRIES):
            plan_input = {
                "spatial_context": spatial_context,
                "style_hint": style_hint,
                "style_config": style_config,
                "style_preset_name": style_preset_name,
                "video_fps": fps,
            }
            if mg_feedback:
                plan_input["feedback"] = mg_feedback

            plan_result = await workflow.execute_activity(
                "vfx_plan_motion_graphics",
                plan_input,
                start_to_close_timeout=activity_timeout,
            )

            plan = plan_result.get("composition_plan", {})
            issues = plan_result.get("validation_issues", [])

            # Expose plan for CLI query
            self._mg_plan_data = {
                "components": plan.get("components", []),
                "color_palette": plan.get("colorPalette", []),
                "reasoning": plan_result.get("raw_plan", {}).get("reasoning", ""),
                "validation_issues": issues,
            }

            if not plan.get("components"):
                workflow.logger.info("LLM produced no motion graphics components, skipping")
                return None

            if issues:
                workflow.logger.info("MG validation issues: %s", issues)

            if auto_approve:
                break

            # Wait for CLI approval
            self._mg_approval_decision = None
            self._mg_rejection_feedback = ""

            try:
                await workflow.wait_condition(
                    lambda: self._mg_approval_decision is not None,
                    timeout=timedelta(minutes=10),
                )
            except asyncio.TimeoutError:
                workflow.logger.warning("MG plan approval timed out, proceeding with current plan")
                break

            if self._mg_approval_decision:
                break  # Approved

            # Rejected — loop with feedback
            mg_feedback = self._mg_rejection_feedback or "User rejected the motion graphics plan. Try different choices."
            workflow.logger.info(
                "MG plan rejected (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, mg_feedback
            )
            self._mg_plan_data = None  # Clear so CLI can detect new plan
        else:
            workflow.logger.warning("MG plan rejected %d times, skipping motion graphics", MAX_RETRIES)
            return None

        return plan if plan.get("components") else None

    async def _render_and_composite_mg(
        self,
        *,
        mg_plan: dict,
        base_video: str,
        output_path: str,
        video_info: dict,
        temp_dir: str,
        activity_timeout: timedelta,
        long_timeout: timedelta,
    ) -> int:
        """Run G8e + G9: render overlay and composite onto base video.

        Called after both the base video render and MG plan approval are done.
        """
        mg_dir = f"{temp_dir}/remotion"

        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)
        fps = int(video_info.get("fps", 30))
        total_frames = video_info.get("total_frames", 0) or int(video_info.get("duration", 10) * fps)

        # ── G8e: Render transparent overlay ──
        overlay_result = await workflow.execute_activity(
            "vfx_render_motion_overlay",
            {
                "composition_plan": mg_plan,
                "output_dir": f"{mg_dir}/output",
                "video_width": width,
                "video_height": height,
                "video_fps": fps,
                "total_frames": total_frames,
            },
            start_to_close_timeout=long_timeout,
            heartbeat_timeout=timedelta(minutes=5),
        )

        overlay_path = overlay_result.get("overlay_path", "")
        if not overlay_path:
            return 0

        # ── G9: Composite overlay onto base video ──
        await workflow.execute_activity(
            "vfx_composite_motion_graphics",
            {
                "base_video": base_video,
                "overlay_video": overlay_path,
                "output_path": output_path,
                "temp_dir": temp_dir,
            },
            start_to_close_timeout=activity_timeout,
        )

        return overlay_result.get("components_rendered", 0)
