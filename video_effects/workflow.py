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
    from video_effects.schemas.workflow import VideoEffectsInput, VideoEffectsOutput

MAX_RETRIES = 5


@workflow.defn(name="VideoEffectsWorkflow")
class VideoEffectsWorkflow:

    def __init__(self):
        self._approval_decision: bool | None = None  # None = pending, True = approved, False = rejected
        self._rejection_feedback: str = ""
        self._timeline_data: dict | None = None

    @workflow.signal
    async def approve_timeline(self, args: list) -> None:
        """Signal from CLI to approve or reject the effect timeline.

        Args is [approved: bool, feedback: str].
        """
        self._approval_decision = args[0]
        self._rejection_feedback = args[1] if len(args) > 1 else ""

    @workflow.query
    def get_timeline(self) -> dict | None:
        """Query to get the current timeline for CLI display."""
        return self._timeline_data

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

        # ── G2: Transcribe audio ──
        transcript_result = await workflow.execute_activity(
            "vfx_transcribe_audio",
            {"audio_path": audio_path},
            start_to_close_timeout=activity_timeout,
        )

        # ── G3-G5 loop: parse → validate → approve (retries on rejection) ──
        rejection_feedback = ""
        for attempt in range(MAX_RETRIES):
            # G3: LLM parse effect cues
            parse_input = {
                "transcript": transcript_result["transcript"],
                "segments": transcript_result["segments"],
                "duration": video_info["duration"],
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
        apply_result = await workflow.execute_activity(
            "vfx_render_video",
            {
                "video_path": video_path, "output_dir": temp_dir,
                "effects": effects, "video_info": video_info,
                "render_plan": render_plan, "cache_dir": temp_dir,
            },
            start_to_close_timeout=long_timeout,
            heartbeat_timeout=timedelta(minutes=2),
        )

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

        # ── G8a-G8e + G9: Motion graphics overlay (optional) ──
        if input.enable_motion_graphics:
            mg_applied = await self._run_motion_graphics(
                base_video=base_output,
                output_path=output_path,
                video_info=video_info,
                transcript=transcript_result["transcript"],
                segments=transcript_result["segments"],
                effects=effects,
                style_hint=input.motion_graphics_style,
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

    async def _run_motion_graphics(
        self,
        *,
        base_video: str,
        output_path: str,
        video_info: dict,
        transcript: str,
        segments: list,
        effects: list,
        style_hint: str,
        temp_dir: str,
        activity_timeout: timedelta,
        long_timeout: timedelta,
    ) -> int:
        """Run G8a-G8e + G9: plan, render, and composite motion graphics.

        Returns number of components rendered (0 if skipped).
        """
        mg_dir = f"{temp_dir}/remotion"

        width = video_info.get("width", 1920)
        height = video_info.get("height", 1080)
        fps = int(video_info.get("fps", 30))
        total_frames = video_info.get("total_frames", 0) or int(video_info.get("duration", 10) * fps)

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

        # ── G8b: LLM plan motion graphics ──
        plan_result = await workflow.execute_activity(
            "vfx_plan_motion_graphics",
            {
                "spatial_context": spatial_context,
                "style_hint": style_hint,
                "video_fps": fps,
            },
            start_to_close_timeout=activity_timeout,
        )

        plan = plan_result.get("composition_plan", {})
        if not plan.get("components"):
            workflow.logger.info("LLM produced no motion graphics components, skipping")
            return 0

        issues = plan_result.get("validation_issues", [])
        if issues:
            workflow.logger.info("MG validation issues: %s", issues)

        # ── G8e: Render transparent overlay ──
        overlay_result = await workflow.execute_activity(
            "vfx_render_motion_overlay",
            {
                "composition_plan": plan,
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
        # Activity writes to temp path then renames to output_path
        # (FFmpeg can't overwrite its own input)
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
