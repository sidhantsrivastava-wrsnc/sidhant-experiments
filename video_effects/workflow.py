import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from video_effects.schemas.styles import get_style
    from video_effects.schemas.workflow import VideoEffectsInput, VideoEffectsOutput

MAX_RETRIES = 5

SUBTITLE_BOUNDS = {"x": 0.0, "y": 0.78, "w": 1.0, "h": 0.22}


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
        self._rejection_feedback = args[1] if len(args) > 1 else "" # TODO:There's a better way to understand that feedback is always some kind of sentence string.

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
        # TODO: Move this to a smaller function outside
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
        audio_path = audio_result.get("audio_path")
        original_audio_path = audio_result.get("original_audio_path")

        # ── G2: Transcribe audio ──
        transcript_result = await workflow.execute_activity(
            "vfx_transcribe_audio",
            {"audio_path": audio_path},
            start_to_close_timeout=activity_timeout,
        )

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
            parse_input = { # TODO: Come one man why isn't this a pydantic model its 2026 ffs?
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
                break

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

        # --mg and --infographics both route through the code-gen pipeline
        enable_infographics = getattr(input, "enable_infographics", False) or input.enable_motion_graphics # TODO: Honestly I might just get rid of infographics and just use programmer.
        enable_programmer = getattr(input, "enable_programmer", False)
        has_transcript = bool(transcript_result.get("segments"))
        if not effects and not enable_infographics and not enable_programmer and not has_transcript:
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

        if not render_plan.get("has_effects") and not enable_infographics and not enable_programmer and not has_transcript:
            return VideoEffectsOutput(
                output_video=video_path,
                effects_applied=len(effects),
                transcript_length=len(transcript_result["transcript"]),
                phases_executed=0,
            )

        # ── Face detection (before G8a so face cache exists) ──
        await workflow.execute_activity(
            "vfx_detect_faces",
            {
                "video_path": video_path,
                "video_info": video_info,
                "cache_dir": temp_dir,
            },
            start_to_close_timeout=long_timeout,
            heartbeat_timeout=timedelta(minutes=5),
        )

        # ── G8a: Build spatial context (shared by MG planning + infographics + subtitles) ──
        spatial_context = None
        if True:  # Always build — subtitles always need it
            spatial_context = await workflow.execute_activity(
                "vfx_build_remotion_context",
                {
                    "video_info": video_info,
                    "transcript": transcript_result["transcript"],
                    "segments": transcript_result["segments"],
                    "effects": effects,
                    "cache_dir": temp_dir,
                },
                start_to_close_timeout=activity_timeout,
            )

        # ── Component generation (parallel with video render) ──
        # ProgrammerWorkflow takes priority over InfographicGeneratorWorkflow
        code_generation_task = None
        wid = workflow.info().workflow_id
        if enable_programmer and spatial_context:
            code_generation_task = workflow.execute_child_workflow(
                "ProgrammerWorkflow",
                {
                    "spatial_context": spatial_context,
                    "transcript": transcript_result["transcript"],
                    "segments": transcript_result["segments"],
                    "style_config": style_config,
                    "video_fps": int(video_info.get("fps", 30)),
                    "video_info": video_info,
                    "workflow_prefix": wid,
                },
                id=f"{wid}/programmer-gen",
            )
        elif enable_infographics and spatial_context:
            code_generation_task = workflow.execute_child_workflow(
                "InfographicGeneratorWorkflow",
                {
                    "spatial_context": spatial_context,
                    "transcript": transcript_result["transcript"],
                    "segments": transcript_result["segments"],
                    "style_config": style_config,
                    "video_fps": int(video_info.get("fps", 30)),
                    "video_info": video_info,
                    "workflow_prefix": wid,
                },
                id=f"{wid}/infographic-gen",
            )

        # ── G6b: Setup processors (face tracking, etc.) ──
        await workflow.execute_activity( 
            "vfx_setup_processors",
            {
                "video_path": video_path, "effects": effects,
                "video_info": video_info, "cache_dir": temp_dir, # TODO: there's probably a better way to do this. temp dir thing. This doesn't feel right. 
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

        apply_result, code_generation_result = await asyncio.gather(render_task, code_generation_task)

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

        # Pre-compute word segments (needed for subtitle obstacle + subtitle injection)
        subtitle_segments = transcript_result.get("segments", [])
        word_segments = [s for s in subtitle_segments if s.get("type") == "word" and s.get("text", "").strip()]

        # ── Merge infographic components into MG plan ──
        if code_generation_result is not None:
            gen_comps = code_generation_result.get("generated_components", [])
            fb_comps = code_generation_result.get("fallback_components", [])
            extra_components = gen_comps + fb_comps

            if extra_components:
                if mg_plan is None:
                    mg_plan = {
                        "components": [],
                        "colorPalette": [],
                        "includeBaseVideo": False,
                        "faceDataPath": spatial_context.get("face_data_path", "") if spatial_context else "",
                        "zoomStatePath": spatial_context.get("zoom_state_path", "") if spatial_context else "",
                    }
                    if style_config:
                        mg_plan["styleConfig"] = style_config

                mg_plan["components"] = mg_plan.get("components", []) + extra_components
                workflow.logger.info(
                    "Merged %d infographic components (%d generated, %d fallback)",
                    len(extra_components), len(gen_comps), len(fb_comps),
                )

                # Re-validate merged plan (infographic overlap, face overlap, concurrency)
                # Components here are still in time-domain (start_time/end_time seconds)
                time_domain_comps = []
                fps = int(video_info.get("fps", 30))
                for c in mg_plan["components"]:
                    if "start_time" in c:
                        time_domain_comps.append(c)
                    else:
                        # Convert frame-domain back to time-domain for validation
                        time_domain_comps.append({
                            **c,
                            "start_time": c["startFrame"] / fps,
                            "end_time": (c["startFrame"] + c["durationInFrames"]) / fps,
                        })

                revalidation = await workflow.execute_activity(
                    "vfx_validate_merged_plan",
                    {
                        "components": time_domain_comps,
                        "spatial_context": spatial_context,
                        "video_fps": fps,
                        "static_obstacles": [SUBTITLE_BOUNDS] if word_segments else [],
                    },
                    start_to_close_timeout=activity_timeout,
                )

                # Convert validated components back to frame-domain
                revalidated = []
                for c in revalidation["components"]:
                    start_frame = round(c.get("start_time", 0) * fps)
                    end_frame = round(c.get("end_time", 0) * fps)
                    dur_frames = max(1, end_frame - start_frame)
                    revalidated.append({
                        **{k: v for k, v in c.items() if k not in ("start_time", "end_time")},
                        "startFrame": start_frame,
                        "durationInFrames": dur_frames,
                    })
                mg_plan["components"] = revalidated

                if revalidation.get("validation_issues"):
                    workflow.logger.info(
                        "Post-merge validation: %s", revalidation["validation_issues"]
                    )

        # ── Always inject subtitles from transcript ──
        # TODO: This can be a separate function.
        if word_segments:
            fps = int(video_info.get("fps", 30))

            # Build word list in frame-domain for the Subtitles component
            subtitle_words = []
            for seg in word_segments:
                subtitle_words.append({
                    "text": seg["text"],
                    "startFrame": round(seg["start"] * fps),
                    "endFrame": round(seg["end"] * fps),
                })

            # Single subtitle component spanning all speech
            first_frame = subtitle_words[0]["startFrame"]
            last_frame = subtitle_words[-1]["endFrame"]

            # Convert to Sequence-relative frames (useCurrentFrame() returns 0 at Sequence start)
            for w in subtitle_words:
                w["startFrame"] = w["startFrame"] - first_frame
                w["endFrame"] = w["endFrame"] - first_frame
            subtitle_component = {
                "template": "subtitles",
                "startFrame": first_frame,
                "durationInFrames": max(1, last_frame - first_frame),
                "props": {
                    "words": subtitle_words,
                    "fontSize": 44,
                },
                "bounds": {"x": 0.1, "y": SUBTITLE_BOUNDS["y"], "w": 0.8, "h": 0.16},
                "zIndex": 100,  # Subtitles always on top
                "anchor": "static",
            }

            if mg_plan is None:
                mg_plan = {
                    "components": [],
                    "colorPalette": [],
                    "includeBaseVideo": False,
                    "faceDataPath": spatial_context.get("face_data_path", "") if spatial_context else "",
                    "zoomStatePath": spatial_context.get("zoom_state_path", "") if spatial_context else "",
                }
                if style_config:
                    mg_plan["styleConfig"] = style_config

            mg_plan["components"].append(subtitle_component)
            workflow.logger.info(
                "Injected subtitles: %d words, frames %d-%d",
                len(subtitle_words), first_frame, last_frame,
            )

        #TODO: Soon there will be an approval step here. Basically it will be the "export" step after showing a preview of the motion graphics.
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
