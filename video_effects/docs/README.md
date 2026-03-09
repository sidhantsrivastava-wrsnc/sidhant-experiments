# video_effects Documentation

Automated video post-production powered by LLM analysis, OpenCV frame processing, and Remotion motion graphics — orchestrated by Temporal workflows.

## Quick Start

```bash
# 1. Start the Temporal worker
python -m video_effects.worker

# 2. Run the pipeline on a video
python -m video_effects.cli run input.mp4 --output output.mp4

# 3. With motion graphics + infographics
python -m video_effects.cli run input.mp4 -o output.mp4 --motion-graphics --infographics

# 4. Auto-approve (skip interactive review)
python -m video_effects.cli run input.mp4 -o output.mp4 --mg --auto-approve --style bold-energy
```

## How It Works

1. **Transcribe** the video audio (ElevenLabs / Whisper)
2. **Analyze** the transcript with an LLM to infer effects cues
3. **Apply** OpenCV effects in a single-pass frame pipeline
4. **Plan** motion graphics overlays using LLM + face tracking context
5. **Render** transparent overlays via Remotion (ProRes 4444)
6. **Composite** everything with FFmpeg

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System overview, pipeline stages, end-to-end data flow |
| [Effects Pipeline](effects-pipeline.md) | OpenCV frame processing: phase ordering, each effect's internals, encoder settings |
| [Motion Graphics](motion-graphics.md) | Remotion MG system: LLM planning, spatial validation, ProRes rendering, FFmpeg compositing |
| [Infographics](infographics.md) | Code-gen pipeline: plan → generate TSX → validate → registry → fallback |
| [Remotion Components](remotion-components.md) | Component implementations, animation patterns, hooks API, props interfaces |
| [Face Tracking](face-tracking.md) | Face detection, spatial context, safe regions, zoom compensation, anchor modes |
| [Styles](styles.md) | Style presets, theming system, font loading, palette conventions, creative designer |
| [LLM Prompts](llm-prompts.md) | Prompt system, structured output, feedback loops, model selection |
| [CLI & Config](cli-and-config.md) | CLI commands, interactive approval flow, environment variables, Temporal worker setup |

## Project Structure

```
video_effects/
├── cli.py                      # CLI entry point
├── worker.py                   # Temporal worker
├── config.py                   # Settings (VFX_ env vars)
├── workflow.py                 # Main VideoEffectsWorkflow
├── creative_workflow.py        # Style auto-detection child workflow
├── infographic_workflow.py     # Code-gen child workflow
├── effect_registry.py          # Phase ordering & effect type → processor map
├── activities/
│   ├── apply_effects.py        # Single-pass frame pipeline
│   ├── parse_cues.py           # LLM effect cue parsing
│   ├── creative.py             # Style design activity
│   ├── remotion.py             # MG planning, spatial context, rendering
│   ├── infographic.py          # Infographic code generation & validation
│   ├── compose.py              # Final audio mux
│   ├── transcribe.py           # Audio transcription
│   └── video_info.py           # Video metadata extraction
├── effects/
│   ├── base.py                 # BaseEffect ABC
│   ├── zoom.py                 # Face-tracked zoom with easing
│   ├── blur.py                 # Gaussian, face pixelate, background, radial
│   ├── color.py                # Color grading presets
│   ├── whip.py                 # Whip transition
│   ├── vignette.py             # Cinematic vignette
│   ├── speed_ramp.py           # Visual speed effect
│   └── subtitle.py             # Text burn-in (legacy)
├── helpers/
│   ├── llm.py                  # Anthropic API wrapper (call_structured, call_text)
│   ├── face_tracking.py        # MediaPipe face detection pipeline
│   └── remotion.py             # Remotion render + FFmpeg composite helpers
├── schemas/
│   ├── effects.py              # EffectCue, EffectType, VideoInfo, effect params
│   ├── styles.py               # StylePreset, StyleConfig, FontWeights
│   ├── mg_templates.py         # MG template registry & specs
│   ├── motion_graphics.py      # MotionGraphicsComponent, Plan
│   ├── infographic.py          # InfographicSpec, InfographicType, fallback map
│   └── workflow.py             # VideoEffectsInput/Output
├── prompts/
│   ├── parse_effect_cues.md    # Effect cue inference prompt
│   ├── parse_effect_cues_dev.md # Dev mode (explicit verbal cues)
│   ├── design_style.md         # Style auto-detection prompt
│   ├── plan_motion_graphics_base.md  # MG planning prompt
│   ├── plan_infographics.md    # Infographic planning prompt
│   ├── generate_infographic_code.md  # TSX code generation prompt
│   ├── infographic_api_reference.md  # Allowed imports for generated code
│   ├── schema.py               # ParsedEffectCues response model
│   ├── motion_graphics_schema.py # MG plan response model
│   └── mg_guidance/            # Per-template creative guidance
│       ├── animated_title.md
│       ├── lower_third.md
│       ├── listicle.md
│       └── data_animation.md
└── remotion/                   # TypeScript/React Remotion project
    ├── package.json            # Remotion 4.0.242 + React 18
    └── src/
        ├── Root.tsx            # Composition registration
        ├── DynamicComposition.tsx  # Runtime composition engine
        ├── types.ts            # All TypeScript interfaces
        ├── components/
        │   ├── index.ts        # ComponentRegistry
        │   ├── AnimatedTitle.tsx
        │   ├── LowerThird.tsx
        │   ├── Listicle.tsx
        │   ├── DataAnimation.tsx
        │   ├── Subtitles.tsx
        │   └── generated/      # LLM-generated infographic components
        │       └── _registry.ts
        └── lib/
            ├── spatial.ts      # useFaceAwareLayout, zoom compensation
            ├── styles.ts       # StyleProvider, useStyle
            ├── context.ts      # FaceDataProvider, useFaceFrame
            ├── zoom-context.ts # ZoomDataProvider, useZoomFrame
            ├── easing.ts       # Spring configs (GENTLE, BOUNCY, SNAPPY, SMOOTH)
            ├── fonts.ts        # Google Fonts loader
            └── infographic-utils.ts  # SVG math helpers
```
