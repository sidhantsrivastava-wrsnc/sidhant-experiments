# Architecture

## System Overview

video_effects is a Temporal-orchestrated video post-production pipeline. It transcribes a video, uses an LLM to infer where effects should go, applies OpenCV frame-level effects in a single pass, and optionally renders Remotion motion graphics overlays that are composited via FFmpeg.

Three Temporal workflows coordinate the work:

| Workflow | Role |
|----------|------|
| `VideoEffectsWorkflow` | Main orchestrator (G1–G9) |
| `CreativeDesignerWorkflow` | Auto-detect style preset from transcript |
| `InfographicGeneratorWorkflow` | LLM-generated TSX components (A0–A4) |

## End-to-End Data Flow

```
                         ┌─────────────┐
                         │  Input .mp4 │
                         └──────┬──────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                │
        ┌─────────────┐  ┌───────────┐           │
   G1a  │ Video Info   │  │ Extract   │  G1b      │
        │ (ffprobe)    │  │ Audio     │           │
        └──────┬──────┘  └─────┬─────┘           │
               │               │                  │
               │               ▼                  │
               │        ┌────────────┐            │
               │   G2   │ Transcribe │            │
               │        │ (ElevenLabs│            │
               │        │  / Whisper)│            │
               │        └─────┬──────┘            │
               │              │                   │
               │    ┌─────────┴──────────┐        │
               │    │                    │        │
               │    ▼                    ▼        │
               │ ┌──────────┐  ┌──────────────┐  │
               │ │ Creative │  │ Parse Effect │  │
               │ │ Designer │  │ Cues (LLM)   │  │
               │ │ (LLM)    │  │              │  │
               │ └────┬─────┘  └──────┬───────┘  │
               │      │               │          │
               │      │          ┌────┴────┐     │
               │      │          │Validate │     │
               │      │          │Timeline │     │
               │      │          └────┬────┘     │
               │      │               │          │
               │      │          ┌────┴────┐     │
               │      │     G5   │  HITL   │     │
               │      │          │Approval │◄─── │─── User
               │      │          └────┬────┘     │
               │      │               │          │
               │      │     ┌─────────┴──────────┤
               │      │     │                    │
               ▼      ▼     ▼                    ▼
        ┌──────────────────────────────────────────┐
        │           Post-approval setup            │
        │  • Inject color grading from style       │
        └────────┬───────────┬───────────┬─────────┘
                 │           │           │
    ┌────────────┤     ┌─────┴─────┐     │
    │            │     │  G8a:     │     │
    │            │     │  Build    │     │
    │            │     │  Spatial  │     │
    │            │     │  Context  │     │
    │            │     └─────┬─────┘     │
    │            │           │           │
    │   ┌────────┤     ┌─────┴─────┐     ├────────────┐
    │   │        │     │G8b: Plan  │     │            │
    │   │        │     │MG (LLM)  │     │            │
    │   │        │     │+ validate │     │            │
    │   │        │     │+ approve  │     │            │
    │   │        │     └─────┬─────┘     │            │
    │   │        │           │           │            │
    ▼   ▼        ▼           │           ▼            ▼
 ┌────────┐ ┌────────┐      │     ┌───────────┐ ┌──────────┐
 │Setup   │ │Render  │      │     │Infographic│ │Compose   │
 │Process-│ │Video   │      │     │Generator  │ │Final     │
 │ors     │ │(single │      │     │(child wf) │ │(mux      │
 │(G6b)   │ │ pass)  │      │     │           │ │ audio)   │
 │        │ │(G6c)   │      │     │A0→A4      │ │(G7)      │
 └────────┘ └───┬────┘      │     └─────┬─────┘ └────┬─────┘
                │            │           │             │
                │            │     ┌─────┴─────┐       │
                │            │     │  Merge    │       │
                │            │     │  infogr.  │       │
                │            │     │  into MG  │       │
                │            │     │  plan     │       │
                │            │     └─────┬─────┘       │
                │            │           │             │
                │            ├───────────┘             │
                │            │                         │
                │            ▼                         │
                │     ┌─────────────┐                  │
                │     │G8e: Render  │                  │
                │     │MG Overlay   │                  │
                │     │(Remotion    │                  │
                │     │ ProRes 4444)│                  │
                │     └──────┬──────┘                  │
                │            │                         │
                │            ▼                         │
                │     ┌─────────────┐                  │
                │     │G9: FFmpeg   │◄─────────────────┘
                │     │Composite    │
                │     │overlay onto │
                │     │base video   │
                │     └──────┬──────┘
                │            │
                └────────────┘
                         │
                  ┌──────┴──────┐
                  │ Output .mp4 │
                  └─────────────┘
```

## Component Map

### Python Layer

| Component | File | Responsibility |
|-----------|------|----------------|
| CLI | `cli.py` | Parse args, trigger workflow, interactive approval UI |
| Worker | `worker.py` | Register workflows + activities, run Temporal worker |
| Config | `config.py` | `VFX_*` environment variables via Pydantic BaseSettings |
| Main Workflow | `workflow.py` | `VideoEffectsWorkflow` — orchestrates G1–G9 |
| Creative Workflow | `creative_workflow.py` | Auto-style detection child workflow |
| Infographic Workflow | `infographic_workflow.py` | Code-gen child workflow (A0–A4) |
| Effect Registry | `effect_registry.py` | Phase ordering, `EffectType` → processor map |
| LLM Helper | `helpers/llm.py` | `call_structured()`, `call_text()`, `load_prompt()` |
| Face Tracking | `helpers/face_tracking.py` | MediaPipe detection + EMA smoothing |
| Remotion Helpers | `helpers/remotion.py` | `render_media()`, `composite_overlay()` |

### Activities (27 total)

| Group | Activities |
|-------|-----------|
| Video extraction | `vfx_get_video_info`, `vfx_extract_audio` |
| Transcription | `vfx_transcribe_audio` |
| Effect parsing | `vfx_parse_effect_cues` |
| Timeline | `vfx_validate_timeline` |
| Render | `vfx_prepare_render`, `vfx_setup_processors`, `vfx_render_video` |
| Composition | `vfx_compose_final` |
| Motion graphics | `vfx_build_remotion_context`, `vfx_plan_motion_graphics`, `vfx_validate_merged_plan`, `vfx_load_composition_plan`, `vfx_render_motion_overlay`, `vfx_composite_motion_graphics`, `vfx_preview_motion_graphics` |
| Infographics | `vfx_cleanup_generated`, `vfx_plan_infographics`, `vfx_plan_diagrams`, `vfx_plan_timelines`, `vfx_plan_quotes`, `vfx_plan_code_blocks`, `vfx_plan_comparisons`, `vfx_generate_infographic_code`, `vfx_validate_infographic`, `vfx_build_generated_registry` |
| Creative | `vfx_design_style` |

### TypeScript Layer (Remotion)

| Component | File | Responsibility |
|-----------|------|----------------|
| Root | `remotion/src/Root.tsx` | Composition registration |
| DynamicComposition | `remotion/src/DynamicComposition.tsx` | Runtime composition engine |
| ComponentRegistry | `remotion/src/components/index.ts` | Template → React component lookup |
| 5 Built-in Components | `remotion/src/components/*.tsx` | AnimatedTitle, LowerThird, Listicle, DataAnimation, Subtitles |
| Generated Registry | `remotion/src/components/generated/_registry.ts` | LLM-generated infographic components |
| Spatial Hook | `remotion/src/lib/spatial.ts` | `useFaceAwareLayout()`, zoom compensation |
| Style Context | `remotion/src/lib/styles.ts` | `StyleProvider`, `useStyle()` |
| Face Context | `remotion/src/lib/context.ts` | `FaceDataProvider`, `useFaceFrame()` |
| Zoom Context | `remotion/src/lib/zoom-context.ts` | `ZoomDataProvider`, `useZoomFrame()` |

## Pipeline Stages

### Stage 1: Analysis (G1–G2)

Extract video metadata via ffprobe, split audio, and transcribe with word-level timestamps. These run in parallel where possible.

### Stage 2: Creative Design

A child workflow sends a transcript excerpt to the LLM to auto-detect the best [style preset](styles.md) (or uses an explicit `--style` override).

### Stage 3: Effect Planning (G3–G5)

The LLM reads the transcript and infers effect cues — zooms, color grades, whip transitions, etc. A validation pass resolves conflicts (overlapping cues, unpaired zooms). The user reviews and approves or rejects with feedback (up to 5 rounds).

### Stage 4: Parallel Processing (G6–G8)

Two streams run concurrently after approval:

1. **OpenCV render** (G6b–G6c): Single-pass frame pipeline applies effects in phase order. See [Effects Pipeline](effects-pipeline.md).
2. **Infographic generation** (G8a + child workflow): Build spatial context (face windows, safe regions, zoom state), then 6 specialist planners run in parallel (infographics, diagrams, timelines, quotes, code blocks, comparisons), results merged by score, then LLM generates custom TSX components, validates with TypeScript + test renders, falls back to templates on failure. Both `--mg` and `--infographics` route through this pipeline. See [Infographics](infographics.md).

### Stage 5: Composition (G7–G9)

1. Mux processed video with original audio (G7)
2. Merge infographic components into MG plan, re-validate, inject subtitles
3. Render transparent overlay via Remotion ProRes 4444 (G8e)
4. FFmpeg alpha-composite overlay onto base video (G9)

## Key Design Decisions

- **Single-pass frame processing**: No intermediate files between effects. Decode → apply all effects in phase order → encode.
- **Phase ordering**: Effects execute in strict numeric order (vignette → color → blur → whip → zoom → speed_ramp) to ensure correct composition.
- **Transparent overlays**: MG rendered as ProRes 4444 with alpha, composited via FFmpeg `premultiply + overlay`. Keeps the OpenCV and Remotion pipelines fully decoupled.
- **LLM-driven planning**: Effects, motion graphics, and infographics are all planned by LLMs analyzing the transcript — not by explicit user markup.
- **Human-in-the-loop**: One approval gate (timeline) with feedback loop, skippable via `--auto-approve`.
- **Face awareness**: Face tracking data flows through the entire system — from zoom dampening in OpenCV to spatial layout hooks in Remotion components.
