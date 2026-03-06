## When to use

Use `animated_title` for opening titles, topic transitions, and quoting key statements from the speaker. It works best when there's a clear moment worth highlighting — a section change, an important claim, or a memorable phrase. One title every 15-20 seconds of content is a good rhythm; fewer is usually better.

## Style matching

- **fade** — default, works for calm or conversational delivery. Pair with lower font weights (400-500).
- **slide-in** — good for structured content with clear sections. Feels editorial.
- **typewriter** — energetic, draws attention. Use when the speaker is making a punchy statement or listing items.
- **bounce** — playful, high-energy. Best for enthusiastic speakers or humorous moments. Avoid for serious/professional tone.

Match `fontWeight` to emphasis: 400-500 for ambient labels, 700-900 for key statements.

## When NOT to use

- Don't use as a subtitle replacement — if the text just mirrors what's being said word-for-word with no editorial value, skip it.
- Don't use for names or speaker introductions — that's what `lower_third` is for.
- Don't stack multiple titles close together (< 2s gap). Let each one breathe.
- Avoid during zoom effects unless the title fits within the zoomed viewport.

## Placement

Default to the top 20% of the frame (y: 0.05-0.20) or center for emphasis moments. Always check safe_regions to avoid the face. Keep text short — 3-8 words works best.
