# Design Read

**Reading this as:** desktop product settings for power users running a local GPU dubbing pipeline, with a dark technical-tool language, leaning toward earned familiarity (Linear / Raycast / Stripe Dashboard) using existing DV purple accent (`#7357ff`) on `#0c0e13` shell.

## Dials
- DESIGN_VARIANCE: **5** — symmetric, predictable layout; no artsy asymmetry in forms
- MOTION_INTENSITY: **3** — 150–200ms state transitions only
- VISUAL_DENSITY: **5** — dense but scannable; settings are task-focused

## Core IA changes
1. **Horizontal tab bar** replaces inner left sidebar — recovers ~220px content width; scrollable on narrow windows
2. **Section-based layout** replaces nested card stacks — one surface, dividers between groups
3. **TTS engine** — scrollable pill rail + single active config panel; preview sticky on `lg+`, stacked below on mobile
4. **Readiness** — compact chips in header, not duplicate pipeline row
5. **Responsive grid** — `1 col` → `2 col` at `md` → TTS `config | preview` at `lg`

## Non-goals
- Changing main app sidebar
- Removing advanced disclosures (Gemini keys, VAD tuning, duration repair)
