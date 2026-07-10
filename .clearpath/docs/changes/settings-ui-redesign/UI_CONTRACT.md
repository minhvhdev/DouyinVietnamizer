# UI Contract — Settings Redesign

## Layout shell
| Breakpoint | Behavior |
|------------|----------|
| `< lg` | Horizontal tab bar scrolls; single column fields; TTS preview below config |
| `lg+` | TTS: `grid-cols-[1fr_300px]` — preview sticky top |
| All | Remove inner left `settings-nav-panel`; use horizontal tabs under header |

## Components to implement in React (`App.tsx` + `styles.css`)

### `SettingsTabBar`
- 5 tabs with icon + label on `sm+`, label-only on xs
- Active: bottom border `dv-accent`, not full card highlight
- Health dot on tab (ready / attention)

### `SettingsPanel`
- Single `rounded-2xl` container, not multiple nested cards per field group
- Sections separated by `border-t border-dv-line`

### `SettingsEngineRail`
- Horizontal scroll `flex gap-2 overflow-x-auto`
- Cards `min-w-[140px]`, active ring accent
- Replaces `settings-engine-grid` 5-column grid

### `SettingsFieldGrid`
- `grid-cols-1 sm:grid-cols-2 gap-4`
- Full-width spans for textareas / warnings

### `SettingsPreviewAside`
- Sticky `lg:sticky lg:top-4` only on TTS tab
- Gradient accent border treatment

### `SettingsDisclosure`
- Keep existing component; restyle to match prototype accordion

## CSS tokens (reuse existing)
- `--dv-bg: #0c0e13`, accent `#7357ff`, surface `#141820`
- Remove: `settings-nav-panel`, `settings-layout` 2-col inner nav
- Update: `settings-readiness` → header chips (keep)

## Files to change
- `frontend/src/renderer/App.tsx` — settings JSX structure
- `frontend/src/renderer/styles.css` — new layout classes, remove obsolete

## Tests
- Update `App.test.tsx` if tab selectors change (aria-label on tabs)
- Visual: resize window 360px / 768px / 1280px

## Acceptance
- [ ] No inner left sidebar in settings
- [ ] All 5 tabs accessible and fields preserved
- [ ] TTS engine rail scrolls on narrow width
- [ ] Preview sticky on desktop, stacked on mobile
- [ ] No horizontal overflow on 360px viewport
