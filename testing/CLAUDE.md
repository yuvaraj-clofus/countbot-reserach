# Countbot

## How to Approach Tasks

1. **Start with `docs/overview.md`** — understand the feature area and where it fits
2. **Find the relevant feature doc** in `docs/claude/features/` — read it fully before touching code
3. **Open only files referenced** by the feature doc — don't scan the entire repo
4. **Read skills before implementing** — see `.claude/skills/` table below
5. **Update docs** if you change architecture or add new patterns (PR merge auto-runs post-merge-cleanup)

Never open files speculatively. Only read what the feature doc points to.

---

## Documentation Structure

```
docs/
  overview.md              ← start here for any task
  claude/
    features/              ← per-feature context and rules
      dark-mode.md         ← MANDATORY before any UI work
      ...
.claude/
  skills/                  ← reusable coding conventions (read when relevant)
    ui-styling.md          ← mac design system + dark mode patterns
    autocrud.md            ← backend collection scaffolding
```

**Progressive reading**: `overview.md` → feature doc → implementation files. Stop as soon as you have enough context.

---

## Skills (read when implementing related functionality)

| Skill | When to read |
|-------|--------------|
| `.claude/skills/ui-styling.md` | **Any UI task** — mac design system classes, dark mode rules, color palette, form patterns |
| `.claude/skills/autocrud.md` | Adding new backend collections |

---

## Dark Mode — MANDATORY before any UI work

Read `docs/claude/features/dark-mode.md` before writing any HTML, CSS, or Angular template.

Key rules (details in the doc):
- Never hardcode `color`, `background`, or `border` in inline `style` attributes
- Every new component CSS block needs a `[data-bs-theme="dark"]` counterpart
- Use mac design system classes (`mac-list`, `mac-phase-card`, `mac-table-wrap`) — built-in dark support
- Use badge combos `bg-*-subtle text-*` — handled in `dark.css`

---

## Coding Philosophy

- **Follow existing patterns** — match style of surrounding code exactly
- **Minimal changes** — solve the task, nothing more; no unrelated refactoring
- **No new frameworks** — don't add libraries without explicit approval
- **No premature abstractions** — three similar lines > a helper function used once
- **No new services** unless the task specifically requires it
- **Test manually** before marking done

---

## Stack
- **Frontend**: Angular 16 SPA (Bootstrap 5.3, Bootstrap Icons, Angular Material)
- **Backend**: Python 3.11 / Flask (AWS Lambda + API Gateway)
- **Database**: MongoDB Atlas (multi-tenant: `db_{clientid}`)
- **Infrastructure**: AWS Lambda + S3 + CloudFront

## Architecture
```
Browser → CloudFront → S3 (Angular SPA)
                     → API Gateway → Lambda (Flask)
                                        → MongoDB Atlas
                                        → S3 (files)
                                        → External APIs
```

---

## Submitting Work (Team / Manual Dev)
After implementing the feature on the task branch:
```bash
git add -A && git commit -m "Implement: <short description>"
git push origin HEAD
gh pr create --base main --title "$(head -1 task.md | sed 's/^# Task: //')" --body "$(git log origin/main..HEAD --pretty=format:'- %s (%h)' --reverse)"
```
`summary.md` is optional — validate-task auto-generates it from git log if missing.

---

## Coding Rules

### Do
- Follow existing patterns — match the style of surrounding code
- Use the autocrud schema system for new collections (see `.claude/skills/autocrud.md`)
- Use Bootstrap 5.3 + Mac design system CSS classes for UI
- Use role-based directives for access control (`appAdminOnly`, `AppAdminManagerOnlyDirective`, etc.)
- Use `ConfigService.apiUrl` for all API calls
- Keep changes minimal — solve the task, nothing more
- Test manually before marking done

### Don't
- Don't introduce new frameworks or libraries without explicit approval
- Don't add unnecessary error handling, abstractions, or "improvements" beyond the task
- Don't over-engineer — three similar lines > premature abstraction
- Don't create new services unless the task specifically requires it
- Don't modify global styles unless the task is about styling
- Don't skip existing validation or auth patterns

---

## Backend Patterns

### API Route Structure
```python
@bp.route('/endpoint', methods=['POST'])
@token_required
def handler(current_user):
    mclient = MongoClient(current_app.config['MONGO_URI'])
    db = mclient[current_user['databasename']]
    # ... work with db[collection]
```

### Access Control Decorators
| Decorator | Allowed Roles |
|-----------|---------------|
| `@token_required` | Any authenticated user |
| `@access_admin` | admin only |
| `@access_adminandmanager` | admin, manager |
| `@access_adminandmanagerandstandard` | admin, manager, standard |

### Adding New Collections
1. Create schema file in `app/schemas/` with `autocrud: True`
2. Routes are auto-generated — no manual route code needed
3. See `.claude/skills/autocrud.md` for the full schema format

---

## Frontend Patterns

### Component Structure
- Direct `HttpClient` usage in components (no centralized API service)
- State via `BehaviorSubject` in `ConfigService`
- Role directives control element visibility

### CSS Classes (Mac Design System)
| Category | Classes |
|----------|---------|
| Page | `page-container`, `page-header-bar`, `page-icon--blue` |
| Lists | `mac-list`, `mac-list-item`, `mac-list-avatar` |
| Tables | `mac-table-wrap`, `mac-table` |
| Stats | `mac-stats-row`, `mac-stat-card` |
| Cards | `mac-phase-card`, `mac-empty-state` |
| Buttons | `btn-mac-edit`, `btn-mac-delete`, `btn-mac-view` |

### Adding New Screens
1. Create component: `ng generate component feature-name`
2. Add route in `app-routing.module.ts` inside the `LayoutComponent` children
3. Add sidebar link in `sidebar.component.html` with appropriate role directive
4. Declare in `app.module.ts`

---

## Multi-Tenancy
- Global DB `taskman` → users, organizations
- Tenant DB `db_{clientid}` → all business data
- Always use `current_user['databasename']` to get the correct tenant DB
- Never hardcode database names

## File Uploads
- Upload to S3 bucket via `boto3`
- Store S3 key in MongoDB document
- Use presigned URLs for downloads
