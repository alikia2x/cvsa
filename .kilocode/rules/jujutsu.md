# Jujutsu (jj) Quick Reference

This project uses Jujutsu(jj) with git for version control. This document shows how to use jj.

## Setup
```bash
jj git init --colocate          # Initialize in existing git repo
jj config set --user user.name "Name"
jj config set --user user.email "email"
```

## Daily Workflow
```bash
jj new                         # Create new change
jj desc -m "message"          # Add message
jj commit -m "message"        # New change + message
jj status                     # View status
jj log                        # View history
```

## Navigation
```bash
jj edit CHANGE-ID             # Edit specific change
@                            # Current change
@-                           # Parent change
@::                          # All descendants
jj undo                      # Undo last operation
```

## Bookmarks (Branches)
```bash
jj bookmark set -r @ name     # Create bookmark
jj bookmark list             # List bookmarks
jj git push -b name          # Push to remote
jj git fetch                # Fetch remote
```

## History Operations
```bash
jj split                     # Split change
jj squash                    # Squash into parent
jj abandon CHANGE-ID         # Remove change
jj rebase -d DESTINATION     # Move change(s)
```

## Git → jj Mappings
- `git status` → `jj status`
- `git commit -m` → `jj commit -m`
- `git branch` → `jj bookmark`
- `git checkout` → `jj edit`
- `git rebase` → `jj rebase`
- `git stash` → `jj new @-`
- `git add` → Automatic

## Common Patterns
```bash
# Feature work
jj new; jj commit -m "Feature"

# Update bookmark
jj new; jj commit -m "Updates"
jj bookmark set -r @ feature; jj git push

# Sync with remote
jj git fetch; jj rebase -d trunk

# Quick fix
jj new @-; # changes...
jj bookmark set -r @ fix; jj git push
```

## Key Concepts
- Changes are mutable
- No staging (automatic)
- Branchless by default
- `jj undo` reverses operations
- Revsets: `@`, `@-`, `@::`