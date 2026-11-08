#!/bin/sh
cd "$(dirname "$0")"
{
echo "== gh auth"; gh auth status 2>&1 | head -5
rm -f .git/index.lock .git/*.lock 2>/dev/null
git add -A
git -c user.name="Mukesh Agrawal" -c user.email="mukeshkbj@gmail.com" commit -q -m "Lost in Summary: experiments, results and report assets

All code written during the Apart Swarm Alignment sprint (6-8 Nov 2026).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012F8tgYdfnfmePS8QxUy78j" && echo committed
git branch -M main
git remote remove origin 2>/dev/null
git remote add origin https://github.com/mukeshkbj/lost-in-summary.git
git push -u origin main 2>&1
echo "== exit $?"
git log --oneline | head -3
git ls-files | grep -c . 
git ls-files | grep -E "\.env$" || echo "no .env tracked"
} > push_repo.out 2>&1
