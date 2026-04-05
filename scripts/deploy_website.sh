#!/bin/bash
set -e

# Get the current branch (just in case)
current_branch=$(git branch --show-current)

# Trap any exit (like Ctrl+C or a crash) to ensure we always return to the original branch
cleanup() {
    echo "Cleaning up..."
    # Only try to checkout and stash pop if we actually left the original branch
    if [ "$(git branch --show-current)" != "$current_branch" ]; then
        git checkout "$current_branch" || true
        git stash pop -q || true
    fi
}
trap cleanup EXIT INT TERM

echo "1) Generating the website..."
python scripts/generate_website.py

echo "2) Committing current changes to main (results, runs, scripts)..."
git add results.csv runs/ scripts/
if ! git diff-index --quiet HEAD; then
    git commit -m "Auto-update results and scripts [$(date +'%Y-%m-%d %H:%M:%S')]"
    git push origin $current_branch
else
    echo "   No changes to commit to main."
fi

echo "3) Deploying to gh-pages branch..."
# Stash any other modifications just in case so checkout doesn't fail
git stash -q

git checkout gh-pages
cp website/index.html .
git add index.html

if ! git diff-index --quiet HEAD; then
    git commit -m "Auto-deploy website [$(date +'%Y-%m-%d %H:%M:%S')]"
    git push origin gh-pages
else
    echo "   No changes to index.html."
fi

# The cleanup trap will automatically handle the git checkout back to the $current_branch
# upon exit, including a successful finish.

echo "✅ Website successfully updated and pushed to GitHub!"
