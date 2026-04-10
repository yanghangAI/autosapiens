# Autonomous Research Pipeline

## The Agents

**1. Orchestrator (The Coordinator)**
* **Role:** The central manager of the entire research pipeline. It is the only entity allowed to spawn other sub-agents.
* **Responsibilities:** Delegates tasks sequentially, passes directory paths between agents (acting as a zero-overhead messenger), and manages the pipeline state. It determines when to spawn the Architect, Designer, Builder, or Reviewer based on the current filesystem state and tracker statuses.

**2. Architect (The Ideation Lead)**
* **Role:** The high-level researcher defining the structural search space.
* **Responsibilities:** Reflects on past experiments via `results.csv`, conceptualizes novel design axes or architectural changes (ensuring they fit within a 20-epoch 1080ti budget), and decides how many detailed variations to explore per idea.

**3. Designer (The Technical Draftsman)**
* **Role:** The detail-oriented blueprint creator.
* **Responsibilities:** Takes the Architect's broad ideas and writes exact, mathematically sound, implementable configuration variations (saved to `design.md`). Explicitly defines the starting point (e.g., a previous design or the baseline) and all experimental hyperparameters.

**4. Builder (The Software Engineer)**
* **Role:** The PyTorch implementation layer.
* **Responsibilities:** Takes a finalized `design.md` and writes/modifies the Python training loop (`train.py`) to match it. It is responsible for making sure the code runs by executing local 2-epoch test runs (`python scripts/cli.py submit-test <design_dir>`) to ensure there are no immediate code/OOM crashes before requesting a code review.

**5. Reviewer (The Strict Gatekeeper)**
* **Role:** The dedicated auditor for both designs and code.
* **Responsibilities:** Replaces the review overhead previously handled by the Architect and Designer. 
  - *Design Review:* Evaluates a drafted `design.md` against mathematical feasibility and the 11GB VRAM budget.
  - *Code Review:* Evaluates the Builder's `train.py` and `config.py` to ensure they strictly implement the specifications defined in `design.md` with no hardcoded shortcuts.

---

## Script Execution Model

This repository now prefers explicit command execution over automatic hooks or cron-driven workflow changes. Agents should call the script tooling directly when the workflow reaches the appropriate step.

Typical explicit commands:

- `python scripts/cli.py summarize-results`
- `python scripts/cli.py sync-status`
- `python scripts/cli.py submit-test <design_dir>`
- `python scripts/cli.py submit-train <train.py> <job_name>`
- `python scripts/cli.py submit-implemented`
- `python scripts/cli.py update-all`

This keeps status transitions, submissions, and dashboard updates deliberate and easy to debug.

---

## The Working Pipeline

1. **Ideation:** The **Orchestrator** spawns the **Architect**, who proposes a new research 'Idea' based on past results and states how many variations to explore.
2. **Drafting (batched):** The Orchestrator spawns the **Designer**, which drafts **all** requested design variations for the idea in one run, then returns the list of design folder paths.
3. **Design Review (batched):** The Orchestrator spawns the **Reviewer** with all design paths. The Reviewer evaluates each `design.md` and returns per-design verdicts. If any are rejected, the Orchestrator re-spawns the Designer with rejection feedback, then re-spawns the Reviewer — looping until all designs are approved. The Orchestrator runs `python scripts/cli.py sync-status` after approvals.
4. **Implementation (batched):** The Orchestrator spawns the **Builder**, which implements and sanity-tests **all** 'Not Implemented' designs in one run, then returns the list of completed design folder paths.
5. **Code Audit (batched):** The Orchestrator spawns the **Reviewer** with all implemented design paths. The Reviewer checks each `train.py` against its `design.md` and returns per-design verdicts. If any are rejected, the Orchestrator re-spawns the Builder with rejection feedback, then re-spawns the Reviewer — looping until all code is approved. The Orchestrator runs `python scripts/cli.py sync-status` and then `python scripts/cli.py submit-implemented` to put jobs in the SLURM queue.
6. **Execution (Asynchronous):** The AI agents can move on to new tasks. When the user or agent wants to refresh metrics and statuses from completed training runs, they explicitly run `python scripts/cli.py summarize-results` followed by `python scripts/cli.py sync-status`.
