# Complete GitLab CI/CD & Self-Hosted Runner Documentation

This document provides an end-to-end guide for setting up, registering, configuring, and executing an MLOps pipeline on a self-hosted Ubuntu VPS using GitLab CI/CD and the Shell executor.

---

## 1. Architecture Overview

GitLab CI/CD separates pipeline management from job execution:

* **GitLab Control Plane (Manager):** Hosts the repository, handles pipeline triggers, manages user permissions, and displays live build logs.
* **GitLab Runner (Worker):** A service running on your self-hosted Ubuntu VPS that listens for pending jobs, checks out your repository code, sets up environments, and executes heavy workloads (e.g., machine learning training).

### Why Use a Self-Hosted Runner?

* **Direct Local Resource Access:** Executes scripts directly on your VPS hardware (local filesystems, persistent models, database paths, GPU resources).
* **No Quota Limits:** Bypasses public cloud CI/CD minute caps.
* **Environment Persistence:** Allows caching virtual environments and heavy packages locally to speed up build times.

---

## 2. Prerequisites

* An Ubuntu VPS server with `sudo` privileges.
* `gitlab-runner` binary installed on the server.
* Python 3.10+ and `python3-venv` installed on the host.
* A GitLab repository with **Maintainer** or **Owner** access.

---

## 3. GitLab Runner Setup & Registration on VPS

### Step 3.1: Generate a Runner Token in GitLab

1. In your project, go to **Settings > CI/CD**.
2. Expand **Runners** and click **New project runner**.
3. Add relevant tags (e.g., `Test_runs`).
4. Click **Create runner**.
5. Copy the generated runner authentication token (starts with `glrt-`).

### Step 3.2: Clean Up Old / Conflicting Runners (If Needed)

If previous registration attempts failed or expired tokens exist in `/etc/gitlab-runner/config.toml`:

```bash
# Stop the runner service
sudo gitlab-runner stop

# Unregister all invalid or stale runner profiles
sudo gitlab-runner unregister --all-runners

```

### Step 3.3: Register the Runner Interactively

Execute the registration command on your VPS terminal:

```bash
sudo gitlab-runner register --url https://gitlab.com --token <YOUR_RUNNER_TOKEN>

```

When prompted, provide the following parameters:

| Prompt | Recommended Input | Description |
| --- | --- | --- |
| **GitLab instance URL** | `[https://gitlab.com](https://gitlab.com)` | Press **Enter** to accept default |
| **Runner name** | `mlops-vps` | Local identifier stored in `config.toml` |
| **Tags** | `Test_runs` | Must match tags defined in `.gitlab-ci.yml` |
| **Executor** | `shell` | Runs pipeline scripts directly in bash on the VPS |

### Step 3.4: Verify and Start the Service

```bash
# Start the runner service
sudo gitlab-runner start

# Confirm running status
sudo gitlab-runner status

```

In GitLab (**Settings > CI/CD > Runners**), your runner will display a **green active indicator**.

---

## 4. Pipeline Configuration (`.gitlab-ci.yml`)

The pipeline structure is configured via `.gitlab-ci.yml` in the root of your project:

```yaml
stages:
  - run

workflow:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
    - if: '$CI_PIPELINE_SOURCE == "push"'

run-pipeline:
  stage: run
  timeout: 5 hours
  tags:
    - Test_runs
  script:
    - echo "Activating virtual environment and installing dependencies..."
    - python3 -m venv venv || true
    - source venv/bin/activate
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - echo "Running main ML pipeline..."
    - python main.py

```

---

## 5. Pipeline Execution Workflow

When a build job triggers on the runner, the execution follows these stages:

```text
[ Git Push / Schedule ]
          │
          ▼
[ GitLab CI/CD Dispatcher ]
          │ (Matches 'Test_runs' tag)
          ▼
[ VPS Shell Executor ]
          │
          ├── 1. Repository Checkout (Git clone/fetch into /home/gitlab-runner/builds)
          ├── 2. Environment Setup (Python virtualenv creation & activation)
          ├── 3. Dependency Installation (pip install -r requirements.txt)
          └── 4. Pipeline Execution (python main.py execution & live logging)

```

---

## 6. Troubleshooting Guide

### Issue 1: `status=400 Bad Request / Failed to verify runner`

* **Cause:** Token expired or previously consumed during a failed interactive run.
* **Fix:** Create a new project runner token in GitLab settings, run `sudo gitlab-runner unregister --all-runners`, and register using the new token.

### Issue 2: `parse URL: first path segment in URL cannot contain colon`

* **Cause:** Quotes, parentheses, or trailing slashes wrapped around `[https://gitlab.com](https://gitlab.com)`.
* **Fix:** Ensure `--url [https://gitlab.com](https://gitlab.com)` is entered cleanly without wrapping quotes or trailing parameters.

### Issue 3: Job stuck in "Pending" status

* **Cause:** No online runner matches the tags listed in `.gitlab-ci.yml`.
* **Fix:** Check that the runner tag in GitLab matches the `tags:` section under `run-pipeline` (e.g., `Test_runs`).

---

---

## 7. Advanced: Optimizing MLOps with Caching

Machine Learning dependencies (like `torch`, `xgboost`, `pandas`) are massive. Downloading them on every pipeline run wastes time and bandwidth. Because you are using the **Shell Executor**, you can configure GitLab to cache your `venv/` directory or pip's download cache locally on the VPS.

Update your `.gitlab-ci.yml` to include a `cache` block:

```yaml
run-pipeline:
  stage: run
  timeout: 5 hours
  tags:
    - Test_runs
  variables:
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  cache:
    key: "ml-dependencies"
    paths:
      - .cache/pip
      - venv/
  script:
    - python3 -m venv venv
    - source venv/bin/activate
    - pip install --upgrade pip
    - pip install -r requirements.txt
    - python main.py

```

* **How it works:** After the first successful run, GitLab will compress the `venv/` and `.cache/pip` folders. On subsequent runs, it restores them instantly before executing the `script`, cutting your setup time from minutes down to seconds.

---

## 8. Advanced: CI/CD Variables & Secrets Management

Hardcoding database credentials, S3 bucket keys, or MLflow tracking URIs in `main.py` is a massive security risk. Instead, use GitLab's **CI/CD Variables** to securely pass them to your VPS.

### How to set them up:

1. Go to your GitLab project **Settings > CI/CD**.
2. Expand the **Variables** section and click **Add variable**.
3. Add your keys (e.g., Key: `MLFLOW_TRACKING_URI`, Value: `http://localhost:5000`).
4. Check **Mask variable** so it never prints in your pipeline logs.

In your `.gitlab-ci.yml` or `main.py`, these are automatically injected as environment variables.

```python
# In main.py
import os
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

```

---

## 9. Advanced: Capturing ML Artifacts

When your `main.py` script finishes training (e.g., CatBoost, XGBoost models), you likely want to save the trained model files (`.pkl`, `.joblib`) or performance reports (`metrics.csv`) so you can download them directly from the GitLab UI.

Update your `.gitlab-ci.yml` to define an `artifacts` path:

```yaml
run-pipeline:
  stage: run
  # ... (previous config) ...
  script:
    - source venv/bin/activate
    - python main.py
  artifacts:
    name: "model-outputs-$CI_COMMIT_SHORT_SHA"
    paths:
      - models/            # Saves the entire 'models' directory generated by main.py
      - metrics.csv        # Saves specific files
    expire_in: 1 week      # Automatically deletes them after a week to save VPS disk space

```

* **How it works:** Once the job succeeds, GitLab Runner uploads these files to GitLab. You can browse and download them directly from the **Build > Artifacts** page in your repository.

---

## 10. Understanding VPS Permissions (The `gitlab-runner` User)

When using the Shell Executor, your pipeline does **not** run as the `root` or `ubuntu` user. It runs as a dedicated background system user named `gitlab-runner`.

* **Workspace Location:** The code is cloned and executed inside `/home/gitlab-runner/builds/`.
* **Permission Denied Errors:** If your `main.py` tries to write a file to `/root/` or `/var/www/`, it will fail. Ensure your script only reads/writes data within the current working directory (`./`), which `gitlab-runner` fully owns.
* **If you need Docker:** If your MLOps pipeline requires building Docker images (e.g., containerizing the model API), you must add the runner to the docker group on your VPS:
```bash
sudo usermod -aG docker gitlab-runner

```
