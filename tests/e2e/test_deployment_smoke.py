"""Deployment Smoke Test Suite.

Validates:
- Health check programmatic API and CLI commands
- Startup readiness and dataset integrity verification
- Clean-environment chatbot execution in deterministic mode
- Deployment configuration and documentation files
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.api.chatbot import Chatbot, ANSWERED
from src.api.health import run_health_check, main as health_main
from src.data.loaders import load_dataset


ROOT = Path(__file__).resolve().parents[2]


class TestHealthCheck:
    def test_health_check_passes_on_live_data(self):
        report = run_health_check()
        assert report["healthy"] is True
        assert report["status"] == "HEALTHY"
        assert len(report["issues"]) == 0
        assert report["components"]["dataset"]["healthy"] is True
        assert report["components"]["contract"]["healthy"] is True
        assert report["components"]["catalogue"]["healthy"] is True
        assert report["components"]["router"]["healthy"] is True
        assert report["components"]["llm"]["healthy"] is True

    def test_health_check_detects_missing_dir(self, tmp_path):
        fake_dir = tmp_path / "non_existent_data"
        report = run_health_check(data_dir=fake_dir)
        assert report["healthy"] is False
        assert report["status"] == "UNHEALTHY"
        assert any("Data directory not found" in issue for issue in report["issues"])

    def test_health_check_cli_quiet(self):
        exit_code = health_main(["--quiet"])
        assert exit_code == 0

    def test_health_check_cli_json(self, capsys):
        exit_code = health_main(["--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["healthy"] is True
        assert data["status"] == "HEALTHY"


class TestDeploymentArtifacts:
    def test_env_example_present(self):
        env_example = ROOT / ".env.example"
        assert env_example.exists()
        content = env_example.read_text(encoding="utf-8")
        assert "GROQ_API_KEY" in content
        assert "TALK_TO_DATA_DISABLE_LLM" in content
        assert "STREAMLIT_SERVER_PORT" in content

    def test_dockerfile_present(self):
        dockerfile = ROOT / "Dockerfile"
        assert dockerfile.exists()
        content = dockerfile.read_text(encoding="utf-8")
        assert "FROM python:3.10-slim" in content
        assert "HEALTHCHECK" in content
        assert "EXPOSE 8501" in content
        assert "ENTRYPOINT" in content

    def test_dockerignore_present(self):
        dockerignore = ROOT / ".dockerignore"
        assert dockerignore.exists()
        content = dockerignore.read_text(encoding="utf-8")
        assert "__pycache__" in content
        assert ".git" in content

    def test_deployment_docs_present(self):
        deploy_doc = ROOT / "docs" / "DEPLOYMENT.md"
        assert deploy_doc.exists()
        content = deploy_doc.read_text(encoding="utf-8")
        assert "Deployment Guide" in content
        assert "Docker Deployment" in content
        assert "Security, Compliance" in content


class TestCleanEnvironmentExecution:
    def test_e2e_chatbot_query_in_deterministic_mode(self, monkeypatch):
        # Force deterministic mode (no external network or LLM dependency)
        monkeypatch.setenv("TALK_TO_DATA_DISABLE_LLM", "1")
        
        dataset = load_dataset()
        bot = Chatbot(dataset=dataset)
        
        response = bot.ask("What is the total APE written by the portfolio?")
        assert response.outcome == ANSWERED
        assert response.problem_id == "PS-003"
        assert any(k in response.answer for k in ["2,306,202,662", "2,306,202,661", "2.31B", "2306202662"])
        assert len(response.tables) > 0
        assert response.evidence is not None
