"""确认新决策层实际进入API和画像闭环，旧特征契约仍可复用。"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook

from job_agent.api import create_app
from job_agent.corpus import Corpus
from job_agent.domain import parse_profile
from job_agent.decision_support import evidence_bounds, plan_questions
from job_agent.evidence_alignment import align_requirements, LIMITS
from job_agent.workflow import Workflow


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "岗位.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.append(["岗位名称", "企业", "岗位薪资", "岗位要求", "岗位职责", "岗位地址", "职位类型名称", "二级分类"])
    sheet.append(["Python数据工程师", "样例企业甲", "15-20K", "全日制本科，3年工作经验", "使用Python完成数据清洗。", "深圳南山区", "Python", "技术"])
    sheet.append(["Python开发工程师", "样例企业乙", "10-12K", "本科，经验不限", "熟悉Python或Java。", "深圳福田区", "Python", "技术"])
    book.save(path)
    return Corpus(path)


TEXT = "项目一：我使用Python开发订单接口。\n\n项目二：我使用Excel完成数据清洗。"
PREFS = {"city": "深圳", "intent": "Python", "education": "本科", "experience_years": 3}


def confirmed(client, text=TEXT, preferences=None):
    payload = {"text": text, "preferences": preferences or dict(PREFS)}
    preview = client.post("/api/v2/profile/preview", json=payload).json()
    payload["preferences"] = preview["preferences"]
    response = client.post("/api/v2/profile/confirm", json={**payload, "user_confirmed": True, "acknowledged_conflicts": preview["conflicts"]})
    assert response.status_code == 200
    payload["confirmation_token"] = response.json()["confirmation_token"]
    return payload


def test_recommend_and_diagnose_expose_task_binding_and_action_plan(corpus, tmp_path):
    with TestClient(create_app(corpus=corpus, private_root=tmp_path / "private")) as client:
        payload = confirmed(client)
        result = client.post("/api/v2/recommend", json=payload)
        assert result.status_code == 200
        data = result.json()
        assert data["feature_version"] == "evidence-ranker-v2-35"
        assert data["comparison_frontier"]["points"]
        target = next(row for row in data["jobs"] if row["company"] == "样例企业甲")
        assert len(target["features"]) == 35
        assert target["decision_certificate"]["coverage_bounds"] == [0, 1]
        assert target["alignment_summary"]["downgraded_groups"] > 0
        assert all(row["status"] == "unknown" for row in target["matched"])
        detail = client.post("/api/v2/diagnose", json={**payload, "job_id": target["id"]})
        assert detail.status_code == 200
        diagnosis = detail.json()
        assert diagnosis["mention_coverage"] == 100 and diagnosis["coverage"] == 0
        assert any(action["kind"] == "task_binding" for plan in diagnosis["action_plans"]["plans"] for action in plan["actions"])


def test_explicit_full_time_answer_changes_filter_and_requires_confirmation(corpus, tmp_path):
    with TestClient(create_app(corpus=corpus, private_root=tmp_path / "private")) as client:
        payload = confirmed(client)
        result = client.post("/api/v2/recommend", json=payload).json()
        assert any(q["field"] == "education_full_time" for q in result["decision_support"]["questions"])
        edited = {**payload, "preferences": {**payload["preferences"], "education_full_time": False}}
        assert client.post("/api/v2/recommend", json=edited).status_code == 409
        updated = confirmed(client, preferences=edited["preferences"])
        result = client.post("/api/v2/recommend", json=updated).json()
        assert all(row["company"] != "样例企业甲" for row in result["jobs"])


def test_task_relationship_question_is_not_repeated_skill_question(corpus):
    profile = parse_profile(TEXT, {**PREFS, "education_full_time": True})
    target = corpus.jobs[0]
    output = plan_questions(profile, [target], alignments={target.id: align_requirements(profile, target)})
    assert output["questions"] and output["questions"][0]["kind"] == "task_binding"
    assert "同一个任务" in output["questions"][0]["question"]


def test_alignment_budget_cannot_fall_back_to_supported_certificate(corpus):
    profile = parse_profile("使用Python完成数据清洗。" + "\n" + "字" * (LIMITS["block_chars"] + 1), PREFS)
    target = corpus.jobs[0]
    alignment = align_requirements(profile, target)
    assert alignment["summary"]["status"] == "not_evaluated"
    assert evidence_bounds(profile, target, alignment)["coverage_bounds"] == [0, 1]


def test_decision_layer_does_not_mutate_features_or_resume(corpus):
    workflow = Workflow(corpus)
    original = workflow.replay_features(TEXT, PREFS, [job.id for job in corpus.jobs])
    result = workflow.recommend(TEXT, PREFS)
    replay = workflow.replay_features(TEXT, PREFS, [job.id for job in corpus.jobs])
    assert original == replay
    assert result["profile"]["claim_note"]
    assert all(not row["alignment_summary"]["ability_verified"] for row in result["jobs"])


def test_full_time_field_is_present_in_browser_and_confirmation_sources():
    from job_agent.profiles import profile_preview
    output = profile_preview(TEXT, {**PREFS, "education_full_time": False})
    field = next(row for row in output["fields"] if row["key"] == "education_full_time")
    assert field["value"] is False and field["source"] == "user_input"
    root = Path(__file__).resolve().parents[1]
    assert 'id="education-full-time"' in (root / "web/platform/index.html").read_text()
