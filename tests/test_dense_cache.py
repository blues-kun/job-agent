"""验证索引落盘重载与编码器换版，使用内存HTTP桩避免调用外部模型。"""
from copy import deepcopy

import httpx
import numpy as np
import pytest

import job_agent.dense as dense


def fake_encoder(monkeypatch):
    state = {"version": "a"*64, "encode_calls": 0}

    def request(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"encoder_contract_hash": state["version"], "dimension": 2,
                                            "normalized": True, "device": "fixture"})
        state["encode_calls"] += 1
        return httpx.Response(200, json={"encoder_contract_hash": state["version"], "vectors": [[1.0, 0.0]]})
    client_type = httpx.Client
    monkeypatch.setattr(dense.httpx, "Client", lambda **kw: client_type(transport=httpx.MockTransport(request), **kw))
    # 使用实际合同内的元组形状，复现JSON落盘后列表不等的问题。
    monkeypatch.setattr(dense, "contract_info", lambda: {"query_modes": ("intent", "experience", "structured"), "hash": "fixture"})
    monkeypatch.setenv("JOB_AGENT_EMBEDDING_URL", "http://127.0.0.1:8093")
    return state


def test_restart_uses_valid_persisted_index_without_reencoding(tmp_path, monkeypatch):
    state = fake_encoder(monkeypatch)
    first = dense.DenseIndex(tmp_path, "snapshot", ["fiction-job"], ["Python练习岗位"])
    expected = deepcopy(first.matrix)
    first.client.close()
    second = dense.DenseIndex(tmp_path, "snapshot", ["fiction-job"], ["Python练习岗位"])
    assert state["encode_calls"] == 1
    np.testing.assert_array_equal(expected, second.matrix)
    second.client.close()


def test_changed_encoder_rebuilds_index_and_rejects_old_query_contract(tmp_path, monkeypatch):
    state = fake_encoder(monkeypatch)
    old = dense.DenseIndex(tmp_path, "snapshot", ["fiction-job"], ["Python练习岗位"])
    state["version"] = "b"*64
    new = dense.DenseIndex(tmp_path, "snapshot", ["fiction-job"], ["Python练习岗位"])
    assert state["encode_calls"] == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2
    with pytest.raises(ValueError, match="模型版本"):
        old.score("虚构简历")
    old.client.close()
    new.client.close()
