"""本地平台 API。原始简历只随请求处理，不写日志或反馈库。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.formparsers import MultiPartParser
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from .corpus import Corpus
from .workflow import Workflow
from .samples import SAMPLES
from .research_store import ResearchStore, public_material
from .documents import extract_document
from .body_limit import BodyLimitMiddleware
from .profiles import profile_preview, compare_versions
from .domain import parse_profile, digest, display_text
from .retrieval_contract import stable_hash
from .journey import JourneyStore

ROOT = Path(__file__).resolve().parents[1]


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(default="", max_length=30)
    intent: str = Field(default="", max_length=150)
    education: Literal["", "初中", "高中", "中专", "大专", "本科", "硕士", "博士"] = ""
    education_full_time: bool | None = None
    experience_years: float | None = Field(default=None, ge=0, le=60)
    salary_min: float | None = Field(default=None, ge=0, le=1000000)
    salary_max: float | None = Field(default=None, ge=0, le=1000000)
    district: str = Field(default="", max_length=40)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="", max_length=20000)
    preferences: Preferences = Field(default_factory=Preferences)
    limit: int = Field(default=10, ge=1, le=20)
    method: Literal["bm25", "retrieval", "hybrid"] = "hybrid"
    strict_unknown: bool = False
    query_mode: Literal["intent","experience","structured"] = "structured"
    confirmation_token: str = Field(default="",max_length=200)
    field_origins: dict[str,Literal["sample","sample_stale","user_input","text_extracted","unknown"]] = Field(default_factory=dict)
    research_consent: bool = False


class ConfirmRequest(ResumeRequest):
    user_confirmed: bool = False
    acknowledged_conflicts: list[str] = Field(default_factory=list,max_length=10)


class TargetRequest(ResumeRequest):
    job_id: str = Field(min_length=1, max_length=64)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=32, max_length=32)
    job_id: str = Field(min_length=1, max_length=64)
    action: Literal["like", "skip"]


class CompanyPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1,max_length=64)
    exclude: bool = True


class ActionRequest(TargetRequest):
    group_id: str = Field(min_length=1,max_length=64)


class EvidenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(min_length=32,max_length=32)
    evidence: str = Field(min_length=15,max_length=3000)
    consent_to_store: bool = False


class EvidenceConfirmation(TargetRequest):
    action_id: str = Field(min_length=32,max_length=32)
    revised_text: str = Field(min_length=1,max_length=20000)
    user_confirmed: bool = False


class SkillAnnotation(BaseModel):
    model_config=ConfigDict(extra="forbid")
    skill:str=Field(min_length=1,max_length=80)
    quote:str=Field(min_length=1,max_length=120)
    occurrence:int=Field(default=0,ge=0,le=100,strict=True)


class RequirementAnnotation(BaseModel):
    model_config=ConfigDict(extra="forbid")
    logic:Literal["single","any","all","unknown"]
    modality:Literal["required","preferred","negated","unknown"]
    skills:list[SkillAnnotation]=Field(default_factory=list,max_length=50)
    children:list["RequirementAnnotation"]=Field(default_factory=list,max_length=30)


class ExtractionAnnotation(BaseModel):
    model_config=ConfigDict(extra="forbid")
    groups:list[RequirementAnnotation]=Field(max_length=30)


class AnnotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1,max_length=100)
    annotator: str = Field(min_length=2,max_length=40,pattern=r"^[\w\-\u4e00-\u9fff]+$")
    grade: int | None = Field(default=None,ge=0,le=3,strict=True)
    decision: Literal["", "正确", "错误", "不确定"] = ""
    notes: str = Field(default="",max_length=2000)
    extraction:ExtractionAnnotation | None=None
    independent:bool=False
    task_hash:str | None=Field(default=None,min_length=64,max_length=64)
    action:Literal["recommend","clarify","no_match"] | None=None
    no_match_evidence_ref:str | None=Field(default=None,max_length=200)


def create_app(source: Path | None = None, private_root: Path | None = None, corpus: Corpus | None = None) -> FastAPI:
    source = source or Path(os.environ.get("JOB_AGENT_DATA", str(ROOT / "data/job_data.xlsx"))).expanduser().resolve()
    private_root = private_root or Path(os.environ.get("JOB_AGENT_PRIVATE_ROOT", str(Path.home() / ".cache/job-agent"))).expanduser().resolve()
    if private_root.is_relative_to(ROOT):
        raise ValueError("运行缓存和反馈库必须放在仓库外，请调整 JOB_AGENT_PRIVATE_ROOT。")
    secret = b""

    @asynccontextmanager
    async def lifespan(app):
        nonlocal secret
        private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        secret_file = private_root / "session.key"
        try:
            with secret_file.open("xb") as stream:
                stream.write(secrets.token_bytes(32))
            secret_file.chmod(0o600)
        except FileExistsError:
            pass
        secret = secret_file.read_bytes()
        if len(secret) != 32:
            raise ValueError("会话签名文件格式不正确；请检查仓库外的私有目录。")
        db = private_root / "feedback.sqlite3"
        with sqlite3.connect(db) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS exposures (session TEXT, run_id TEXT, job_id TEXT, snapshot TEXT, position INTEGER, created REAL, PRIMARY KEY(session,run_id,job_id))")
            connection.execute("CREATE TABLE IF NOT EXISTS feedback (session TEXT, run_id TEXT, job_id TEXT, action TEXT CHECK(action IN ('like','skip')), created REAL, PRIMARY KEY(session,run_id,job_id))")
            connection.execute("DELETE FROM feedback WHERE created < ?", (time.time()-86400,))
            connection.execute("DELETE FROM exposures WHERE created < ?", (time.time()-86400,))
        app.state.db = db
        app.state.journey = JourneyStore(private_root)
        app.state.journey.prune()
        dense = os.environ.get("JOB_AGENT_DENSE_DIR")
        if dense and Path(dense).expanduser().resolve().is_relative_to(ROOT):
            raise ValueError("向量缓存必须位于仓库外。")
        app.state.research = ResearchStore(os.environ.get("JOB_AGENT_RESEARCH_DIR"),private_root)
        records=list(app.state.research.jobs.values()) if app.state.research.jobs else None
        app.state.corpus = corpus or Corpus(source, Path(dense) if dense else None,records=records)
        app.state.workflow = Workflow(app.state.corpus)
        yield

    app = FastAPI(title="职向 · 岗位需求与简历工作台", version="0.3.0", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    # 所有上传先受ASGI实际字节限制；阈值高于允许的整个请求，解析器不会暂存简历到磁盘。
    MultiPartParser.spool_max_size=2200001
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver", "[::1]"])

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "DELETE"} and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "不允许跨站修改请求"}, status_code=403)
        try:
            size = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return JSONResponse({"detail": "无效请求长度"}, status_code=400)
        maximum=2200000 if request.url.path=="/api/v2/document" else 150000
        if size < 0 or size > maximum:
            return JSONResponse({"detail": "请求过大"}, status_code=413)
        cookie = request.cookies.get("job_agent_session", "")
        session, _, signature = cookie.partition(".")
        expected = hmac.new(secret, session.encode(), hashlib.sha256).hexdigest()
        fresh = len(session) != 32 or not hmac.compare_digest(signature, expected)
        if fresh:
            session = secrets.token_hex(16)
            signature = hmac.new(secret, session.encode(), hashlib.sha256).hexdigest()
        request.state.session = session
        response = await call_next(request)
        if fresh:
            response.set_cookie("job_agent_session", session + "." + signature, httponly=True, samesite="strict", max_age=30*86400)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/v2/health")
    def health():
        return {"status": "ready", "jobs": len(app.state.corpus.jobs), "snapshot": app.state.corpus.snapshot[:16], "build_seconds": app.state.corpus.build_seconds}

    @app.get("/api/v2/overview")
    def overview(category: str = ""):
        if category and category not in app.state.corpus.categories:
            raise HTTPException(404, "未找到该职位类型")
        return app.state.corpus.overview(category)

    @app.get("/api/v2/samples")
    def samples():
        return {"samples": SAMPLES, "note": "全部为虚构样例，不代表真实用户或人工金标。"}

    def confirmation_digest(payload):
        prefs=payload.preferences.model_dump()
        if prefs["salary_min"] is not None and prefs["salary_max"] is not None and prefs["salary_max"] < prefs["salary_min"]:
            raise HTTPException(422,"期望薪资上限不能低于最低薪资")
        return stable_hash({"profile_version":parse_profile(payload.text,prefs)["version"],"preferences":prefs})

    def require_confirmation(payload, request):
        parts=payload.confirmation_token.split(".")
        if len(parts)!=3:raise HTTPException(409,"请先解析并确认当前画像")
        stamp,fingerprint,signature=parts
        if not stamp.isdigit() or not 0 <= time.time()-int(stamp) < 86400 or fingerprint!=confirmation_digest(payload):
            raise HTTPException(409,"简历、偏好或解析版本已变化，请重新确认画像")
        expected=hmac.new(secret,f"{request.state.session}:{stamp}:{fingerprint}".encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature,expected):raise HTTPException(409,"画像确认不属于当前会话")

    @app.post("/api/v2/profile/preview")
    def preview_profile(payload: ResumeRequest):
        confirmation_digest(payload)
        return profile_preview(payload.text,payload.preferences.model_dump(),payload.field_origins)

    @app.post("/api/v2/profile/confirm")
    def confirm_profile(payload: ConfirmRequest, request: Request):
        if not payload.user_confirmed:raise HTTPException(422,"请明确确认画像字段")
        preview=profile_preview(payload.text,payload.preferences.model_dump(),payload.field_origins)
        if any(source=="sample_stale" for source in payload.field_origins.values()):raise HTTPException(422,"请移除旧样例来源后重新确认")
        if set(preview["conflicts"])-set(payload.acknowledged_conflicts):raise HTTPException(422,"请核对冲突字段后明确采用当前表单值")
        if preview["preferences"] != payload.preferences.model_dump():
            raise HTTPException(409,"正文抽取出了补充字段，请核对预览中的有效画像后重新提交确认")
        fingerprint=confirmation_digest(payload);stamp=str(int(time.time()))
        signature=hmac.new(secret,f"{request.state.session}:{stamp}:{fingerprint}".encode(),hashlib.sha256).hexdigest()
        return {"confirmation_token":f"{stamp}.{fingerprint}.{signature}","profile_version":parse_profile(payload.text,payload.preferences.model_dump())["version"],
                "preferences":payload.preferences.model_dump(),"note":"已确认当前画像；后续编辑需要重新确认。"}

    @app.post("/api/v2/document")
    async def document(request:Request):
        try:
            async with request.form(max_files=1,max_fields=0,max_part_size=2200000) as form:
                file=form.get("file")
                if not isinstance(file,StarletteUploadFile):raise ValueError("请提供一个名为file的简历文件")
                content=await file.read(2*1024*1024+1)
                return await run_in_threadpool(extract_document,content,file.filename or "")
        except Exception as error:
            message=str(error) if isinstance(error,(ValueError,UnicodeError)) else "文件无法解析，请使用文字型PDF、DOCX或UTF-8 TXT"
            raise HTTPException(422,message[:200]) from None

    @app.get("/api/v2/research")
    def research_summary():
        return {**app.state.research.summary(),"experiments":app.state.research.reports()}

    @app.get("/api/v2/research/task")
    def annotation_task(annotator: str="评审甲",kind: str | None=None):
        if not 2<=len(annotator)<=40:raise HTTPException(422,"评审代号长度需为2至40")
        return app.state.research.task(annotator,kind)

    @app.post("/api/v2/research/annotation")
    def annotate(payload:AnnotationRequest):
        try:return app.state.research.annotate(**payload.model_dump())
        except (KeyError,ValueError) as error:raise HTTPException(422,str(error)) from None

    @app.get("/api/v2/graph/{job_id}")
    def graph(job_id:str):
        job=app.state.corpus.by_id.get(job_id)
        if job is None:raise HTTPException(404,"该岗位不属于当前快照")
        return public_material({"title":job.public()["title"],"job_family_id":job.job_family_id,"groups":job.groups,"requirement_ast":job.requirement_ast,
                "split":app.state.corpus.research_records.get(job_id,{}).get("split","未划分"),"parser_version":job.parser_version,
                "notice":"原文关系由当前解析器生成，保留要求组及未知状态；预测边不作为事实依据。"})

    @app.post("/api/v2/recommend")
    def recommend(payload: ResumeRequest, request: Request):
        require_confirmation(payload,request)
        exclusions=app.state.journey.exclusions(request.state.session)
        companies=[job.company for row in exclusions if (job:=app.state.corpus.by_id.get(row["job_id"]))]
        result = app.state.workflow.recommend(payload.text, payload.preferences.model_dump(), payload.limit, payload.method, payload.strict_unknown,payload.query_mode,companies)
        result["session_preferences"]={"excluded_companies":len(companies)}
        with sqlite3.connect(app.state.db) as connection:
            connection.execute("DELETE FROM feedback WHERE created < ?", (time.time()-86400,))
            connection.execute("DELETE FROM exposures WHERE created < ?", (time.time()-86400,))
            connection.executemany("INSERT INTO exposures VALUES (?,?,?,?,?,?)", [(request.state.session, result["run_id"], job["id"], result["snapshot"], i+1, time.time()) for i, job in enumerate(result["jobs"])])
        if payload.research_consent:
            app.state.journey.save_research_event(request.state.session,result["run_id"],{
                "profile_version":result["profile"]["version"],"snapshot":result["snapshot"],"feature_version":result.get("feature_version"),
                "retrieval_contract":result.get("retrieval_contract"),"encoder_contract":result.get("encoder_contract"),
                "exposures":[{"job_id":job["id"],"position":i+1,"features":job["features"]} for i,job in enumerate(result["jobs"])],
                "consent":True,"raw_resume_stored":False})
        return result

    def require_job(payload):
        if payload.job_id not in app.state.corpus.by_id:
            raise HTTPException(404, "岗位不存在或不属于当前快照")

    @app.post("/api/v2/diagnose")
    def diagnose(payload: TargetRequest, request: Request):
        require_confirmation(payload,request)
        require_job(payload)
        return app.state.workflow.diagnose(payload.text, payload.preferences.model_dump(), payload.job_id)

    @app.post("/api/v2/coach")
    def coach(payload:TargetRequest, request: Request):
        require_confirmation(payload,request)
        require_job(payload)
        diagnosis=app.state.workflow.diagnose(payload.text,payload.preferences.model_dump(),payload.job_id)
        return app.state.workflow.coach.diagnose(diagnosis)

    @app.post("/api/v2/rewrite")
    def rewrite(payload: TargetRequest, request: Request):
        require_confirmation(payload,request)
        require_job(payload)
        return app.state.workflow.rewrite(payload.text, payload.preferences.model_dump(), payload.job_id)

    @app.post("/api/v2/interview")
    def interview(payload: TargetRequest, request: Request):
        require_confirmation(payload,request)
        require_job(payload)
        return app.state.workflow.interview(payload.text, payload.preferences.model_dump(), payload.job_id)

    @app.post("/api/v2/compare")
    def compare(payload: ResumeRequest, request: Request):
        require_confirmation(payload,request)
        rows = []
        for method, label in [("bm25", "BM25基线"), ("retrieval", "多路召回"), ("hybrid", "多路召回＋证据精排")]:
            companies=[job.company for row in app.state.journey.exclusions(request.state.session) if (job:=app.state.corpus.by_id.get(row["job_id"]))]
            run = app.state.workflow.recommend(payload.text, payload.preferences.model_dump(), payload.limit, method, payload.strict_unknown,payload.query_mode,companies)
            jobs = run["jobs"]
            rows.append({"method": method, "label": label, "count": len(jobs), "latency_ms": run["latency_ms"],
                         "mean_skill_coverage": round(sum(job["skill_coverage"] for job in jobs)/len(jobs), 1) if jobs else None,
                         "company_count": len({job["company"] for job in jobs}),
                         "hard_violations": sum(any(check["status"] == "fail" for check in job["constraints"]) for job in jobs),
                         "job_ids": [job["id"] for job in jobs], "titles": [job["title"] for job in jobs[:3]]})
        return {"rows": rows, "snapshot": app.state.corpus.snapshot[:16], "note": "同一简历和快照的实际运行结果。技能覆盖为规则代理指标；无人工金标，不能据此声称nDCG或AUC提升。"}

    @app.post("/api/v2/feedback")
    def feedback(payload: FeedbackRequest, request: Request):
        with sqlite3.connect(app.state.db) as connection:
            row = connection.execute("SELECT 1 FROM exposures WHERE session=? AND run_id=? AND job_id=? AND created>=?", (request.state.session, payload.run_id, payload.job_id,time.time()-86400)).fetchone()
            if not row:
                raise HTTPException(404, "只能反馈本会话已展示的岗位")
            connection.execute("INSERT INTO feedback VALUES (?,?,?,?,?) ON CONFLICT(session,run_id,job_id) DO UPDATE SET action=excluded.action,created=excluded.created", (request.state.session, payload.run_id, payload.job_id, payload.action, time.time()))
        return {"saved": True, "note": "跳过作为偏好记录，不自动当作不相关负例。"}

    @app.get("/api/v2/preferences")
    def preferences_summary(request: Request):
        rows=app.state.journey.exclusions(request.state.session)
        return {"excluded_companies":[{"job_id":row["job_id"],"company":job.public()["company"]} for row in rows if (job:=app.state.corpus.by_id.get(row["job_id"]))],"retention_days":30}

    @app.post("/api/v2/preferences/company")
    def company_preference(payload: CompanyPreference, request: Request):
        job=app.state.corpus.by_id.get(payload.job_id)
        if job is None:raise HTTPException(404,"岗位不存在")
        if payload.exclude:
            with sqlite3.connect(app.state.db) as db:
                shown=db.execute("SELECT 1 FROM exposures WHERE session=? AND job_id=? AND created>=?",(request.state.session,payload.job_id,time.time()-86400)).fetchone()
            if not shown:raise HTTPException(404,"请从本会话已展示岗位设置偏好")
        app.state.journey.exclude(request.state.session,digest(job.company),payload.job_id,payload.exclude)
        return {"saved":True,"exclude":payload.exclude,"company":job.public()["company"],"note":"下次推荐会使用此偏好，可以随时撤销。"}

    @app.get("/api/v2/journey")
    def journey(request: Request):
        app.state.journey.prune()
        return app.state.journey.history(request.state.session)

    @app.post("/api/v2/journey/actions")
    def start_action(payload: ActionRequest, request: Request):
        require_confirmation(payload,request);require_job(payload)
        diagnosis=app.state.workflow.diagnose(payload.text,payload.preferences.model_dump(),payload.job_id)
        gap=next((gap for gap in diagnosis["gaps"] if gap.get("group_id")==payload.group_id),None)
        if gap is None:raise HTTPException(409,"该缺口已变化，请刷新诊断后创建行动")
        return app.state.journey.start(request.state.session,payload.job_id,payload.group_id,diagnosis["profile_version"],gap["skill"])

    @app.post("/api/v2/journey/evidence")
    def save_evidence(payload: EvidenceRequest, request: Request):
        if not payload.consent_to_store:raise HTTPException(422,"请明确同意在本机保存这条证据")
        try:return app.state.journey.save_evidence(request.state.session,payload.action_id,payload.evidence.strip())
        except KeyError as error:raise HTTPException(404,str(error)) from None
        except ValueError as error:raise HTTPException(409,str(error)) from None

    @app.post("/api/v2/journey/confirm-evidence")
    def confirm_evidence(payload: EvidenceConfirmation, request: Request):
        require_confirmation(payload,request);require_job(payload)
        if not payload.user_confirmed:raise HTTPException(422,"请核对新增表述确实来自自己的经历")
        try:
            action=app.state.journey.action(request.state.session,payload.action_id)
            if action["job_id"]!=payload.job_id or action["profile_version"]!=parse_profile(payload.text,payload.preferences.model_dump())["version"]:
                raise ValueError("行动不属于当前岗位或画像版本，请重新诊断")
            if not action["evidence"] or action["evidence"] not in payload.revised_text:
                raise ValueError("新简历需要包含已保存的原文证据")
            # 补证入口只允许追加当前确认的单条原文；一般改写使用独立的画像确认流程。
            expected=payload.text.rstrip()+"\n\n"+action["evidence"]
            if payload.revised_text!=expected:raise ValueError("补证确认仅接受原简历加这条证据，不能同时改动其他经历")
            diff=compare_versions(payload.text,payload.revised_text,payload.preferences.model_dump())
            previous=app.state.workflow.diagnose(payload.text,payload.preferences.model_dump(),payload.job_id)
            current=app.state.workflow.diagnose(payload.revised_text,payload.preferences.model_dump(),payload.job_id)
            current_ids={gap.get("group_id") for gap in current["gaps"]}
            closed=[gap["skill"] for gap in previous["gaps"] if gap.get("group_id") not in current_ids]
            summary={key:diff[key] for key in ["before_version","after_version","changed_fields","added_skill_statements"]}
            summary.update(added_blocks=len(diff["added_blocks"]),removed_blocks=len(diff["removed_blocks"]),closed_gaps=closed,
                           remaining_gaps=[gap["skill"] for gap in current["gaps"]],confirmation="user_confirmed_not_externally_verified")
            app.state.journey.confirm(request.state.session,payload.action_id,summary,expected_evidence=action['evidence'])
            return {"revised_text":payload.revised_text,"diff":diff,"closed_gaps":closed,"remaining_gaps":current["gaps"],
                    "note":"已记录用户确认的证据。请核对新画像后再次匹配；没有外部验证能力真实性。"}
        except (KeyError,ValueError) as error:raise HTTPException(422,str(error)) from None

    @app.delete("/api/v2/journey")
    def delete_journey(request: Request):
        app.state.journey.clear(request.state.session)
        return {"deleted":True}

    @app.get("/api/v2/research-events")
    def export_research_events(request: Request):
        with app.state.journey.connect() as db:
            rows=[json.loads(row["payload"]) for row in db.execute("SELECT payload FROM research_events WHERE session=? AND created>=? ORDER BY created",(request.state.session,time.time()-30*86400))]
        return {"events":rows,"note":"仅含你明确同意留存的特征与曝光上下文；不含完整简历，不自动生成训练标签。"}

    @app.get("/api/v2/feedback")
    def feedback_summary(request: Request):
        with sqlite3.connect(app.state.db) as connection:
            rows = connection.execute("SELECT action,COUNT(*) FROM feedback WHERE session=? AND created>=? GROUP BY action", (request.state.session,time.time()-86400)).fetchall()
        return {"counts": dict(rows)}

    @app.delete("/api/v2/feedback")
    def delete_feedback(request: Request):
        with sqlite3.connect(app.state.db) as connection:
            connection.execute("DELETE FROM feedback WHERE session=?", (request.state.session,))
            connection.execute("DELETE FROM exposures WHERE session=?", (request.state.session,))
        app.state.journey.clear(request.state.session,preferences_only=True)
        return {"deleted": True}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/platform/index.html")

    app.mount("/assets", StaticFiles(directory=ROOT / "web/platform"), name="assets")
    return app
