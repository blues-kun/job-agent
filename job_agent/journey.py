"""会话偏好与补证记录；仅在用户明确保存时留存单条证据，不保存完整简历。"""
import json
import sqlite3
import time
import uuid
from pathlib import Path


class JourneyStore:
    def __init__(self, directory):
        self.path=Path(directory)/"journey.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS preferences (session TEXT,company_key TEXT,job_id TEXT,created REAL,PRIMARY KEY(session,company_key))")
            db.execute("CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY,session TEXT,job_id TEXT,group_id TEXT,profile_version TEXT,label TEXT,status TEXT,evidence TEXT,created REAL,updated REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS actions_session ON actions(session,updated)")
            db.execute("CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY,session TEXT,action_id TEXT,job_id TEXT,summary TEXT,created REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS versions_session ON versions(session,created)")
            db.execute("CREATE TABLE IF NOT EXISTS research_events (session TEXT,run_id TEXT,payload TEXT,created REAL,PRIMARY KEY(session,run_id))")
        self.path.chmod(0o600)

    def connect(self):
        db=sqlite3.connect(self.path,timeout=20)
        db.row_factory=sqlite3.Row
        return db

    def prune(self):
        with self.connect() as db:
            cutoff=time.time()-30*86400
            for table,field in [("preferences","created"),("actions","updated"),("versions","created"),("research_events","created")]:
                db.execute(f"DELETE FROM {table} WHERE {field} < ?",(cutoff,))

    def exclusions(self, session):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT company_key,job_id FROM preferences WHERE session=? AND created>=?",(session,time.time()-30*86400))]

    def exclude(self, session, company_key, job_id, enabled=True):
        with self.connect() as db:
            if enabled:
                db.execute("INSERT INTO preferences VALUES (?,?,?,?) ON CONFLICT(session,company_key) DO UPDATE SET job_id=excluded.job_id,created=excluded.created",(session,company_key,job_id,time.time()))
            else:
                db.execute("DELETE FROM preferences WHERE session=? AND company_key=?",(session,company_key))

    def action(self, session, identifier):
        with self.connect() as db:
            row=db.execute("SELECT * FROM actions WHERE id=? AND session=? AND updated>=?",(identifier,session,time.time()-30*86400)).fetchone()
        if row is None:raise KeyError("补证行动不存在或不属于本会话")
        return dict(row)

    def start(self, session, job_id, group_id, profile_version, label):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            found=db.execute("SELECT * FROM actions WHERE session=? AND job_id=? AND group_id=? AND profile_version=? AND status!='confirmed' AND updated>=?",(session,job_id,group_id,profile_version,time.time()-30*86400)).fetchone()
            if found:return dict(found)
            identifier=uuid.uuid4().hex;now=time.time()
            db.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?)",(identifier,session,job_id,group_id,profile_version,label,"pending","",now,now))
        return self.action(session,identifier)

    def save_evidence(self, session, identifier, text):
        self.action(session,identifier)
        with self.connect() as db:
            changed=db.execute("UPDATE actions SET evidence=?,status='evidence_submitted',updated=? WHERE id=? AND session=? AND status IN ('pending','evidence_submitted') AND updated>=?",(text,time.time(),identifier,session,time.time()-30*86400))
            if changed.rowcount!=1:raise ValueError("行动已确认或过期，请新建行动；历史证据不可覆盖")
        return self.action(session,identifier)

    def confirm(self, session, identifier, summary, *, expected_evidence):
        action=self.action(session,identifier)
        if action["status"]!="evidence_submitted":raise ValueError("先保存单条证据，再核对确认")
        if action['evidence']!=expected_evidence:raise ValueError("证据在核对后已变化，请刷新后重新确认")
        with self.connect() as db:
            changed=db.execute("UPDATE actions SET status='confirmed',updated=? WHERE id=? AND session=? AND status='evidence_submitted' AND evidence=? AND updated>=?",(time.time(),identifier,session,expected_evidence,time.time()-30*86400))
            if changed.rowcount!=1:raise ValueError("证据状态已变化，请刷新后核对")
            db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)",(uuid.uuid4().hex,session,identifier,action["job_id"],json.dumps(summary,ensure_ascii=False),time.time()))

    def history(self, session):
        with self.connect() as db:
            actions=[dict(row) for row in db.execute("SELECT id,job_id,group_id,profile_version,label,status,evidence,created,updated FROM actions WHERE session=? ORDER BY updated DESC LIMIT 100",(session,))]
            versions=[{**dict(row),"summary":json.loads(row["summary"])} for row in db.execute("SELECT id,action_id,job_id,summary,created FROM versions WHERE session=? ORDER BY created DESC LIMIT 100",(session,))]
        return {"actions":actions,"versions":versions,"retention_days":30,"note":"仅保存你明确提交的单条证据与版本摘要；完整简历不入库。"}

    def save_research_event(self, session, run_id, payload):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO research_events VALUES (?,?,?,?)",(session,run_id,json.dumps(payload,ensure_ascii=False,allow_nan=False),time.time()))

    def clear(self, session, preferences_only=False):
        with self.connect() as db:
            for table in (["preferences"] if preferences_only else ["preferences","actions","versions","research_events"]):
                db.execute(f"DELETE FROM {table} WHERE session=?",(session,))
