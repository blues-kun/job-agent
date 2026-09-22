#!/usr/bin/env python3
"""将已验清单的岗位与图关系导出为仓库外只增不覆盖的 SQLite 目录。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

SCHEMA = 'job-agent-private-catalog-v1'
SOURCE_SCHEMA = 'job-agent-research-data-v1'
CITIES = ('深圳', '广州', '北京', '上海', '杭州', '东莞', '惠州')
SOURCE_ID = re.compile(r'^sha256:([0-9a-f]{64})/sheet:(\d+)/row:(\d+)$')
FIELDS = ('title', 'company', 'salary_raw', 'requirements', 'description', 'address', 'category', 'category_parent')

DDL = '''
CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL CHECK(json_valid(value_json)));
CREATE TABLE job_families (
 job_family_id TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train','dev','test')),
 UNIQUE(job_family_id,split)
);
CREATE TABLE jobs (
 job_id TEXT PRIMARY KEY, job_version_id TEXT NOT NULL CHECK(job_version_id=job_id),
 legacy_id TEXT NOT NULL, snapshot TEXT NOT NULL, content_sha256 TEXT NOT NULL,
 title TEXT NOT NULL, company TEXT NOT NULL, company_hash TEXT NOT NULL,
 category TEXT NOT NULL, category_parent TEXT NOT NULL,
 requirements TEXT NOT NULL, description TEXT NOT NULL, address TEXT NOT NULL,
 salary_raw TEXT NOT NULL, salary_status TEXT NOT NULL,
 salary_monthly_low REAL, salary_monthly_high REAL, salary_annual_low REAL, salary_annual_high REAL,
 city TEXT, district TEXT, experience_min REAL, education_min INTEGER,
 job_family_id TEXT NOT NULL, split TEXT NOT NULL CHECK(split IN ('train','dev','test')),
 salary_json TEXT NOT NULL CHECK(json_valid(salary_json)),
 skills_json TEXT NOT NULL CHECK(json_valid(skills_json)),
 source_fields_json TEXT NOT NULL CHECK(json_valid(source_fields_json)),
 field_sources_json TEXT NOT NULL CHECK(json_valid(field_sources_json)),
 quality_flags_json TEXT NOT NULL CHECK(json_valid(quality_flags_json)),
 UNIQUE(job_id,split), FOREIGN KEY(job_family_id,split) REFERENCES job_families(job_family_id,split),
 CHECK(salary_monthly_low IS NULL OR (salary_status='月薪' AND salary_monthly_low>=0)),
 CHECK(salary_monthly_high IS NULL OR (salary_status='月薪' AND salary_monthly_high>=salary_monthly_low))
);
CREATE TABLE source_lineage (
 source_record_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
 snapshot TEXT NOT NULL, sheet_number INTEGER NOT NULL CHECK(sheet_number>0),
 row_number INTEGER NOT NULL CHECK(row_number>0)
);
CREATE TABLE requirement_groups (
 group_id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id), ordinal INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('single','any','all')), preferred INTEGER NOT NULL CHECK(preferred IN (0,1)),
 label TEXT NOT NULL, skills_json TEXT NOT NULL CHECK(json_valid(skills_json)),
 evidence_json TEXT NOT NULL CHECK(json_valid(evidence_json)),
 UNIQUE(job_id,ordinal)
);
CREATE TABLE graph_edges (
 edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
 source_type TEXT NOT NULL, target_type TEXT NOT NULL, relation TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('observed','machine_parsed','hypothesis')),
 split TEXT NOT NULL CHECK(split IN ('train','dev','test')),
 job_id TEXT, fact_citable INTEGER NOT NULL CHECK(fact_citable IN (0,1)),
 evidence_json TEXT CHECK(evidence_json IS NULL OR json_valid(evidence_json)),
 properties_json TEXT NOT NULL CHECK(json_valid(properties_json)),
 FOREIGN KEY(job_id,split) REFERENCES jobs(job_id,split),
 CHECK((status='hypothesis' AND fact_citable=0 AND evidence_json IS NULL AND job_id IS NULL
        AND split='train' AND relation='cooccurs_hypothesis')
    OR (status IN ('observed','machine_parsed') AND evidence_json IS NOT NULL AND job_id IS NOT NULL
        AND relation!='cooccurs_hypothesis'))
);
CREATE VIEW citable_evidence AS SELECT * FROM graph_edges
 WHERE status IN ('observed','machine_parsed') AND fact_citable=1 AND evidence_json IS NOT NULL;
CREATE VIEW statistical_hypotheses AS SELECT * FROM graph_edges WHERE status='hypothesis';
CREATE INDEX jobs_family_idx ON jobs(job_family_id);
CREATE INDEX jobs_split_idx ON jobs(split);
CREATE INDEX jobs_city_idx ON jobs(city);
CREATE INDEX jobs_district_idx ON jobs(district);
CREATE INDEX jobs_salary_low_idx ON jobs(salary_monthly_low);
CREATE INDEX jobs_salary_high_idx ON jobs(salary_monthly_high);
CREATE INDEX jobs_experience_idx ON jobs(experience_min);
CREATE INDEX jobs_education_idx ON jobs(education_min);
CREATE INDEX jobs_category_idx ON jobs(category);
CREATE INDEX jobs_filter_idx ON jobs(split,city,category,salary_monthly_high);
CREATE INDEX source_lineage_job_idx ON source_lineage(job_id);
CREATE INDEX requirement_groups_job_idx ON requirement_groups(job_id);
CREATE INDEX graph_edges_source_idx ON graph_edges(source_id,status,split);
CREATE INDEX graph_edges_target_idx ON graph_edges(target_id,status,split);
CREATE INDEX graph_edges_job_idx ON graph_edges(job_id);
CREATE INDEX graph_edges_relation_idx ON graph_edges(relation,status,split);
'''


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def checked_evidence(job, evidence):
    if not isinstance(evidence, dict):
        raise ValueError('事实或机器解析边必须包含原文证据。')
    field = evidence.get('field'); start = evidence.get('start'); end = evidence.get('end')
    if field not in FIELDS or type(start) is not int or type(end) is not int:
        raise ValueError('引用字段或跨度类型无效。')
    if not 0 <= start < end <= len(job[field]) or job[field][start:end] != evidence.get('quote'):
        raise ValueError('引用跨度与真实岗位原文不一致。')
    if evidence.get('job_id', job['job_id']) != job['job_id']:
        raise ValueError('引用指向其他岗位。')
    if 'source_record_ids' in evidence and evidence['source_record_ids'] != job['source_record_ids']:
        raise ValueError('引用的原记录血缘不一致。')


def insert_mapping(connection, table, value):
    names = tuple(value)
    connection.execute(f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", tuple(value[name] for name in names))


def export_catalog(dataset: Path, output: Path, repository: Path):
    repository = repository.expanduser().resolve(); dataset = dataset.expanduser().resolve()
    output = output.expanduser()
    if output.is_symlink() or output.exists():
        raise FileExistsError('目录数据库已存在，拒绝覆盖；新版本请使用新文件名。')
    output = output.resolve()
    if dataset.is_relative_to(repository) or output.is_relative_to(repository):
        raise ValueError('研究输入与目录数据库必须位于仓库外。')
    manifest_path = dataset/'manifest.json'
    source_paths = {'jobs.jsonl': dataset/'jobs.jsonl', 'edges.jsonl': dataset/'edges.jsonl'}
    if manifest_path.is_symlink() or any(path.is_symlink() for path in source_paths.values()):
        raise ValueError('数据清单和数据文件不能是符号链接。')
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if manifest.get('schema_version') != SOURCE_SCHEMA:
        raise ValueError('研究数据版本不兼容。')
    hashes = {}
    for name, path in source_paths.items():
        expected = manifest.get('files', {}).get(name, {}).get('sha256')
        hashes[name] = file_hash(path)
        if not expected or hashes[name] != expected:
            raise ValueError('岗位或关系文件不符合版本清单。')
    snapshot = manifest.get('config', {}).get('snapshot')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest.get('source_sha256', '')) or snapshot != manifest['source_sha256']:
        raise ValueError('快照和原文件哈希不一致。')
    if not manifest.get('config_hash'):
        raise ValueError('研究数据缺少配置版本。')
    sys.path.insert(0, str(repository))
    from job_agent.domain import DISTRICTS, education, experience

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(prefix='.catalog-', suffix='.sqlite3', dir=output.parent)
    os.close(fd); temporary = Path(temporary_name)
    connection = None; by_id = {}; families = {}; group_ids = set(); statuses = Counter()
    try:
        connection = sqlite3.connect(temporary)
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('PRAGMA synchronous=FULL')
        connection.executescript(DDL)
        connection.execute('BEGIN IMMEDIATE')
        for job in rows(source_paths['jobs.jsonl']):
            if job['job_id'] in by_id or job['job_id'] != job['job_version_id'] or job['snapshot'] != snapshot:
                raise ValueError('岗位版本重复或不属于当前快照。')
            family = job['job_family_id']; split = job['split']
            if family in families and families[family] != split:
                raise ValueError('岗位家族跨数据分区。')
            if family not in families:
                connection.execute('INSERT INTO job_families VALUES (?,?)', (family, split)); families[family] = split
            salary = job['salary']; monthly = salary.get('status') == '月薪'
            value = {key: job[key] for key in ('job_id','job_version_id','legacy_id','snapshot','content_sha256','title','company','company_hash','category','category_parent','requirements','description','address','salary_raw','job_family_id','split')}
            value.update(salary_status=salary.get('status','未知'),
                         salary_monthly_low=salary.get('monthly_low') if monthly else None,
                         salary_monthly_high=salary.get('monthly_high') if monthly else None,
                         salary_annual_low=salary.get('annual_low'), salary_annual_high=salary.get('annual_high'),
                         city=next((name for name in CITIES if name in job['address']),None),
                         district=next((name for name in DISTRICTS if name in job['address']),None),
                         experience_min=experience(job['requirements'],True), education_min=education(job['requirements'],True),
                         salary_json=canonical(salary), skills_json=canonical(job['skills']),
                         source_fields_json=canonical(job['source_fields']), field_sources_json=canonical(job['field_sources']),
                         quality_flags_json=canonical(job['quality_flags']))
            insert_mapping(connection, 'jobs', value)
            if not job['source_record_ids']:
                raise ValueError('岗位缺少原记录血缘。')
            for record_id in job['source_record_ids']:
                match = SOURCE_ID.fullmatch(record_id)
                if not match or match[1] != snapshot:
                    raise ValueError('原记录标识与快照不一致。')
                connection.execute('INSERT INTO source_lineage VALUES (?,?,?,?,?)', (record_id,job['job_id'],snapshot,int(match[2]),int(match[3])))
            for ordinal, group in enumerate(job['groups']):
                citation = group['evidence']; checked_evidence(job, citation)
                group_id = 'requirement_group:'+digest([job['job_id'],ordinal,group['kind'],group['skills'],citation['field'],citation['start'],citation['end']])[:20]
                if type(group['preferred']) is not bool:
                    raise ValueError('优先项标记必须为布尔值。')
                connection.execute('INSERT INTO requirement_groups VALUES (?,?,?,?,?,?,?,?)', (group_id,job['job_id'],ordinal,group['kind'],int(group['preferred']),group['label'],canonical(group['skills']),canonical(citation)))
                group_ids.add(group_id)
            by_id[job['job_id']] = job
        for edge in rows(source_paths['edges.jsonl']):
            if type(edge['fact_citable']) is not bool:
                raise ValueError('关系可引用性必须为布尔值。')
            if edge['status'] == 'hypothesis':
                if edge['fact_citable'] or edge['evidence'] is not None or edge['job_id'] is not None or edge['split'] != 'train' or edge['relation'] != 'cooccurs_hypothesis' or edge.get('derived_from_split') != 'train':
                    raise ValueError('统计假设不可混入事实引用或使用非训练分区。')
            else:
                if edge['job_id'] not in by_id or edge['split'] != by_id[edge['job_id']]['split']:
                    raise ValueError('原文关系没有同分区岗位来源。')
                checked_evidence(by_id[edge['job_id']], edge['evidence'])
            for side in ('source','target'):
                if edge[side+'_type'] == 'requirement_group' and edge[side+'_id'] not in group_ids:
                    raise ValueError('需求关系引用未知逻辑组。')
            names = ('edge_id','source_id','target_id','source_type','target_type','relation','status','split','job_id')
            value = {key:edge[key] for key in names}
            value.update(fact_citable=int(edge['fact_citable']), evidence_json=canonical(edge['evidence']) if edge['evidence'] is not None else None,
                         properties_json=canonical({key:item for key,item in edge.items() if key not in {*names,'fact_citable','evidence'}}))
            insert_mapping(connection,'graph_edges',value); statuses[edge['status']] += 1
        if not by_id:
            raise ValueError('岗位目录为空。')
        counts = {table:connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in ('jobs','job_families','source_lineage','requirement_groups','graph_edges','citable_evidence','statistical_hypotheses')}
        if connection.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('目录外键检查失败。')
        for name,path in source_paths.items():
            if file_hash(path) != hashes[name]:
                raise ValueError('导出期间研究输入发生变化。')
        if file_hash(manifest_path) != manifest_hash:
            raise ValueError('导出期间研究清单发生变化。')
        metadata = {'schema_version':SCHEMA,'source_schema_version':SOURCE_SCHEMA,'source_manifest_sha256':manifest_hash,
                    'dataset_config_hash':manifest['config_hash'],'snapshot':snapshot,'source_sha256':manifest['source_sha256'],
                    'jobs_sha256':hashes['jobs.jsonl'],'edges_sha256':hashes['edges.jsonl'],'source_unchanged':True,
                    'counts':counts,'edge_status_counts':dict(statuses),'created_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    'export_code_sha256':file_hash(Path(__file__)), 'derived_domain_sha256':file_hash(repository/'job_agent/domain.py'),
                    'derived_profile_rules_sha256':file_hash(repository/'scripts/profile_data.py'),
                    'derived_fields_note':'地点、经验和学历沿用当前规则；缺失保留NULL，机器解析不等于人工确认。',
                    'citation_note':'citable_evidence保留observed/machine_parsed区分；它只证明原文依据，不代表需求语义已经人工确认。hypothesis永不可事实引用。',
                    'parquet_exported':False,'parquet_note':'本次仅物化SQLite；运行环境未提供pyarrow。' if importlib.util.find_spec('pyarrow') is None else '本次SQLite导出不包含Parquet。',
                    'runtime_note':'平台仍从JSONL加载；此数据库为独立研究数据层产物，尚未接入在线SQL检索。'}
        connection.executemany('INSERT INTO metadata VALUES (?,?)',[(key,canonical(value)) for key,value in metadata.items()])
        connection.commit()
        if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite完整性检查失败。')
        connection.close(); connection = None
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        # 同目录硬链接创建是原子且不覆盖的；已有目标（包括并发创建）使操作失败。
        os.link(temporary,output); temporary.unlink()
        return {'path':str(output),'sha256':file_hash(output),'metadata':metadata}
    finally:
        if connection is not None:
            connection.rollback(); connection.close()
        if temporary.exists():
            temporary.unlink()
        journal = Path(str(temporary)+'-journal')
        if journal.exists():
            journal.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export_catalog(args.dataset,args.output,args.repo_root),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
