"""只用临时虚构数据，验证版本隔离、血缘条数及假设关系不可混作事实。"""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research import export_catalog as catalog

REPOSITORY=Path(__file__).resolve().parents[1]
SNAPSHOT='a'*64


def fixture(directory):
    directory.mkdir();jobs=[]
    for ordinal,split in enumerate(('train','dev')):
        job_id=str(ordinal)*20
        text='熟悉Python或Java'
        record_id=f'sha256:{SNAPSHOT}/sheet:1/row:{ordinal+2}'
        evidence={'field':'description','start':2,'end':8,'quote':'Python','job_id':job_id,'source_record_ids':[record_id]}
        group={'kind':'any','preferred':False,'skills':['Python','Java'],'label':'Python或Java','evidence':{'field':'description','start':2,'end':13,'quote':'Python或Java'}}
        jobs.append({'job_id':job_id,'job_version_id':job_id,'legacy_id':'l'+job_id,'snapshot':SNAPSHOT,
                     'content_sha256':str(ordinal)*64,'title':'虚构技术岗位','company':'虚构企业','company_hash':'c'*64,
                     'category':'Python','category_parent':'后端开发','requirements':'1-3年本科','description':text,
                     'address':'深圳南山区','salary_raw':'10-20K' if ordinal==0 else '20-30万/年',
                     'salary':{'status':'月薪','monthly_low':10000,'monthly_high':20000} if ordinal==0 else {'status':'年薪','annual_low':200000,'annual_high':300000},
                     'job_family_id':'family'+str(ordinal),'split':split,'source_record_ids':[record_id],
                     'groups':[group] if ordinal==0 else [],'skills':{'Python':evidence},'source_fields':{'岗位名称':'虚构技术岗位'},
                     'field_sources':{'title':'岗位名称'},'quality_flags':[]})
    edges=[]
    for ordinal,job in enumerate(jobs):
        edges.append({'edge_id':'e'+str(ordinal),'source_id':job['job_id'],'target_id':'skill:Python','source_type':'job','target_type':'skill',
                      'relation':'mentions','status':'observed','split':job['split'],'job_id':job['job_id'],'fact_citable':True,'evidence':job['skills']['Python']})
    group=jobs[0]['groups'][0];quote=group['evidence']
    group_id='requirement_group:'+catalog.digest([jobs[0]['job_id'],0,'any',['Python','Java'],'description',2,13])[:20]
    edges.append({'edge_id':'e-group','source_id':jobs[0]['job_id'],'target_id':group_id,'source_type':'job','target_type':'requirement_group',
                  'relation':'has_requirement_group','status':'machine_parsed','split':'train','job_id':jobs[0]['job_id'],'fact_citable':True,'evidence':quote})
    edges.append({'edge_id':'e-hyp','source_id':'skill:Python','target_id':'skill:Java','source_type':'skill','target_type':'skill',
                  'relation':'cooccurs_hypothesis','status':'hypothesis','split':'train','job_id':None,'fact_citable':False,'evidence':None,
                  'derived_from_split':'train','support_count':10})
    write_fixture(directory,jobs,edges)
    return jobs,edges


def write_fixture(directory,jobs,edges):
    for name,values in [('jobs.jsonl',jobs),('edges.jsonl',edges)]:
        (directory/name).write_text(''.join(catalog.canonical(value)+'\n' for value in values),encoding='utf-8')
    manifest={'schema_version':catalog.SOURCE_SCHEMA,'source_sha256':SNAPSHOT,'config_hash':'fixture-v1','config':{'snapshot':SNAPSHOT},
              'files':{name:{'sha256':catalog.file_hash(directory/name)} for name in ['jobs.jsonl','edges.jsonl']}}
    (directory/'manifest.json').write_text(catalog.canonical(manifest),encoding='utf-8')


class CatalogTests(unittest.TestCase):
    def test_counts_source_version_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='catalog-test-') as name:
            root=Path(name);source=root/'dataset';fixture(source);output=root/'catalog.sqlite3'
            before={name:catalog.file_hash(source/name) for name in ['jobs.jsonl','edges.jsonl','manifest.json']}
            result=catalog.export_catalog(source,output,REPOSITORY)
            self.assertEqual(result['metadata']['counts'],{'jobs':2,'job_families':2,'source_lineage':2,'requirement_groups':1,'graph_edges':4,'citable_evidence':3,'statistical_hypotheses':1})
            self.assertEqual(result['metadata']['jobs_sha256'],before['jobs.jsonl'])
            with sqlite3.connect(output) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs WHERE city=? AND district=? AND experience_min=? AND education_min=?',('深圳','南山区',1,3)).fetchone()[0],2)
                self.assertIsNone(db.execute("SELECT salary_monthly_low FROM jobs WHERE salary_status='年薪'").fetchone()[0])
                self.assertEqual(len(db.execute("PRAGMA index_list('jobs')").fetchall()),12)
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
            with self.assertRaises(FileExistsError):catalog.export_catalog(source,output,REPOSITORY)
            self.assertEqual(catalog.file_hash(output),result['sha256'])
            self.assertEqual(before,{name:catalog.file_hash(source/name) for name in before})
            with (source/'jobs.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaisesRegex(ValueError,'版本清单'):catalog.export_catalog(source,root/'tampered.sqlite3',REPOSITORY)
            self.assertFalse((root/'tampered.sqlite3').exists())

    def test_hypothesis_cannot_become_citable_and_invalid_export_is_atomic(self):
        with tempfile.TemporaryDirectory(prefix='catalog-test-') as name:
            root=Path(name);source=root/'dataset';jobs,edges=fixture(source);output=root/'catalog.sqlite3'
            catalog.export_catalog(source,output,REPOSITORY)
            with sqlite3.connect(output) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM citable_evidence WHERE status='hypothesis'").fetchone()[0],0)
                with self.assertRaises(sqlite3.IntegrityError):db.execute("UPDATE graph_edges SET fact_citable=1 WHERE status='hypothesis'")
                db.rollback()
                with self.assertRaises(sqlite3.IntegrityError):db.execute("UPDATE graph_edges SET evidence_json='{}' WHERE status='hypothesis'")
                db.rollback()
            altered=deepcopy(edges);altered[-1]['fact_citable']=True;write_fixture(source,jobs,altered)
            with self.assertRaisesRegex(ValueError,'统计假设'):catalog.export_catalog(source,root/'invalid.sqlite3',REPOSITORY)
            self.assertFalse((root/'invalid.sqlite3').exists());self.assertFalse(list(root.glob('.catalog-*')))


if __name__=='__main__':unittest.main(verbosity=2)
