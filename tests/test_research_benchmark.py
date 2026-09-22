"""仅临时合成岗位；不访问模型、真实简历或真实岗位文件。"""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from research import build_benchmark as benchmark

REPOSITORY=Path(__file__).resolve().parents[1]


def fixture(folder):
    rows=[]
    for split in ('train','dev','test'):
        for index in range(65):
            text=f'使用Python和SQL完成第{index}个项目的接口开发与测试。'
            row={'job_id':f'{split}-{index:03d}','job_family_id':f'family-{split}-{index:03d}','split':split,
                 'title':f'虚构Python岗位{index}','category':'Python','requirements':'3-5年本科','description':text,
                 'skills':{'Python':{'level':'实践','field':'description','start':2,'end':8,'quote':'Python'},
                           'SQL':{'level':'实践','field':'description','start':9,'end':12,'quote':'SQL'}},
                 'salary':{'status':'月薪','monthly_low':8000,'monthly_high':18000}}
            rows.append(row)
        duplicate=dict(rows[-1]);duplicate['job_id']+='-variant';rows.append(duplicate)
    folder.mkdir()
    benchmark.write_jsonl(folder/'jobs.jsonl',rows)
    benchmark.write_json(folder/'manifest.json',{'config_hash':'synthetic-only','files':{'jobs.jsonl':{'sha256':benchmark.sha256(folder/'jobs.jsonl')}}})


class BenchmarkTests(unittest.TestCase):
    def test_fiction_counts_and_fixed_strata(self):
        queries=benchmark.make_queries()
        self.assertEqual(len(queries),48)
        self.assertEqual(Counter(q['split'] for q in queries),Counter(dev=12,test=36))
        self.assertEqual(Counter(q['split']+'/'+q['scenario_group'] for q in queries),Counter({'dev/recommendation_case':6,'dev/clarification_stress':6,'test/recommendation_case':30,'test/clarification_stress':6}))
        self.assertTrue(all(q['is_real_resume'] is False and q['is_human_gold'] is False for q in queries))
        for direction in benchmark.DIRECTIONS:
            cases=[q for q in queries if q['direction']==direction['name']]
            self.assertEqual(len(cases),8);self.assertEqual(Counter(q['split'] for q in cases),Counter(dev=2,test=6))

    def test_pools_labels_lineage_and_tamper_protection(self):
        with tempfile.TemporaryDirectory(prefix='job-benchmark-test-') as folder:
            dataset=Path(folder)/'dataset';fixture(dataset)
            before=benchmark.sha256(dataset/'jobs.jsonl')
            run,manifest=benchmark.build(dataset,REPOSITORY,review_seeds=5)
            self.assertFalse(manifest['dense']['used'])
            self.assertEqual(manifest['checks']['relevance_tasks'],1440)
            self.assertEqual(manifest['checks']['review_profiles'],10)
            self.assertEqual((run/'qrels.jsonl').read_bytes(),b'')
            tasks=benchmark.read_jsonl(run/'tasks.jsonl')
            self.assertTrue(all(task['label'] is None and task['kind']=='relevance' for task in tasks))
            self.assertEqual(len({task['task_id'] for task in tasks}),1440)
            pools=benchmark.read_jsonl(run/'pools.jsonl')
            jobs={job['job_id']:job for job in benchmark.read_jsonl(dataset/'jobs.jsonl')}
            for pool in pools:
                self.assertEqual(len(pool['candidates']),30)
                self.assertEqual(len({candidate['job_family_id'] for candidate in pool['candidates']}),30)
                self.assertTrue(all(jobs[candidate['job_id']]['split']==pool['split'] for candidate in pool['candidates']))
                self.assertTrue(all(source['source']!='frozen_bge' for candidate in pool['candidates'] for source in candidate['sources']))
            profiles=benchmark.read_jsonl(run/'review_profiles.jsonl')
            self.assertEqual(len({row['job_family_id'] for row in profiles}),5)
            for profile in profiles:
                self.assertTrue(profile['not_for_training']);self.assertEqual(profile['source'],'programmatic_unreviewed')
                self.assertEqual(jobs[profile['seed_job_id']]['split'],'train')
                for evidence in profile['jd_evidence']:
                    self.assertEqual(jobs[evidence['job_id']][evidence['field']][evidence['start']:evidence['end']],evidence['quote'])
            same,again=benchmark.build(dataset,REPOSITORY,review_seeds=5)
            self.assertEqual(same,run);self.assertEqual(again,manifest)
            self.assertEqual(benchmark.sha256(dataset/'jobs.jsonl'),before)
            with (run/'tasks.jsonl').open('a') as stream:stream.write('{}\n')
            with self.assertRaisesRegex(ValueError,'被改动'):
                benchmark.build(dataset,REPOSITORY,review_seeds=5)
            self.assertEqual(benchmark.sha256(dataset/'jobs.jsonl'),before)

    def test_symlink_and_insufficient_seeds_never_publish(self):
        with tempfile.TemporaryDirectory(prefix='job-benchmark-test-') as folder:
            root=Path(folder);dataset=root/'dataset';fixture(dataset)
            with self.assertRaisesRegex(ValueError,'家族不足'):
                benchmark.build(dataset,REPOSITORY,review_seeds=500)
            self.assertFalse(list((dataset/'benchmark').iterdir()))
            (dataset/'benchmark').rmdir();(dataset/'benchmark').symlink_to(REPOSITORY,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'符号链接'):
                benchmark.build(dataset,REPOSITORY,review_seeds=0)


if __name__=='__main__':unittest.main(verbosity=2)
