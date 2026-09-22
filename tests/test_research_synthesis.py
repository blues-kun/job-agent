"""仅临时合成岗位与桩模型，确认隔离、未审阅状态和不重试；不访问模型服务。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import synthesize_resume_pilot as pilot

REPOSITORY = Path(__file__).resolve().parents[1]


def evidence(text):
    return {skill: {} for skill in ('Python', 'SQL', 'Java') if skill in text}


def fixture(folder):
    rows = []
    for split in ('train', 'dev', 'test'):
        for index in range(25):
            text = f'使用Python完成虚构第{index}项数据整理任务，编写接口和复查结果；本材料仅用于软件测试，不是真实招聘。'
            rows.append({
                'job_id': f'{split}-{index:03d}',
                'job_family_id': f'family-{split}-{index:03d}', 'split': split,
                'category': f'方向{index % 3}', 'requirements': 'Python',
                'description': text, 'company': '', 'snapshot': 'fixture-only',
                'source_record_ids': [f'fixture:sheet:1:row:{index + 2}'],
                'skills': {'Python': {'level': '实践', 'field': 'description',
                                     'start': 2, 'end': 8, 'quote': 'Python'}},
            })
    duplicate = dict(rows[0]); duplicate['job_id'] += '-version2'; rows.append(duplicate)
    folder.mkdir()
    pilot.write_jsonl(folder/'jobs.jsonl', rows)
    pilot.write_json(folder/'manifest.json', {
        'source_sha256': 'fixture-only',
        'files': {'jobs.jsonl': {'sha256': pilot.sha256(folder/'jobs.jsonl')}},
    })
    return rows


class PilotTests(unittest.TestCase):
    def test_json_and_scope_checks(self):
        valid = {'fictional': True, 'resume_text': '【完全虚构研究样例】求职方向为数据整理，虚构项目使用Python处理输入文件并复查结果。', 'skills': ['Python']}
        self.assertTrue(pilot.check_output(valid, ['Python'], evidence, [])['automatic_gate_pass'])
        extra = dict(valid, resume_text=valid['resume_text']+'另用SQL整理。')
        self.assertIn('skills_outside_jd_dictionary_scope', pilot.check_output(extra, ['Python'], evidence, [])['flags'])
        declared = dict(valid, skills=['SQL'])
        self.assertIn('declared_skill_not_found_in_text', pilot.check_output(declared, ['Python', 'SQL'], evidence, [])['flags'])
        contact = dict(valid, resume_text=valid['resume_text']+' 邮箱：example@example.invalid')
        self.assertIn('contact_detected', pilot.check_output(contact, ['Python'], evidence, [])['flags'])
        self.assertFalse(pilot.check_output({'fictional': True}, ['Python'], evidence, [])['schema_valid'])

    def test_exact_twenty_distinct_train_families(self):
        with tempfile.TemporaryDirectory(prefix='job-pilot-test-') as folder:
            rows = fixture(Path(folder)/'dataset')
            selected = pilot.choose_seeds(rows, 42)
            self.assertEqual(selected, pilot.choose_seeds(list(reversed(rows)), 42))
            self.assertEqual(len(selected), 20)
            self.assertEqual(len({row['job_family_id'] for row in selected}), 20)
            self.assertTrue(all(row['split'] == 'train' for row in selected))
            with self.assertRaisesRegex(ValueError, '不足20'):
                pilot.choose_seeds([row for row in rows if row['split'] == 'test'], 42)

    def test_no_retry_or_training_eligibility_and_tamper_protection(self):
        with tempfile.TemporaryDirectory(prefix='job-pilot-test-') as folder:
            root = Path(folder); dataset = root/'dataset'; fixture(dataset)
            before = pilot.sha256(dataset/'jobs.jsonl')
            calls = []
            health = {'config_fingerprint': 'fixture-model', 'model': '桩模型'}

            def fake_request(url, data=None):
                if url.endswith('/health'):
                    return health
                calls.append(data)
                if len(calls) == 3:
                    raise ValueError('模拟JSON失败，不重试')
                decision = {'fictional': True, 'resume_text': '【完全虚构研究样例】求职方向为数据整理，虚构项目使用Python读取练习文件并对输入结果执行检查，记录处理流程。', 'skills': ['Python']}
                return {'model': health, 'decision': decision}

            with patch.object(pilot, 'local_request', side_effect=fake_request), patch('builtins.print'):
                target, manifest = pilot.run(dataset, REPOSITORY, root/'runs', 'http://127.0.0.1:8092')
                self.assertEqual(len(calls), 20)
                self.assertEqual(manifest['summary']['generation_returned_dict'], 19)
                self.assertEqual(manifest['summary']['eligible_for_supervised_training'], 0)
                self.assertEqual(manifest['summary']['human_reviewed'], 0)
                records = pilot.read_jsonl(target/'samples.jsonl')
                self.assertEqual(len(records), 20)
                self.assertEqual(records[2]['generation_error']['type'], 'ValueError')
                self.assertTrue(all(row['status'] == 'synthetic_unreviewed' and row['not_for_training'] and row['label'] is None and row['human_review'] is None for row in records))
                same, again = pilot.run(dataset, REPOSITORY, root/'runs', 'http://127.0.0.1:8092')
                self.assertEqual(same, target); self.assertEqual(again, manifest)
                self.assertEqual(len(calls), 20)
                with (target/'samples.jsonl').open('a') as stream:
                    stream.write('{}\n')
                with self.assertRaisesRegex(ValueError, '被改动'):
                    pilot.run(dataset, REPOSITORY, root/'runs', 'http://127.0.0.1:8092')
                self.assertEqual(len(calls), 20)
            self.assertEqual(pilot.sha256(dataset/'jobs.jsonl'), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
