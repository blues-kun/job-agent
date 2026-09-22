"""只用虚构材料，检查语义、跨度、三态和旧调用兼容；不连接服务。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from job_agent.domain import (
    Job, PARSER_VERSION, constraints, education, experience, experience_details,
    group_support, parse_profile, requirement_groups, skill_evidence, supported_matches, task_evidence,
)
from job_agent.semantics import ast_leaves

PREFS={'city':'深圳','intent':'技术','education':'本科','experience_years':0}


def job(description,requirements='经验不限本科'):
    return Job('fixture','v','虚构岗位','虚构企业','技术','技术','10-20K',requirements,description,'深圳南山区',1)


def profile(text,**prefs):
    return parse_profile(text,{**PREFS,**prefs})


@pytest.mark.parametrize('index',range(8))
def test_reviewed_eight_counterexamples(index):
    path=Path(__file__).resolve().parents[1]/'docs/evaluation/problem_analysis_20260922.json'
    case=json.loads(path.read_text(encoding='utf-8'))['需求解析反例'][index]
    target=job(case['jd'],case['requirements'])
    person=profile(case['resume'],education='大专' if index==6 else '本科')
    matches,gaps,coverage=supported_matches(person,target)
    checks={item['name']:item['status'] for item in constraints(person,target)}
    if index==0:
        assert 'Python' not in {skill for group in target.groups for skill in group['skills']}
        assert not gaps and coverage==1
    elif index==1:
        assert person['skills']['Python']['claim_type']=='background'
        assert coverage==0 and not matches and gaps[0]['status']=='unknown'
    elif index==2:
        ast=target.groups[0]['ast']
        assert ast['op']=='any' and [child['op'] for child in ast['children']]==['all','all']
        assert coverage==1 and not gaps
        assert len(matches[0]['leaf_support'])==4
    elif index==3:
        assert [group['preferred'] for group in target.groups]==[True,False]
        assert coverage==1 and all(gap['preferred'] for gap in gaps)
    elif index==4:
        assert 'Spring' not in target.skills and len(target.groups)==1
        assert coverage==1 and not gaps
    elif index==5:
        assert target.experience_min==3 and checks['经验要求']=='fail'
    elif index==6:
        assert target.education_min==3 and checks['学历要求']=='fail'
    else:
        assert any(group['kind']=='any' and set(group['skills'])=={'Python','Java'} for group in target.groups)
        assert coverage==1 and all(gap['preferred'] for gap in gaps)


def test_same_leaf_set_different_parentheses_has_different_meaning():
    person=profile('使用Python和Java完成项目。')
    first=job('掌握(Python或Java)和(SQL或MySQL)。')
    second=job('掌握(Python或SQL)和(Java或MySQL)。')
    assert {leaf['key'] for leaf in ast_leaves(first.groups[0]['ast'])}=={leaf['key'] for leaf in ast_leaves(second.groups[0]['ast'])}
    assert group_support(person,first)[0]['status']=='unknown'
    assert group_support(person,second)[0]['status']=='pass'


@pytest.mark.parametrize('text',['熟悉Python或Java或Go','熟悉Python、Java、Go至少一种','熟悉Python/Java/Go中的任意一种'])
def test_explicit_three_way_alternatives(text):
    target=job(text)
    assert target.groups[0]['kind']=='any'
    assert set(target.groups[0]['skills'])=={'Python','Java','Go'}
    assert group_support(profile('使用Go完成服务开发。'),target)[0]['status']=='pass'


def test_ambiguous_slash_is_unknown_not_forced_and():
    target=job('熟悉Python/Java。')
    result=group_support(profile('使用Python和Java完成项目。'),target)[0]
    assert target.groups[0]['parse_status']=='unknown'
    assert result['status']=='unknown' and (result['lower'],result['upper'])==(0,1)


def test_comma_ends_complete_alternative_before_another_requirement():
    target=job('熟悉Python或Java或Go任意一种，使用SQL进行数据处理。')
    assert [group['kind'] for group in target.groups]==['any','single']
    assert set(target.groups[0]['skills'])=={'Python','Java','Go'}
    support=group_support(profile('使用Python完成项目。'),target)
    assert [row['status'] for row in support]==['pass','unknown']
    assert supported_matches(profile('使用Python和SQL完成项目。'),target)[2]==1


def test_missing_negative_and_contradictory_evidence_are_distinct():
    target=job('熟悉Python。')
    assert group_support(profile('本科毕业。'),target)[0]['status']=='unknown'
    assert group_support(profile('不会Python。'),target)[0]['status']=='fail'
    assert group_support(profile('曾使用Python完成项目。目前不会Python。'),target)[0]['status']=='unknown'
    assert skill_evidence('并非不会Python。')['Python']['parse_status']=='unknown'


def test_future_background_and_literal_mastery_do_not_invent_practice():
    for text in ['希望使用Python完成项目。','同事使用Python完成项目。','任职要求：使用Python完成项目。']:
        evidence=skill_evidence(text)['Python']
        assert evidence['weight']<.7
    assert skill_evidence('精通Python。')['Python']['claim_type']=='self_report'
    assert skill_evidence('熟练使用Python。')['Python']['level']!='精通'
    assert skill_evidence('使用Python开发招聘系统。')['Python']['claim_type']=='self_reported_action'


def test_compound_overlap_and_independent_generic_concept():
    evidence=skill_evidence('熟悉Spring Boot和Spring Cloud。')
    assert set(evidence)=={'Spring Boot','Spring Cloud'}
    evidence=skill_evidence('熟悉Spring，也使用Spring Boot完成服务开发。')
    assert set(evidence)=={'Spring','Spring Boot'}
    assert 'Java' not in skill_evidence('使用JavaScript编写页面。')


def test_total_and_specialized_years_are_separate():
    details=experience_details('5年工作经验，2年Python开发经验。')
    assert details['total_years']==5
    assert details['specialized'][0]['minimum']==2 and details['specialized'][0]['skills']==['Python']
    assert experience('3年Python开发经验。') is None
    target=job('熟悉Python。','3年以上工作经验，2年Python开发经验，本科。')
    checks={item['name']:item['status'] for item in constraints(profile('5年工作经验。使用Python完成项目。',experience_years=5),target)}
    assert checks['经验要求']=='pass' and checks['专项经验：Python']=='unknown'
    checks={item['name']:item['status'] for item in constraints(profile('5年工作经验，1年Python开发经验。',experience_years=5),target)}
    assert checks['专项经验：Python']=='fail'


def test_unknown_days_ranges_and_degree_progress():
    assert experience('3天/周本科',True) is None
    assert experience('3-5年本科',True)==3
    assert experience('1年以内本科',True)==0
    assert experience('3-5年工作经验。') is None
    assert education('硕士在读，本科毕业。')==3
    assert education('本科必须，硕士优先。',True)==3
    assert education('本科优先。',True) is None
    assert education('本科，优秀者可放宽到大专。',True)==3


def test_quoted_job_conditions_and_conflicting_durations_are_not_profile_facts():
    assert education('目标岗位要求本科以上。') is None
    assert experience('招聘要求：3年工作经验。') is None
    assert experience('第一段工作有2年经验，第二段工作有1年工作经验。') is None
    target=job('熟悉Python。','经验不限，必须3年工作经验。')
    assert next(row for row in constraints(profile('使用Python完成项目。'),target) if row['name']=='经验要求')['status']=='unknown'


def test_preference_and_parser_version_participate_in_profile_version():
    text='使用Python完成项目。'
    first=profile(text);same=profile(text)
    assert first['version']==same['version'] and first['parser_version']==PARSER_VERSION
    for preference in [{'city':'广州'},{'experience_years':2},{'salary_min':15000},{'intent':'数据分析'}]:
        assert profile(text,**preference)['version']!=first['version']
    changed=deepcopy(PREFS);changed['salary_max']=20000
    assert parse_profile(text,changed)['salary_max']==20000


def test_task_evidence_can_cover_non_dictionary_work_without_fake_skills():
    person=profile('负责客户维护，完成客户需求分析和项目协调。')
    target=job('负责客户维护，开展客户需求分析，负责项目协调。')
    assert not person['skills']
    assert {'customer_success','customer_discovery','project_coordination'}<=set(person['tasks'])
    assert target.tasks and any(leaf['type']=='task' for leaf in ast_leaves(target.requirement_ast))
    assert any(item['type']=='task' for item in supported_matches(person,target)[0])


def test_every_returned_evidence_round_trips_to_its_own_field():
    target=job('加分项：熟悉Spring Boot。\n必需：掌握(Python和SQL)或(Java和MySQL)。','本科，Python经验优先。')
    text='本科毕业。\n使用Python和SQL完成数据清洗项目。'
    person=profile(text)
    for group in target.groups:
        for leaf in ast_leaves(group['ast']):
            citation=leaf['evidence']
            assert getattr(target,citation['field'])[citation['start']:citation['end']]==citation['quote']
    for block in person['experience_blocks']:
        assert text[block['start']:block['end']]==block['quote']
    for match in supported_matches(person,target)[0]:
        for support in match['leaf_support']:
            citation=support['job_evidence']
            assert getattr(target,citation['field'])[citation['start']:citation['end']]==citation['quote']
            evidence=support['resume_evidence']
            if evidence:
                assert text[evidence['start']:evidence['end']]==evidence['quote']
    json.dumps(target.requirement_ast,ensure_ascii=False,allow_nan=False)


@pytest.mark.parametrize('text',[
    '项目团队使用Python和SQL完成数据接口，我负责项目协调、客户需求分析和客户维护。',
    '研发团队熟练使用Python和SQL，我负责项目协调。',
    '同事使用Python和SQL而我负责项目协调。',
    '我负责项目协调而团队使用Python和SQL。',
    '团队使用Python，使用SQL处理数据，我负责项目协调。',
    '我参与的项目使用Python和SQL，我负责项目协调。',
    '我们使用Python和SQL，我负责项目协调。',
    '我负责项目协调，指导同事使用Python和SQL。',
])
def test_actor_scope_keeps_other_people_technology_out_of_personal_practice(text):
    person=profile(text)
    for key in ('Python','SQL'):
        evidence=person['skills'][key]
        assert evidence['actor']=='background' and evidence['weight']<.7
        assert evidence['polarity']=='unknown' and evidence['claim_type']=='background'
        cite=evidence['actor_evidence']
        assert cite and text[cite['start']:cite['end']]==cite['quote']
    assert person['tasks']['project_coordination']['weight']>=.7
    rows=group_support(person,job('必须掌握Python和SQL。'))
    assert rows[0]['status']=='unknown'


@pytest.mark.parametrize('text',[
    '团队使用Python，我使用SQL完成查询。',
    '团队使用Python而我使用SQL完成查询。',
    '本人使用SQL完成查询而同事使用Python。',
    '同事不会Python，我使用SQL完成查询。',
    '同事使用Python，我们使用Go，本人使用SQL完成查询。',
])
def test_different_action_subjects_remain_separate_even_without_commas(text):
    evidence=skill_evidence(text)
    assert evidence['Python']['actor']=='background' and evidence['Python']['weight']<.7
    assert evidence['Python']['polarity']=='unknown'
    assert evidence['SQL']['actor']=='self' and evidence['SQL']['level']=='实践'


def test_explicit_negative_of_other_actor_does_not_mean_personal_inability():
    text='团队不会Python，我负责项目协调。'
    assert group_support(profile(text),job('必须掌握Python。'))[0]['status']=='unknown'
    text='团队使用Python而我不会SQL。'
    evidence=skill_evidence(text)
    assert evidence['Python']['actor']=='background'
    assert evidence['SQL']['actor']=='self' and evidence['SQL']['polarity']=='negative'
    assert group_support(profile(text),job('必须掌握SQL。'))[0]['status']=='fail'


def test_background_postposition_and_ambiguous_collective_are_conservative():
    assert skill_evidence('Python由同事使用，我负责项目协调。')['Python']['actor']=='background'
    assert skill_evidence('系统使用Python，我负责项目协调。')['Python']['weight']<.7
    assert skill_evidence('我与团队共同使用Python。')['Python']['weight']<.7
    assert skill_evidence('团队中本人独立使用Python完成脚本。')['Python']['level']=='实践'


def test_actor_rules_do_not_break_implicit_resume_actions_task_nouns_or_job_fields():
    assert skill_evidence('使用Python和SQL完成清洗脚本。')['Python']['level']=='实践'
    person=profile('负责项目协调和系统部署，执行测试，完成客户需求分析。')
    for key in ('project_coordination','deployment','test_execution','customer_discovery'):
        assert person['tasks'][key]['weight']>=.7 and person['tasks'][key]['actor']=='self'
    jd=skill_evidence('项目团队使用Python和SQL。','description')
    assert all(item['actor']=='job' for item in jd.values())


def test_actor_semantics_version_changes_profile_and_group_versions():
    assert 'v2.0.2-actor-scope' in PARSER_VERSION
    person=profile('团队使用Python，我负责项目协调。')
    target=job('掌握Python。')
    assert person['parser_version']==PARSER_VERSION
    assert target.parser_version==PARSER_VERSION


@pytest.mark.parametrize('text',[
    '我使用Python完成脚本，SQL由同事使用。',
    '我使用Python完成脚本而SQL由团队使用。',
])
def test_postposed_explicit_actor_overrides_inherited_self_subject(text):
    evidence=skill_evidence(text)
    assert evidence['Python']['actor']=='self' and evidence['Python']['weight']>=.7
    assert evidence['SQL']['actor']=='background' and evidence['SQL']['weight']<.7
