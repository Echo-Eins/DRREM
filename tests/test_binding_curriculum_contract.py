import re
from scripts.probe_binding_curriculum import training_tasks
from scripts.probe_binding_limits import recode
from scripts.audit_transport_functions import tasks


def check_task(task):
    text=task['prefix'].decode()
    assert text.endswith('Answer: ')
    assert len(task['codes'])==len(set(task['codes']))==4
    assert all(len(c.encode())==3 for c in task['codes'])
    table=dict(re.findall(r'^([A-Z][a-z]+) = ([^;]+);$',text,re.M))
    query=re.findall(r'What code belongs to ([A-Z][a-z]+)\?',text)[-1]
    assert table[query]==task['codes'][task['target']]
    assert task['target']!=task['donor']
    if task['style']=='demonstration':
        assert re.findall(r'Answer: ([^\n]+)\n',text)==[task['codes'][task['donor']]]


def test_training_rewrites_preserve_entity_value_relation_and_distractor():
    batches=training_tasks(96)
    for rows in batches:
        for task in rows:check_task(task)
    assert any(len({c[:2] for c in t['codes']})==1 for rows in batches for t in rows)


def test_evaluation_rewrites_preserve_relation_and_unique_candidates():
    for family in ['unrestricted','shared_prefix','letters']:
        for task in recode(tasks(128),family):check_task(task)
