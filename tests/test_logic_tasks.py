from drrem.diagnostics.logic_tasks import build_tasks, assess_answer, summarize


def test_each_counterfactual_changes_the_target_without_exceeding_context():
    rows = build_tasks()
    pairs = {}
    for row in rows:
        assert len(row['prompt'].encode())+1 <= 512
        assert assess_answer(row, {'text':' '+row['expected'][0]+'.'})['correct']
        pairs.setdefault(row['pair'], []).append(row)
    assert len(rows)==48 and len(pairs)==24
    for pair in pairs.values():
        assert len(pair)==2
        assert pair[0]['prompt'] != pair[1]['prompt']
        assert set(pair[0]['expected']).isdisjoint(pair[1]['expected'])
        assert pair[0]['sample_seed']==pair[1]['sample_seed']


def test_scorer_does_not_reward_mentioning_both_or_a_late_correct_label():
    row = build_tasks()[0]
    assert row['expected']==['417']
    for text in ('417 or 862.', '862. Later 417.', 'The code is elsewhere.', '',
                 'The code is not 417.', 'What is the code 417?'):
        assert not assess_answer(row, {'text':text})['correct']
    assert assess_answer(row, {'text':'The code is 417.'})['correct']


def test_english_article_is_not_a_generated_pattern_answer():
    row = next(r for r in build_tasks() if r['family']=='alternation' and r['revision']==0)
    assert not assess_answer(row, {'text':'This is a difficult question.'})['correct']
    assert assess_answer(row, {'text':'A.'})['correct']
