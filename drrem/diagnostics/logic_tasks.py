"""Locked, paired logic prompts: all premises are supplied in the prompt.

Question and continuation forms separate instruction following from a basic
conditional completion. Scoring is conservative and supplements the complete
generated text; it is not a substitute for reading ambiguous answers.
"""
import hashlib
import re


def build_tasks():
    rows = []
    for language in ('en', 'ru'):
        for revision in (0, 1):
            code, other = ('417', '862') if revision == 0 else ('862', '417')
            first, last = ('Nim', 'Pav') if revision == 0 else ('Pav', 'Nim')
            yes, no = ('yes', 'no') if language == 'en' else ('да', 'нет')
            expected_bool = yes if revision == 0 else no
            if language == 'en':
                items = [
                    ('binding', f'Nim has code {code}. Pav has code {other}.',
                     'What is the code of Nim?', 'The code of Nim is', [code], [[code], [other]], 'wrong_binding'),
                    ('last_update', f'The key is in {first}. Then the key is moved to {last}.',
                     'Where is the key now?', 'Now the key is in', [last], [[last], [first]], 'stale_state'),
                    ('negation', f'Exactly one of Nim and Pav is selected. {first} is not selected.',
                     'Who is selected?', 'The selected one is', [last], [[last], [first]], 'negation_error'),
                    ('order', f'{first} is taller than Tor. Tor is taller than {last}.',
                     'Who is taller, Nim or Pav?', 'Of Nim and Pav, the taller one is', [first], [[first], [last]], 'order_error'),
                    ('equality', 'A equals B. B ' + ('equals C.' if revision == 0 else 'does not equal C.'),
                     'Does A equal C? Answer yes or no.', 'Does A equal C? The answer is', [expected_bool], [[yes], [no]], 'equality_error'),
                    ('implication', 'Rule: every marked item is blue. This item is ' + ('marked.' if revision == 0 else 'blue.'),
                     ('Must this item be blue?' if revision == 0 else 'Must this item be marked?') + ' Answer yes or no.',
                     ('Must this item be blue?' if revision == 0 else 'Must this item be marked?') + ' The answer is',
                     [expected_bool], [[yes], [no]], 'invalid_inference'),
                    ('count', 'There are two counters. One counter is ' + ('added.' if revision == 0 else 'removed.'),
                     'How many counters are there now?', 'The number of counters now is',
                     ['3', 'three'] if revision == 0 else ['1', 'one'], [['3', 'three'], ['1', 'one']], 'count_error'),
                    ('alternation', 'The letters alternate between A and B. The sequence so far is ' + ('A B A B.' if revision == 0 else 'B A B A.'),
                     'What is the next letter?', 'The next letter is', ['A'] if revision == 0 else ['B'], [['A'], ['B']], 'pattern_error'),
                ]
            else:
                # A language control, reported separately from the English suite.
                items = [
                    ('binding', f'У Nim код {code}. У Pav код {other}.',
                     'Какой код у Nim?', 'Код Nim —', [code], [[code], [other]], 'wrong_binding'),
                    ('last_update', f'Ключ находится в {first}. Затем ключ переместили в {last}.',
                     'Где теперь ключ?', 'Теперь ключ находится в', [last], [[last], [first]], 'stale_state'),
                    ('negation', f'Выбран ровно один из Nim и Pav. {first} не выбран.',
                     'Кто выбран?', 'Выбран', [last], [[last], [first]], 'negation_error'),
                    ('count', 'Есть два жетона. Один жетон ' + ('добавили.' if revision == 0 else 'убрали.'),
                     'Сколько теперь жетонов?', 'Количество жетонов теперь —',
                     ['3', 'три'] if revision == 0 else ['1', 'один'], [['3', 'три'], ['1', 'один']], 'count_error'),
                ]
            for family, facts, question, completion, answer, candidates, error in items:
                for style in ('question', 'completion'):
                    prompt = facts + ('\nQuestion: ' + question + '\nAnswer:' if language == 'en' else '\nВопрос: ' + question + '\nОтвет:')
                    if style == 'completion':
                        prompt = facts + '\n' + completion
                    pair = f'{language}/{family}/{style}'
                    rows.append(dict(id=f'{pair}/{revision}', pair=pair, language=language, family=family,
                                     style=style, revision=revision, prompt=prompt, expected=answer,
                                     candidates=candidates, wrong_answer_type=error,
                                     sample_seed=int.from_bytes(hashlib.sha256(pair.encode()).digest()[:4], 'little')))
    if any(len(row['prompt'].encode()) + 1 > 512 for row in rows):
        raise ValueError('a logic prompt exceeds the trained context frame')
    return rows


def assess_answer(task, record):
    text = record['text']
    # Restrict automatic grading to the first answer sentence/line. Mentioning
    # the correct label somewhere in a rambling continuation is insufficient.
    lead = re.split(r'[\n.!?]', text.strip(), maxsplit=1)[0].strip()
    mentions = []
    for aliases in task['candidates']:
        flags = 0 if task['family'] == 'alternation' and len(lead) > 1 else re.IGNORECASE
        if any(re.search(r'(?<!\w)' + re.escape(a) + r'(?!\w)', lead, flags) for a in aliases):
            mentions.append(aliases[0].lower())
    target = task['expected'][0].lower()
    correct = len(mentions) == 1 and mentions[0] == target
    negated = task['family'] not in ('equality', 'implication') and any(
        re.search(r"(?:\bnot|\bне|isn't)\s+(?:(?:in|в|the)\s+){0,2}" + re.escape(a) + r'(?!\w)',
                  lead, re.IGNORECASE) for aliases in task['candidates'] for a in aliases)
    question = '?' in text.split('\n', 1)[0] and bool(re.match(
        r'^(?:what|where|who|is|does|how|какой|кто|где|сколько)\b', lead, re.IGNORECASE))
    if not text.strip():
        error = 'empty_answer'
    elif len(mentions) > 1:
        error = 'ambiguous_or_contradictory'
    elif negated:
        error = 'negated_candidate_needs_review'
    elif question:
        error = 'question_instead_of_answer'
    elif not mentions:
        error = 'no_parseable_answer'
    elif not correct:
        error = task['wrong_answer_type']
    else:
        error = None
    correct = correct and error is None
    words = text.lower().split()
    triples = list(zip(words, words[1:], words[2:]))
    repetitive = len(triples) >= 6 and len(set(triples)) / len(triples) < .6
    return dict(correct=correct, first_answer=lead, mentioned_candidates=mentions, error=error,
                repetition_flag=repetitive, requires_manual_review=error is not None)


def summarize(records):
    groups = {}
    for row in records:
        key = f"{row['language']}/{row['style']}/{row['mode']}"
        group = groups.setdefault(key, dict(cases=0, correct=0, errors={}, pairs={}))
        group['cases'] += 1
        group['correct'] += int(row['assessment']['correct'])
        if row['assessment']['error']:
            error = row['assessment']['error']
            group['errors'][error] = group['errors'].get(error, 0) + 1
        group['pairs'].setdefault(row['pair'], []).append(row)
    for group in groups.values():
        pairs = [p for p in group.pop('pairs').values() if len(p) == 2]
        group['paired_cases'] = len(pairs)
        group['both_counterfactuals_correct'] = sum(all(r['assessment']['correct'] for r in p) for p in pairs)
        group['identical_counterfactual_text'] = sum(p[0]['generation']['text'] == p[1]['generation']['text'] for p in pairs)
    return groups
