import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_transport_external_guard import question_key, reserve


def test_reservation_excludes_old_and_normalized_question_duplicates(tmp_path):
    full=pa.table({'id':[str(i) for i in range(8)],'system_prompt':['']*8,
                   'question':['old question','old   question','Ａ','A','B','C','D','empty'],
                   'response':['answer']*7+['']})
    previous=full.take(pa.array([0])).append_column('source_row',pa.array([0]))
    source=tmp_path/'source.parquet';old=tmp_path/'old.parquet'
    pq.write_table(full,source);pq.write_table(previous,old)
    guard,_=reserve(source,old,count=4,seed=42)
    assert not set(guard['source_row'].to_pylist())&{0,1,7}
    keys=[question_key(q) for q in guard['question'].to_pylist()]
    assert len(set(keys))==4
    assert guard.equals(reserve(source,old,count=4,seed=42)[0])


def test_reservation_rejects_changed_source_id_mapping(tmp_path):
    full=pa.table({'id':['x','y'],'system_prompt':['',''],'question':['q1','q2'],'response':['r1','r2']})
    old=full.take(pa.array([0])).append_column('source_row',pa.array([1]))
    source=tmp_path/'source.parquet';previous=tmp_path/'previous.parquet'
    pq.write_table(full,source);pq.write_table(old,previous)
    with pytest.raises(ValueError,match='reproduce'):
        reserve(source,previous,1,0)
