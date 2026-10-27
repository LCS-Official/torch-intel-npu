"""Graph ownership and LRU pressure without importing the hardware backend."""
import ast
from collections import OrderedDict
from contextlib import nullcontext, contextmanager
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
from weakref import WeakSet, WeakValueDictionary, ref


def runtime(limit=2):
    class Session:
        def __init__(self):self._closed=False
        def close(self):self._closed=True
        def __del__(self):self.close()
    state=dict(_sessions=OrderedDict(),_rebuildable_sessions=set(),_retired_sessions=WeakSet(),
               _SMOL_SESSION_KINDS=('smollm_fused_norm','smollm_prefill_recipe','smollm_rmsnorm'),_smol_plan_depth=0,contextmanager=contextmanager,
               _session_evictions=0,_session_creations=1,_MAX_SESSIONS=limit,_MAX_SESSION_BYTES=512*1024**2,
               _graph_factories=WeakValueDictionary(),_graph_factory_calls=0,_transient_graph_imports=0,_lock=nullcontext(),
               synchronize=lambda:None,_root_session=Session(),_closed=False,_scalars=OrderedDict(),
               _C=SimpleNamespace(shutdown=lambda:None),
               _native=SimpleNamespace(create_inference_session=lambda *a,**k:Session()))
    tree=ast.parse(Path(__file__).with_name('torch_intel_npu.py').read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('set_graph_cache_budget','_smol_graph_plan','_session_blob_bytes','_make_session_room','_cache_program','shutdown','_register_graph_factory','_session')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<graph-cache>','exec'),state)
    def cached(key):
        state['_sessions'].move_to_end(key)
        return state['_sessions'][key]
    state['_compile_session']=cached
    return state


class GraphCapacityTests(unittest.TestCase):
    def test_budget_resize_validates_without_evicting_live_graphs(self):
        state=runtime();setter=state['set_graph_cache_budget']
        old=state['_cache_program']('kept',SimpleNamespace(graph_blob=b'abc'),rebuildable=True)[0]
        for value in (0,-1,True,1.5,1024**3+1,2):
            with self.assertRaises(ValueError):setter(value)
        with self.assertRaisesRegex(RuntimeError,'plan'):
            with state['_smol_graph_plan']():setter(768*1024**2)
        setter(768*1024**2)
        self.assertEqual(state['_MAX_SESSION_BYTES'],768*1024**2)
        self.assertFalse(old._closed);self.assertEqual(state['_session_evictions'],0)

    def test_oversized_smol_graph_cannot_bypass_the_family_guard_as_transient(self):
        state=runtime();state['_MAX_SESSION_BYTES']=1
        with self.assertRaisesRegex(RuntimeError,'set_graph_cache_budget'):
            state['_cache_program'](('smollm_fused_norm',b'a'),SimpleNamespace(graph_blob=b'xx'),rebuildable=True)
        self.assertEqual(state['_session_creations'],1);self.assertFalse(state['_sessions'])

    def test_smol_pressure_retires_the_whole_family_and_preserves_other_graphs(self):
        state=runtime(4);load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        a=('smollm_fused_norm',b'a');b=('smollm_prefill_recipe',b'b')
        old=ref(load(a,program,rebuildable=True)[0]);load(b,program,rebuildable=True)
        pinned=load('unrelated-pinned',program)[0];ordinary=load('ordinary',program,rebuildable=True)[0]
        load(('smollm_fused_norm',b'c'),program,rebuildable=True);gc.collect()
        self.assertIsNone(old());self.assertNotIn(a,state['_sessions']);self.assertNotIn(b,state['_sessions'])
        self.assertFalse(pinned._closed);self.assertFalse(ordinary._closed)
        self.assertEqual(state['_session_evictions'],2)

    def test_smol_plan_pressure_rejects_before_teardown_and_restores_scope(self):
        state=runtime(2);load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        a=load(('smollm_fused_norm',b'a'),program,rebuildable=True)[0]
        b=load(('smollm_prefill_recipe',b'b'),program,rebuildable=True)[0]
        with self.assertRaisesRegex(RuntimeError,'prepared/pinned'):
            with state['_smol_graph_plan']():
                with state['_smol_graph_plan']():load(('smollm_fused_norm',b'c'),program,rebuildable=True)
        self.assertEqual(state['_smol_plan_depth'],0);self.assertEqual(state['_session_evictions'],0)
        self.assertFalse(a._closed or b._closed);self.assertEqual(len(state['_sessions']),2)

    def test_pinned_smol_family_is_not_partially_evicted(self):
        state=runtime(2);load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        a=load(('smollm_fused_norm',b'a'),program,rebuildable=True)[0]
        b=load(('smollm_prefill_recipe',b'b'),program)[0]
        with self.assertRaisesRegex(RuntimeError,'prepared/pinned'):load('incoming',program,rebuildable=True)
        self.assertFalse(a._closed or b._closed);self.assertEqual(state['_session_evictions'],0)

    def test_lru_releases_only_rebuildable_graphs(self):
        state=runtime();cache=state['_sessions'];load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'graph')
        first=ref(load('first',program,rebuildable=True)[0])
        load('second',program,rebuildable=True)
        load('first',program,rebuildable=True)  # Touch the oldest entry.
        second=ref(cache['second'][0])
        load('third',program,rebuildable=True)
        gc.collect()
        self.assertIsNone(second());self.assertIsNotNone(first())
        self.assertEqual(list(cache),['first','third']);self.assertEqual(state['_session_evictions'],1)

    def test_plan_retains_evicted_graph_until_last_owner_releases(self):
        state=runtime(1);load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        plan=[load('planned',program,rebuildable=True)]
        session=ref(plan[0][0]);load('other',program,rebuildable=True)
        self.assertFalse(session()._closed);self.assertEqual(len(state['_retired_sessions']),1)
        plan.clear();gc.collect()
        self.assertIsNone(session());self.assertEqual(len(state['_retired_sessions']),0)

    def test_frozen_graph_stays_pinned_and_full_cache_rejects_safely(self):
        state=runtime();load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        pinned=load('frozen',program)[0];load('ordinary',program,rebuildable=True)
        load('frozen2',program)
        self.assertFalse(pinned._closed)
        with self.assertRaisesRegex(RuntimeError,'pinned'):load('extra',program,rebuildable=True)
        self.assertEqual(list(state['_sessions']),['frozen','frozen2'])

    def test_disabled_or_failed_cache_does_not_destroy_existing_graphs(self):
        state=runtime();load=state['_cache_program'];program=SimpleNamespace(graph_blob=b'g')
        session=load('existing',program,rebuildable=True)[0]
        for limit in (0,-1,True):
            state['_MAX_SESSIONS']=limit
            with self.assertRaisesRegex(RuntimeError,'cache'):load('extra',program)
        state['_MAX_SESSIONS']=1
        def failed():raise RuntimeError('context failed')
        state['synchronize']=failed
        with self.assertRaisesRegex(RuntimeError,'context failed'):load('extra',program)
        self.assertFalse(session._closed);self.assertEqual(list(state['_sessions']),['existing'])

    def test_native_import_failure_is_propagated_without_a_cached_invalid_session(self):
        state=runtime(1);program=SimpleNamespace(graph_blob=b'g')
        def fail(*a,**k):raise RuntimeError('native import failed')
        state['_native'].create_inference_session=fail
        with self.assertRaisesRegex(RuntimeError,'native import failed'):state['_cache_program']('bad',program,rebuildable=True)
        self.assertFalse(state['_sessions']);self.assertEqual(state['_session_creations'],1)

    def test_shutdown_closes_sessions_retained_by_external_plans(self):
        state=runtime(1);program=SimpleNamespace(graph_blob=b'g');load=state['_cache_program']
        plan=[load('planned',program,rebuildable=True)]
        cached=load('cached',program,rebuildable=True)[0]
        self.assertFalse(plan[0][0]._closed)
        state['shutdown']()
        self.assertTrue(plan[0][0]._closed);self.assertTrue(cached._closed)
        self.assertTrue(state['_closed']);self.assertFalse(state['_sessions']);self.assertFalse(state['_retired_sessions'])

    def test_byte_budget_evicts_by_blob_bytes_before_item_limit(self):
        state=runtime(10);state['_MAX_SESSION_BYTES']=8;load=state['_cache_program']
        load('a',SimpleNamespace(graph_blob=b'aaa'),rebuildable=True)
        load('b',SimpleNamespace(graph_blob=b'bbbb'),rebuildable=True)
        load('c',SimpleNamespace(graph_blob=b'ccc'),rebuildable=True)
        self.assertEqual(list(state['_sessions']),['b','c'])
        self.assertEqual(state['_session_blob_bytes'](),7)
        with self.assertRaisesRegex(RuntimeError,'byte budget'):
            load('oversize-pinned',SimpleNamespace(graph_blob=b'x'*9))
        transient=load('oversize',SimpleNamespace(graph_blob=b'x'*9),rebuildable=True)
        self.assertFalse(transient[0]._closed);self.assertEqual(state['_transient_graph_imports'],1)
        self.assertIn(transient[0],state['_retired_sessions'])
        self.assertEqual(list(state['_sessions']),['b','c'])
        state['shutdown']();self.assertTrue(transient[0]._closed)

    def test_custom_factory_rebuilds_after_eviction_and_shared_plan_survives(self):
        state=runtime(1);builds=[];program=SimpleNamespace(graph_blob=b'g')
        def factory():builds.append(1);return program
        state['_register_graph_factory']('custom',factory)
        held=state['_session']('custom')[0]
        state['_cache_program']('other',program,rebuildable=True)
        self.assertFalse(held._closed)
        replacement=state['_session']('custom')[0]
        self.assertIsNot(held,replacement);self.assertEqual(len(builds),2)
        self.assertIn(held,state['_retired_sessions'])
        state['shutdown']();self.assertTrue(held._closed);self.assertTrue(replacement._closed)

    def test_factory_registration_does_not_retain_model_or_large_captured_weights(self):
        state=runtime()
        class Owner:pass
        owner=Owner();owner.weights=bytearray(1024);owner_ref=ref(owner)
        def make(owner):return lambda:owner.weights
        factory=make(owner);state['_register_graph_factory']('custom',factory)
        self.assertEqual(len(state['_graph_factories']),1)
        del owner,factory;gc.collect()
        self.assertIsNone(owner_ref());self.assertFalse(state['_graph_factories'])

    def test_factory_failure_leaves_no_invalid_session_and_can_retry(self):
        state=runtime();program=SimpleNamespace(graph_blob=b'g')
        def fail():raise ValueError('unsupported compiler profile')
        state['_register_graph_factory']('custom',fail)
        with self.assertRaisesRegex(ValueError,'profile'):state['_session']('custom')
        self.assertFalse(state['_sessions'])
        factory=lambda:program;state['_register_graph_factory']('custom',factory)
        self.assertIs(state['_session']('custom')[1],program)


if __name__=='__main__':unittest.main()
