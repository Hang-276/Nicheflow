import copy
import tempfile
from pathlib import Path
import unittest
from nicheflow.v072.durable import Store,Paused


class DurableRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.frozen={'limits':{'models':{'H':{'calls':10,'input_tokens':1000,'output_tokens':100}},'currency':{'CNY':1.}}}
        self.profile={'currency':'CNY','input_per_million':12.,'output_per_million':36.}
        self.payload={'messages':[{'role':'user','content':'public task'}]}
    def tearDown(self):self.tmp.cleanup()
    def start(self,s,id='a',output=30):return s.begin(id,'H',self.payload,self.profile,'development',50,output)
    def response(self):return {'status':'ok','text':'answer','input_tokens':10,'output_tokens':20,'reference_cost':.00084,'currency':'CNY','finish_reason':'stop'}
    def test_quota_pause_keeps_inflight_receipts_and_resumes_only_missing(self):
        s=Store(self.path,self.frozen)
        self.start(s,'done');s.finish('done',self.response())
        self.start(s,'rejected');self.start(s,'inflight')
        s.finish('rejected',{'status':'http_error','http_status':429,'error_code':'insufficient_quota','reference_cost':None})
        with self.assertRaises(Paused):self.start(s,'not_dispatched')
        s.finish('inflight',self.response());s.backup();s.close()
        self.assertTrue((self.path/'checkpoint.sqlite3').is_file())
        s=Store(self.path,self.frozen,resume=True)
        self.assertEqual(self.start(s,'done'),self.response())
        self.assertEqual(self.start(s,'inflight'),self.response())
        self.assertIsNone(self.start(s,'rejected'))
        s.finish('rejected',self.response())
        self.assertEqual(s.totals()['models']['H']['complete'],3)
        self.assertEqual(s.totals()['models']['H']['attempts'],4)
        self.assertEqual(s.db.execute("SELECT COUNT(*) FROM calls WHERE id='not_dispatched'").fetchone()[0],0)
        s.close()
    def test_crash_after_dispatch_preserves_completed_and_never_retries_unknown(self):
        s=Store(self.path,self.frozen)
        self.start(s,'done');s.finish('done',self.response())
        self.start(s,'interrupted');s.close()
        s=Store(self.path,self.frozen,resume=True)
        self.assertEqual(s.unresolved(),['interrupted'])
        self.assertEqual(self.start(s,'done'),self.response())
        with self.assertRaises(Paused):self.start(s,'interrupted')
        with self.assertRaises(Paused):self.start(s,'next')
        self.assertEqual(s.totals()['models']['H']['reserved_output'],30)
        s.close()
    def test_concurrent_reservations_prevent_overspending(self):
        s=Store(self.path,self.frozen)
        self.start(s,'one',60)
        with self.assertRaisesRegex(Paused,'token_budget'):self.start(s,'two',60)
        s.finish('one',self.response())
        self.assertEqual(s.totals()['models']['H']['output_tokens'],20)
        self.assertEqual(s.totals()['models']['H']['reserved_output'],0)
        s.close()
    def test_changed_protocol_and_changed_request_refuse_receipt_reuse(self):
        s=Store(self.path,self.frozen);self.start(s);s.finish('a',self.response())
        with self.assertRaisesRegex(ValueError,'different input'):
            s.begin('a','H',{'messages':[]},self.profile,'development',50,30)
        s.close()
        changed=copy.deepcopy(self.frozen);changed['new_protocol']=True
        with self.assertRaisesRegex(ValueError,'frozen'):Store(self.path,changed,resume=True)
        s=Store(self.path,self.frozen,resume=True);self.assertEqual(s.receipt('a'),self.response());s.close()
    def test_scoring_failure_cannot_delete_saved_paid_response(self):
        s=Store(self.path,self.frozen);self.start(s);s.finish('a',self.response())
        try:raise ValueError('local scorer failed')
        except ValueError:pass
        s.close();s=Store(self.path,self.frozen,resume=True)
        self.assertEqual(self.start(s),self.response())
        self.assertEqual(s.totals()['models']['H']['attempts'],1);s.close()
    def test_timeout_keeps_reservation_instead_of_recording_zero_cost(self):
        s=Store(self.path,self.frozen);self.start(s)
        s.finish('a',{'status':'unknown','error_type':'TimeoutError','reference_cost':None})
        self.assertEqual(s.totals()['models']['H']['reserved_output'],30)
        self.assertGreater(s.totals()['currencies']['CNY']['reserved'],0)
        s.close();s=Store(self.path,self.frozen,resume=True)
        with self.assertRaises(Paused):self.start(s)
        s.close()
