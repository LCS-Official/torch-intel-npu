import importlib.util
from pathlib import Path
import sys
import unittest

RESEARCH = Path(__file__).parent/'npunlock-probe/research'
sys.path.insert(0, str(RESEARCH))
from gemm_energy_report import mj_per_gemm, verify_work
from npu_power_report import parse


class EnergyAudit(unittest.TestCase):
    def test_watts_to_millijoules_per_task(self):
        self.assertEqual(mj_per_gemm(10, 1000), 10)
        with self.assertRaises(AssertionError):
            mj_per_gemm(10, 0)

    def test_gpu_domain_is_opt_in(self):
        line='Metric: PMT-VCCGT-PWR | Value: 1000.00 mJ | Timestamp: 2000000000 | Duration: 1000000000'
        with self.assertRaises(AssertionError):
            parse(line)
        self.assertEqual(parse(line, accepted=('PMT-VCCGT-PWR',))[0]['metric'], 'PMT-VCCGT-PWR')

    def test_fused_graphs_are_not_counted_as_single_gemms(self):
        row=dict(meta=dict(m=1024,k=1024,n=1024,groups=4,layout='fused4',precision='float16',
                          dpu_modes=[dict(amode=0,wmode=0)]), seconds=1,completed_gemms=64,
                 chunks=[1],gemms_per_chunk=64,gemms_per_second=64,teraops=2*1024**3*64/1e12,
                 validation=dict(relative_l2_max=0),stats_before=dict(npu_calls=0),stats_after=dict(npu_calls=16))
        for name in ('host_upload_copies','host_download_copies','host_device_copies','session_creations',
                     'driver_allocations','live_buffers','live_buffer_bytes'):
            row['stats_before'][name]=row['stats_after'][name]=0
        verify_work(row)
        row['stats_after']['npu_calls']=64
        with self.assertRaises(AssertionError):
            verify_work(row)


if __name__ == '__main__':
    unittest.main()
