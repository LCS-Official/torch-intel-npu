"""Offline checks for activity accounting; these do not load the NPU backend."""
import unittest
from torch_npu_sustained_train import engine_interval
from tools.npu_dxcore import EngineInput, EngineOutput, MemoryInput, MemoryOutput, guid
import ctypes
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('npu_power_report', Path(__file__).parent/'npunlock-probe/research/npu_power_report.py')
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)


class ActivityAccounting(unittest.TestCase):
    def test_dxcore_abi(self):
        self.assertEqual(ctypes.sizeof(EngineInput), 12)
        self.assertEqual(ctypes.sizeof(EngineOutput), 16)
        self.assertEqual(ctypes.sizeof(MemoryInput), 12)
        self.assertEqual(ctypes.sizeof(MemoryOutput), 24)
        self.assertEqual(bytes(guid('b71b0d41-1088-422f-a27c-0250b7d3a988')).hex(),
                         '410d1bb788102f42a27c0250b7d3a988')

    def test_process_separate_from_adapter(self):
        first = dict(time=10., engines={'0:0': dict(all_us=100, process_us=50)})
        last = dict(time=12., engines={'0:0': dict(all_us=2000100, process_us=1500050)})
        self.assertEqual(engine_interval(first, last), {'0:0': dict(all_percent=100., process_percent=75.)})

    def test_missing_process_counter_is_not_full_utilization(self):
        first = dict(time=10., engines={'0:0': dict(all_us=100, process_us=None)})
        last = dict(time=11., engines={'0:0': dict(all_us=100, process_us=100)})
        self.assertEqual(engine_interval(first, last), {'0:0': dict(all_percent=0.)})

    def test_reset_or_nonpositive_interval_rejected(self):
        first = dict(time=10., engines={'0:0': dict(all_us=100)})
        with self.assertRaises(AssertionError):
            engine_interval(first, first)
        with self.assertRaises(AssertionError):
            engine_interval(first, dict(time=11., engines={'0:0': dict(all_us=50)}))

    def test_energy_to_watts_and_time_weighting(self):
        rows = power.parse('Metric: PMT-NPU-PWR | Value: 1000.00 mJ | Timestamp: 2000000000 | Duration: 1000000000\n'
                           'Metric: PMT-NPU-PWR | Value: 4000.00 mJ | Timestamp: 4000000000 | Duration: 2000000000')
        summary = power.summarize(rows)
        self.assertAlmostEqual(summary['average_w'], 5/3)
        self.assertEqual(summary['max_w'], 2)

    def test_power_sample_window_and_units(self):
        rows = power.parse('Metric: PMT-NPU-PWR | Value: 1000.00 mJ | Timestamp: 2000000000 | Duration: 1000000000')
        self.assertEqual(power.summarize(rows, 1000000000, 3000000000)['average_w'], 1)
        with self.assertRaises(AssertionError):
            power.summarize(rows, 1000000001, 3000000000)
        with self.assertRaises(ValueError):
            power.parse('Metric: PMT-NPU-PWR | Value: 1000.00 W | Timestamp: 2000000000 | Duration: 1000000000')


if __name__ == '__main__':
    unittest.main()
