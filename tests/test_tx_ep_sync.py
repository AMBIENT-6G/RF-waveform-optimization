from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import zmq


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from measure_ep_power import (  # noqa: E402
    collect_measurements_until_stop,
    create_tx_notify_socket,
    derive_measure_window_s,
    trim_trailing_samples,
    wait_for_tx_started,
)
from tx_waveform import TxEventNotifier  # noqa: E402


class FakeProcess:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code

    def poll(self) -> int | None:
        return self.exit_code


class FakeSerialPort:
    def __init__(self, timeout: float = 1.0) -> None:
        self.timeout = timeout


class FakeProfiler:
    def __init__(
        self,
        *,
        push_socket: zmq.Socket | None = None,
        tx_done_after_samples: int | None = None,
        sample_delay_s: float = 0.0,
    ) -> None:
        self.serial_port = FakeSerialPort()
        self._push_socket = push_socket
        self._tx_done_after_samples = tx_done_after_samples
        self._sample_delay_s = sample_delay_s
        self._count = 0

    def get_measurement(self) -> dict[str, int] | None:
        self._count += 1
        if self._push_socket is not None and self._tx_done_after_samples == self._count:
            self._push_socket.send_json({"event": "tx_done"})
            time.sleep(max(self._sample_delay_s, 0.01))
        elif self._sample_delay_s > 0:
            time.sleep(self._sample_delay_s)
        return {
            "timestamp_ms": self._count,
            "buffer_voltage_mv": 100,
            "resistance": 1,
            "pwr_pw": self._count,
            "pot_val": 0,
        }


class TxEpSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = zmq.Context()

    def tearDown(self) -> None:
        self.context.term()

    def _create_push_socket(self, endpoint: str) -> zmq.Socket:
        socket = self.context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(endpoint)
        time.sleep(0.05)
        return socket

    def test_derive_measure_window(self) -> None:
        self.assertAlmostEqual(derive_measure_window_s(20.0), 18.0)

    def test_trim_trailing_samples(self) -> None:
        kept, trimmed = trim_trailing_samples([{"sample": idx} for idx in range(12)], trim_count=10)
        self.assertEqual(trimmed, 10)
        self.assertEqual(len(kept), 2)

    def test_trim_trailing_samples_underflow(self) -> None:
        kept, trimmed = trim_trailing_samples([{"sample": idx} for idx in range(5)], trim_count=10)
        self.assertEqual(trimmed, 5)
        self.assertEqual(kept, [])

    def test_wait_for_tx_started_receives_event(self) -> None:
        notify_socket, endpoint = create_tx_notify_socket(self.context)
        push_socket = self._create_push_socket(endpoint)
        try:
            push_socket.send_json({"event": "tx_started", "tone": 4})
            event = wait_for_tx_started(FakeProcess(), notify_socket, timeout_s=1.0)
        finally:
            push_socket.close(0)
            notify_socket.close(0)
        self.assertEqual(event["event"], "tx_started")
        self.assertEqual(event["tone"], 4)

    def test_wait_for_tx_started_fails_when_process_exits(self) -> None:
        notify_socket, _ = create_tx_notify_socket(self.context)
        try:
            with self.assertRaises(RuntimeError):
                wait_for_tx_started(FakeProcess(exit_code=1), notify_socket, timeout_s=0.2)
        finally:
            notify_socket.close(0)

    def test_collect_measurements_stops_on_tx_done_and_trims(self) -> None:
        notify_socket, endpoint = create_tx_notify_socket(self.context)
        push_socket = self._create_push_socket(endpoint)
        profiler = FakeProfiler(push_socket=push_socket, tx_done_after_samples=12, sample_delay_s=0.001)
        try:
            readings, stop_reason, trimmed = collect_measurements_until_stop(
                profiler,
                FakeProcess(),
                notify_socket,
                window_s=1.0,
            )
        finally:
            push_socket.close(0)
            notify_socket.close(0)
        self.assertEqual(stop_reason, "tx_done")
        self.assertEqual(trimmed, 10)
        self.assertEqual(len(readings), 2)

    def test_collect_measurements_tx_done_underflow_trims_all(self) -> None:
        notify_socket, endpoint = create_tx_notify_socket(self.context)
        push_socket = self._create_push_socket(endpoint)
        profiler = FakeProfiler(push_socket=push_socket, tx_done_after_samples=5, sample_delay_s=0.001)
        try:
            readings, stop_reason, trimmed = collect_measurements_until_stop(
                profiler,
                FakeProcess(),
                notify_socket,
                window_s=1.0,
            )
        finally:
            push_socket.close(0)
            notify_socket.close(0)
        self.assertEqual(stop_reason, "tx_done")
        self.assertEqual(trimmed, 5)
        self.assertEqual(readings, [])

    def test_collect_measurements_ignores_tx_done_after_window_elapsed(self) -> None:
        notify_socket, endpoint = create_tx_notify_socket(self.context)
        push_socket = self._create_push_socket(endpoint)
        profiler = FakeProfiler(push_socket=push_socket, tx_done_after_samples=100, sample_delay_s=0.02)
        try:
            readings, stop_reason, trimmed = collect_measurements_until_stop(
                profiler,
                FakeProcess(),
                notify_socket,
                window_s=0.05,
            )
        finally:
            push_socket.close(0)
            notify_socket.close(0)
        self.assertEqual(stop_reason, "window_elapsed")
        self.assertEqual(trimmed, 0)
        self.assertGreater(len(readings), 0)

    def test_tx_event_notifier_emits_started_then_done(self) -> None:
        pull_socket, endpoint = create_tx_notify_socket(self.context)
        notifier = TxEventNotifier(
            endpoint,
            tone=8,
            bw_khz=1000,
            gain_db=80.2,
            iq_file=Path("data/tx_iq/iq_N8_BW1000_TXG80.2db.npz"),
        )
        try:
            notifier.schedule_started(time.monotonic())
            notifier.ensure_started_sent()
            notifier.send_done()

            started = pull_socket.recv_json()
            done = pull_socket.recv_json()
        finally:
            notifier.close()
            pull_socket.close(0)
        self.assertEqual(started["event"], "tx_started")
        self.assertEqual(done["event"], "tx_done")
        self.assertEqual(done["tone"], 8)


if __name__ == "__main__":
    unittest.main()
