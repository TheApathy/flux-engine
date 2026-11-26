# SPDX-License-Identifier: AGPL-3.0-only
"""Offline hostile checks for checkpoint, scheduler and endpoint authority."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import test_r15_release_hold_regressions as hold_fixture

S = hold_fixture.S


class LiveProcessBrokerRegressions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (
            self.proc_root,
            self.process,
            self.plan,
            self.identities,
            self.document,
        ) = hold_fixture.ReleaseHoldRegressions().fake_process(self.root)
        self.argv = (self.process / "cmdline").read_bytes()
        self.environment = (self.process / "environ").read_bytes()
        self.tcp = (self.process / "net/tcp").read_text()
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(S, "PROC_ROOT", self.proc_root))
        stack.enter_context(
            mock.patch.object(
                S.os, "pidfd_open", side_effect=lambda _pid: os.dup(read_fd)
            )
        )
        stack.enter_context(
            mock.patch.object(
                S.subprocess,
                "run",
                side_effect=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    [], 0, json.dumps(self.document).encode(), b""
                ),
            )
        )

    def verify(self, mode="native-mtp-k5"):
        return S.verify_live_server(
            {
                "schema": "flux-r15-live-process-request/1",
                "mode": mode,
                "host_pid": 123,
            },
            self.plan,
            self.identities,
        )

    def bridge_network(self):
        host_namespace = self.proc_root / "self/ns/net"
        host_namespace.unlink()
        host_namespace.write_bytes(b"separate host network namespace")
        self.document["HostConfig"]["NetworkMode"] = "bridge"
        self.document["NetworkSettings"] = {
            "Ports": {"8888/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18888"}]}
        }
        self.identities["base_url"] = "http://127.0.0.1:18888"
        (self.process / "net/tcp").write_text(self.tcp.replace("0100007F", "00000000"))

    def test_native_and_actual_loaded_serial_control_pass(self):
        self.assertEqual(self.verify()["status"], "PASS")
        (self.process / "environ").write_bytes(
            self.environment + b"ATLAS_DISABLE_SPECULATION=1\0"
        )
        self.assertEqual(self.verify("none")["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "native K5"):
            self.verify()

    def test_launcher_only_disable_does_not_prove_plain(self):
        for assignment in (
            b"FLUX_SPEC_MODE=none",
            b"FLUX_SPEC_MODE=off",
            b"FLUX_SPEC_MODE=0",
            b"FLUX_SPEC_MODE=false",
        ):
            (self.process / "environ").write_bytes(
                self.environment + assignment + b"\0"
            )
            with self.subTest(assignment=assignment), self.assertRaisesRegex(
                ValueError, "requires effective ATLAS_DISABLE_SPECULATION=1"
            ):
                self.verify("none")
        (self.process / "environ").write_bytes(self.environment)
        (self.process / "cmdline").write_bytes(self.argv + b"--no-speculative\0")
        with self.assertRaisesRegex(
            ValueError, "requires effective ATLAS_DISABLE_SPECULATION=1"
        ):
            self.verify("none")

    def test_actual_kill_switch_is_strict(self):
        for value in (b"", b"true", b"2", b"01"):
            (self.process / "environ").write_bytes(
                self.environment + b"ATLAS_DISABLE_SPECULATION=" + value + b"\0"
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "malformed"
            ):
                self.verify()
        (self.process / "environ").write_bytes(
            self.environment + b"ATLAS_DISABLE_SPECULATION=0\0"
        )
        self.assertEqual(self.verify()["status"], "PASS")

    def test_actual_kill_switch_alias_matches_binary_precedence(self):
        (self.process / "environ").write_bytes(
            self.environment + b"FLUX_DISABLE_SPECULATION=1\0"
        )
        self.assertEqual(self.verify("none")["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "native K5"):
            self.verify()
        (self.process / "environ").write_bytes(
            self.environment
            + b"FLUX_DISABLE_SPECULATION=1\0ATLAS_DISABLE_SPECULATION=0\0"
        )
        self.assertEqual(self.verify()["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "requires effective"):
            self.verify("none")

    def test_mutable_tui_cannot_change_the_selected_checkpoint(self):
        (self.process / "cmdline").write_bytes(self.argv.replace(b"--no-tui\0", b""))
        with self.assertRaisesRegex(ValueError, "prohibit model swaps"):
            self.verify()

    def test_gating_and_retrieval_cannot_replace_native_proposals(self):
        for selector in (
            b"--mtp-gate\0auto\0",
            b"--mtp-gate=mtp\0",
            b"--mtp-gate=force\0",
        ):
            (self.process / "cmdline").write_bytes(self.argv + selector)
            with self.subTest(selector=selector), self.assertRaisesRegex(
                ValueError, "native K5"
            ):
                self.verify()
        (self.process / "cmdline").write_bytes(self.argv)
        for assignment in (
            b"ATLAS_REST_STORE=/results/foreign",
            b"FLUX_REST_STORE=/results/foreign",
            b"ATLAS_REST_STORE=",
            b"ATLAS_SELF_CONTEXT_DRAFT=1",
            b"FLUX_SELF_CONTEXT_DRAFT=1",
        ):
            (self.process / "environ").write_bytes(
                self.environment + assignment + b"\0"
            )
            with self.subTest(assignment=assignment), self.assertRaisesRegex(
                ValueError, "native K5"
            ):
                self.verify()

    def test_every_foreign_checkpoint_selector_is_rejected_in_both_modes(self):
        selectors = []
        for option in (b"--mtp-from-path", b"--draft-model", b"--dflash-donor-model"):
            selectors.extend(
                (option + b"\0/results/foreign\0", option + b"=/results/foreign\0")
            )
        for option in (b"--dflash", b"--self-speculative", b"--ngram-speculative"):
            selectors.extend((option + b"\0", option + b"=true\0"))
        for selector in selectors:
            (self.process / "cmdline").write_bytes(self.argv + selector)
            for mode in ("none", "native-mtp-k5"):
                (self.process / "environ").write_bytes(
                    self.environment
                    + (b"ATLAS_DISABLE_SPECULATION=1\0" if mode == "none" else b"")
                )
                with self.subTest(selector=selector, mode=mode), self.assertRaisesRegex(
                    ValueError, "native K5"
                ):
                    self.verify(mode)

    def test_selected_checkpoint_cannot_be_spoofed_by_display_alias(self):
        for selected in (b"/results/foreign", b"/models/subdir", b"/models/../foreign"):
            (self.process / "cmdline").write_bytes(
                self.argv.replace(b"/models\0", selected + b"\0")
                + b"--served-model-name\0claimed-checkpoint\0"
            )
            with self.subTest(selected=selected), self.assertRaisesRegex(
                ValueError, "pinned --model-from-path"
            ):
                self.verify()
        (self.process / "cmdline").write_bytes(
            self.argv.replace(
                b"--model-from-path\0/models\0", b"--model-name\0claimed-checkpoint\0"
            )
        )
        with self.assertRaisesRegex(ValueError, "pinned --model-from-path"):
            self.verify()

    def test_split_and_equals_checkpoint_and_k5_are_unambiguous(self):
        (self.process / "cmdline").write_bytes(
            self.argv.replace(
                b"--model-from-path\0/models\0", b"--model-from-path=/models\0"
            ).replace(b"--num-drafts\x004\0", b"--num-drafts=4\0")
        )
        self.assertEqual(self.verify()["status"], "PASS")
        for suffix in (b"--model-from-path=/models\0", b"--model-from-path\0/models\0"):
            (self.process / "cmdline").write_bytes(self.argv + suffix)
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                ValueError, "pinned --model-from-path"
            ):
                self.verify()
        for suffix in (b"--num-drafts=4\0", b"--num-drafts\x004\0"):
            (self.process / "cmdline").write_bytes(self.argv + suffix)
            with self.subTest(suffix=suffix), self.assertRaisesRegex(
                ValueError, "native K5"
            ):
                self.verify()
        (self.process / "cmdline").write_bytes(b"server\0--\0" + self.argv)
        with self.assertRaisesRegex(ValueError, "pinned --model-from-path"):
            self.verify()

    def test_checkpoint_bind_must_be_read_only_and_unique(self):
        mount = self.document["Mounts"][0]
        for field, value in (
            ("RW", True),
            ("Type", "volume"),
            ("Destination", "/other"),
        ):
            original = mount[field]
            mount[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "read-only /models bind"
            ):
                self.verify()
            mount[field] = original
        self.document["Mounts"].append(dict(mount))
        with self.assertRaisesRegex(ValueError, "read-only /models bind"):
            self.verify()

    def test_checkpoint_bind_inode_must_match_actual_process_root(self):
        other = self.root / "other-checkpoint"
        other.mkdir()
        self.document["Mounts"][0]["Source"] = str(other)
        with self.assertRaisesRegex(ValueError, "differs from the pinned /models bind"):
            self.verify()

    def test_checkpoint_mount_rejects_writable_nested_and_missing_mappings(self):
        path = self.process / "mountinfo"
        original = path.read_text()
        for mountinfo in (
            original.replace("/models ro", "/models rw"),
            original.splitlines()[0] + "\n",
            original + "3 2 0:3 / /models/config.json ro - ext4 /dev/other ro\n",
        ):
            path.write_text(mountinfo)
            with self.subTest(mountinfo=mountinfo), self.assertRaisesRegex(
                ValueError, "writable mount|nested overrides"
            ):
                self.verify()

    def test_checkpoint_mount_rejects_process_and_host_symlinks(self):
        models = self.root / "container-root/models"
        replacement = self.root / "other-checkpoint"
        replacement.mkdir()
        models.rmdir()
        models.symlink_to(replacement, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.verify()
        models.unlink()
        models.mkdir()
        alias = self.root / "checkpoint-alias"
        alias.symlink_to(models, target_is_directory=True)
        self.document["Mounts"][0]["Source"] = str(alias)
        with self.assertRaisesRegex(ValueError, "symlinked component"):
            self.verify()

    def test_endpoint_socket_must_belong_to_this_pid(self):
        path = self.process / "fd/7"
        path.unlink()
        path.symlink_to("socket:[9002]")
        with self.assertRaisesRegex(ValueError, "uniquely owned"):
            self.verify()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "uniquely owned"):
            self.verify()

    def test_endpoint_requires_one_listener_on_its_actual_address_and_port(self):
        path = self.process / "net/tcp"
        for table in (
            self.tcp.replace("22B8", "22B9"),
            self.tcp.replace("0100007F", "0200007F"),
            self.tcp.replace(" 0A ", " 01 "),
            self.tcp + self.tcp.splitlines()[1].replace("9001", "9002") + "\n",
        ):
            path.write_text(table)
            with self.subTest(table=table), self.assertRaisesRegex(
                ValueError, "uniquely owned"
            ):
                self.verify()

    def test_endpoint_requires_process_container_and_host_network_linkage(self):
        path = self.process / "ns/net"
        path.unlink()
        path.write_bytes(b"foreign network namespace")
        with self.assertRaisesRegex(ValueError, "network namespace differs"):
            self.verify()
        path.unlink()
        os.link(self.proc_root / "456/ns/net", path)
        self.identities["base_url"] = "http://127.0.0.1:18888"
        with self.assertRaisesRegex(ValueError, "host-network endpoint differs"):
            self.verify()

    def test_published_bridge_endpoint_is_supported(self):
        self.bridge_network()
        self.assertEqual(self.verify()["status"], "PASS")
        self.document["NetworkSettings"]["Ports"]["8888/udp"] = [
            {"HostIp": "127.0.0.1", "HostPort": "18888"}
        ]
        self.assertEqual(self.verify()["status"], "PASS")
        self.document["NetworkSettings"]["Ports"]["8888/tcp"][0]["HostIp"] = "0.0.0.0"
        self.assertEqual(self.verify()["status"], "PASS")

    def test_published_bridge_cannot_claim_another_or_ambiguous_endpoint(self):
        self.bridge_network()
        for mappings in (
            {},
            {"9999/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18888"}]},
            {"8888/tcp": [{"HostIp": "127.0.0.2", "HostPort": "18888"}]},
            {"8888/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18888"}] * 2},
        ):
            self.document["NetworkSettings"]["Ports"] = mappings
            with self.subTest(mappings=mappings), self.assertRaisesRegex(
                ValueError, "published endpoint differs"
            ):
                self.verify()

    def test_published_bridge_must_listen_on_reachable_wildcard(self):
        self.bridge_network()
        (self.process / "net/tcp").write_text(self.tcp)
        with self.assertRaisesRegex(ValueError, "uniquely owned"):
            self.verify()

    def test_listener_rejects_ambiguous_ipv6_endpoint(self):
        path = self.process / "net/tcp6"
        path.write_text(
            path.read_text()
            + "0: "
            + "0" * 32
            + ":22B8 "
            + "0" * 32
            + ":0000 0A 0:0 00:0 0 1000 0 9002 1\n"
        )
        with self.assertRaisesRegex(ValueError, "ambiguous IPv6 listener"):
            self.verify()

    def test_live_selectors_and_checkpoint_are_rechecked_after_listener_proof(self):
        original = S._live_listener_identity
        for target in ("cmdline", "environ", "checkpoint"):
            mutated = [False]

            def mutate(*args, **kwargs):
                proof = original(*args, **kwargs)
                if mutated[0]:
                    return proof
                mutated[0] = True
                if target == "checkpoint":
                    models = self.root / "container-root/models"
                    models.rename(models.with_name("old-models"))
                    models.mkdir()
                else:
                    path = self.process / target
                    path.write_bytes(path.read_bytes() + b"changed\0")
                return proof

            with self.subTest(target=target), mock.patch.object(
                S, "_live_listener_identity", side_effect=mutate
            ):
                with self.assertRaisesRegex(ValueError, "changed during verification"):
                    self.verify()
            if target in ("cmdline", "environ"):
                (self.process / target).write_bytes(
                    self.argv if target == "cmdline" else self.environment
                )

    def test_process_reads_are_bounded(self):
        path = self.process / "cmdline"
        path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "command line is oversized"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
