"""Reader-fd source wiring checks, not native loader or thread execution."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FastReaderFdSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "crates/spark-runtime/src/fast_weights/mod.rs").read_text()
        cls.loader = cls.source.split("fn load_shard_fast(", 1)[1].split("#[cfg(test)]", 1)[0]
        cls.reader = cls.loader.split("std::thread::spawn(move || {", 1)[1].split(
            "// Copier: drains the channel", 1
        )[0]

    def test_exact_selected_file_is_duplicated_before_spawning(self):
        selected = "let data_fd = direct_file.as_ref().unwrap_or(&buffered_file)"
        duplicate = "let reader_file = data_fd .try_clone()"
        loader = " ".join(self.loader.split())
        self.assertLess(loader.index(selected), loader.index(duplicate))
        self.assertLess(loader.index(duplicate), loader.index("std::thread::spawn(move || {"))
        self.assertIn('.context("failed to duplicate fast-loader reader descriptor")?', loader)

    def test_reader_derives_raw_descriptor_from_its_owned_file_inside_move_closure(self):
        raw = "let raw_fd = reader_file.as_raw_fd()"
        self.assertIn(raw, self.reader)
        self.assertLess(self.reader.index(raw), self.reader.index("direct_io::read_tensor_aligned"))
        before_spawn = self.loader.split("std::thread::spawn(move || {", 1)[0]
        self.assertNotIn("let raw_fd =", before_spawn)
        self.assertNotIn("drop(reader_file)", self.reader)
        self.assertNotIn("into_raw_fd", self.reader)

    def test_synchronous_scalar_read_keeps_its_borrowed_descriptor(self):
        self.assertIn("let scalar_raw_fd =", self.loader)
        self.assertIn("read_expert_scalar_slab(scalar_raw_fd, using_direct", self.loader)
        self.assertLess(self.loader.index("read_expert_scalar_slab"), self.loader.index("let reader_file"))

    def test_reader_does_not_reopen_path_or_change_io_mode(self):
        self.assertNotIn("File::open", self.reader)
        self.assertNotIn("open_direct", self.reader)
        # Coalesced runs since 2bce3655e (v9.15.2 16 MiB fast loader): same fd and
        # I/O mode, a run's (offset, length) instead of one tensor's.
        self.assertIn("read_tensor_aligned(raw_fd, run_start, run_len, using_direct)", self.reader)
        # A dropped receiver still stops the reader (per tensor of the run).
        self.assertIn(".is_err()\n                        {\n                            send_failed = true;\n                            break;", self.reader)

    def test_no_error_path_join_or_blocking_synchronization_is_added(self):
        self.assertEqual(self.loader.count(".join()"), 1)
        self.assertLess(self.loader.index("bounce.drain()?"), self.loader.index(".join()"))
        for forbidden in ("scope(", "synchronize(", "process::exit", "process::abort"):
            self.assertNotIn(forbidden, self.loader)

    def test_native_ownership_pattern_regression_is_defined_not_executed(self):
        native = self.source.split("fn reader_owned_clone_survives_parent_drop_and_path_removal", 1)[1].split(
            "fn meta(", 1
        )[0]
        self.assertIn("let reader_file = parent_file.try_clone()?", native)
        self.assertLess(native.index("remove_file(&path)?"), native.index("std::thread::spawn"))
        self.assertIn("let raw_fd = reader_file.as_raw_fd()", native)
        self.assertLess(native.index("drop(parent_file)"), native.index("ready.send(())?"))
        self.assertIn("wait.recv_timeout", native)
        self.assertIn("direct_io::read_tensor_aligned(raw_fd, 0, 4, false)?", native)


if __name__ == "__main__":
    unittest.main()
