#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Static ABI gate for the HDIM=512 contiguous-prefill entries.

`ops::prefill_attention` launches both the generic `inferspark_prefill` and
Gemma-4's HDIM=512 entries with one argument list. `cuLaunchKernel` reads the
parameters by the kernel's own signature, so a stale entry does not fail: it
reads shifted values and silently writes nothing. No CUDA is needed here.
"""

from __future__ import annotations

import pathlib
import unittest

from test_inferspark_prefill_abi import EXPECTED_ARGS, function_parameters

HERE = pathlib.Path(__file__).resolve().parent
GB10 = HERE.parent

WIDE_ENTRIES = {
    "gemma-4-31b": ("inferspark_prefill_512",),
    "gemma-4-26b-a4b": ("inferspark_prefill_512",),
}


class WidePrefillAbiGate(unittest.TestCase):
    def test_scalar_512_entries_match_the_host_launcher(self) -> None:
        for target, symbols in WIDE_ENTRIES.items():
            for symbol in symbols:
                source = (GB10 / target / "nvfp4" / f"{symbol}.cu").read_text()
                with self.subTest(target=target, symbol=symbol):
                    self.assertEqual(function_parameters(source, symbol), EXPECTED_ARGS)

    def test_scalar_512_rows_are_bounded_to_the_requested_query_range(self) -> None:
        for target in WIDE_ENTRIES:
            source = (GB10 / target / "nvfp4" / "inferspark_prefill_512.cu").read_text()
            with self.subTest(target=target):
                self.assertIn(
                    "const unsigned int q_row = query_start + q_block * BR + (tid / 8);",
                    source,
                )
                self.assertIn(
                    "if (q_row >= seq_len || q_row >= query_start + query_len_total) return;",
                    source,
                )

    def test_scalar_512_shuffles_name_only_the_row_group(self) -> None:
        for target in WIDE_ENTRIES:
            source = (GB10 / target / "nvfp4" / "inferspark_prefill_512.cu").read_text()
            with self.subTest(target=target):
                self.assertNotIn("__shfl_xor_sync(0xFFFFFFFF", source)
                self.assertIn("const unsigned int group_mask = 0xFFu << ((tid % 32) & ~7u);", source)
                self.assertEqual(source.count("__shfl_xor_sync(group_mask, dot,"), 3)


if __name__ == "__main__":
    unittest.main()
