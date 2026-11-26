# Fast-Loader Reader Descriptor Lifetime

Exact base: `7c94d3a7d75d900908d43fd916da0bfa20825168`.

This is a separate source-only follow-up to the pinned bounce-buffer repair.
It does not authorize canonical integration, builds, native execution, model,
package, GPU, server, resource claims or process signals. The live diagnostic
and immutable bounce-buffer review lane remain untouched.

## Repair

The inherited background reader captured only a raw integer descriptor owned
by files on the calling thread. An error could detach that reader and close
both original files. The descriptor could then become invalid or be reused
for a different file while the reader continued issuing `pread` calls.

The loader now duplicates the exact selected `File` with `try_clone` before
spawning. The moved closure owns that duplicate, and derives its raw descriptor
inside the closure. It keeps that owner until the closure returns or unwinds.
Original-file cleanup cannot close the duplicate. There is no pathname reopen,
so the selected file identity and `O_DIRECT` selection remain unchanged. The
synchronous scalar-slab read continues borrowing the original selected file.

No error-path join, blocking stream synchronization or thread cancellation is
introduced. Receiver disconnection still ends the reader at its next channel
send. A `pread` already in progress can still outlive the calling load attempt.
This repair does not prove reader termination, bounded kernel I/O, runtime
loading, accuracy, performance, container qualification or release readiness.

## Verification

Six Python source-wiring checks cover selection, duplication before spawn,
closure-owned descriptor extraction, scalar reads, unchanged I/O mode,
unchanged success-only join and one native ownership-pattern regression.
The native definition uses four bytes and an unlinked regular file, closes the
parent owner before releasing the worker and calls the real aligned reader.
It is not a full loader-error injection or an `O_DIRECT` filesystem test.
All Rust/native definitions remain uncompiled and unrun.

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s repro-kit/tests -p
test_fast_reader_fd_source.py -v` passes all six source checks in 0.000 seconds.
Measured maximum RSS with `/usr/bin/time -v` is 15,004 KiB. Scoped
`rustfmt --check` and `git diff --check` PASS. These are source-only checks, not
compiler or native results. Independent review and a separately authorized
compiler/native window remain required before integration or runtime testing.
