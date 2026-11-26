// SPDX-License-Identifier: AGPL-3.0-only
// Opt-in dependency production; successful records describe actual compiler calls.

use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const POLICY: &str = "ATLAS_NATIVE_TRACE_POLICY";
const SCHEMA: &str = "flux.native-command/1";
const MAX_DEPFILE_BYTES: u64 = 8 << 20;
const MAX_RECORD_BYTES: usize = 1 << 20;

pub fn enabled() -> bool {
    match std::env::var_os(POLICY) {
        None => false,
        Some(value) if value.as_os_str() == OsStr::new(SCHEMA) => true,
        Some(_) => panic!("native command trace HOLD: unknown policy"),
    }
}

pub fn configure() {
    println!("cargo:rerun-if-changed=../../build-support/native_command_trace.rs");
    for key in [
        POLICY,
        "ATLAS_NVCC",
        "ATLAS_CXX",
        "CARGO_TARGET_DIR",
        "FLASHINFER_HOME",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    enabled();
}

fn text(value: &OsStr) -> &str {
    let value = value
        .to_str()
        .expect("native trace requires UTF-8 paths and arguments");
    assert!(!value.contains('\0'), "native trace field contains NUL");
    value
}

fn canonical_existing(path: &Path, directory: bool) -> PathBuf {
    assert!(path.is_absolute(), "native trace requires absolute paths");
    let canonical = fs::canonicalize(path).expect("native trace path is missing");
    assert_eq!(canonical, path, "native trace path is not canonical");
    assert!(
        if directory {
            path.is_dir()
        } else {
            path.is_file()
        },
        "native trace path has the wrong type"
    );
    text(path.as_os_str());
    canonical
}

fn directory_env(key: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var_os(key).expect("native trace directory env missing"));
    canonical_existing(&path, true)
}

pub fn pinned_tool(key: &str) -> PathBuf {
    assert!(enabled(), "pinned native tool requires trace policy");
    let path = PathBuf::from(std::env::var_os(key).expect("native trace tool pin missing"));
    canonical_existing(&path, false)
}

fn append(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn relative(path: &Path, root: &Path) -> String {
    let value = path
        .strip_prefix(root)
        .expect("native trace path escaped its root");
    assert!(
        !value.as_os_str().is_empty(),
        "native trace root cannot be an artifact"
    );
    assert!(
        value
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_))),
        "native trace relative path is not normalized"
    );
    let value = text(value.as_os_str());
    assert!(
        value.len() <= 4096 && !value.contains('\\') && !value.chars().any(char::is_control),
        "native trace relative path is not portable"
    );
    value.to_owned()
}

fn remove_previous(path: &Path) {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            assert!(
                metadata.is_file(),
                "native trace refuses stale nonregular artifact"
            );
            fs::remove_file(path).expect("remove stale native trace artifact");
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("inspect stale native trace artifact: {error}"),
    }
}

fn conflicting_option(argument: &str) -> bool {
    // Match Python's whitespace atoms, including its four extra control separators.
    argument
        .split(|c: char| {
            c == ',' || c == '=' || c.is_whitespace() || ('\u{001c}'..='\u{001f}').contains(&c)
        })
        .any(|value| {
            let value = value.trim_matches(['\"', '\'']);
            [
                "-M",
                "--generate-dependenc",
                "--generate-nonsystem-dependenc",
                "--dependency-",
                "@",
                "-optf",
                "--options-file",
                "-o",
                "--output-file",
            ]
            .iter()
            .any(|prefix| value.starts_with(*prefix))
        })
}

fn forwarding_option(value: &str) -> bool {
    matches!(
        value,
        "-Xcompiler"
            | "--compiler-options"
            | "-Xlinker"
            | "--linker-options"
            | "-Xptxas"
            | "--ptxas-options"
            | "-Xcudafe"
            | "--cudafe-options"
            | "-Xnvlink"
            | "--nvlink-options"
            | "-Xpreprocessor"
            | "--preprocessor-options"
            | "-Xassembler"
            | "-Xarchive"
            | "--archive-options"
    )
}

fn exclusive_file(path: &Path) -> File {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
        .open(path)
        .expect("create exclusive native trace artifact")
}

fn sealed_write(path: &Path, bytes: &[u8]) {
    let mut file = exclusive_file(path);
    file.write_all(bytes).expect("write native trace artifact");
    file.sync_all().expect("sync native trace artifact");
}

pub struct Trace {
    source: PathBuf,
    output: PathBuf,
    depfile: PathBuf,
    record: PathBuf,
    job: PathBuf,
    lock: PathBuf,
    source_root: &'static str,
    source_relative: String,
    output_relative: String,
    depfile_relative: String,
    record_relative: String,
    role: &'static str,
    _claim: File,
}

impl Drop for Trace {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock);
    }
}

pub fn prepare(
    command: &mut Command,
    source: &Path,
    output: &Path,
    role: &'static str,
) -> Option<Trace> {
    if !enabled() {
        return None;
    }
    assert!(
        matches!(role, "nvcc" | "cxx"),
        "unknown native trace tool role"
    );
    let source = canonical_existing(source, false);
    assert!(
        output.is_absolute(),
        "native trace requires absolute output"
    );
    canonical_existing(output.parent().expect("native output parent missing"), true);
    assert!(
        matches!(
            output.extension().and_then(OsStr::to_str),
            Some("ptx" | "o")
        ),
        "native trace accepts only generated PTX or object outputs"
    );
    let target_root = directory_env("CARGO_TARGET_DIR");
    let manifest = directory_env("CARGO_MANIFEST_DIR");
    let checkout = canonical_existing(
        manifest
            .parent()
            .and_then(Path::parent)
            .expect("native trace crate is outside checkout"),
        true,
    );
    let (source_root, source_relative) = if source.starts_with(&checkout) {
        ("source", relative(&source, &checkout))
    } else {
        let flashinfer = directory_env("FLASHINFER_HOME");
        ("flashinfer", relative(&source, &flashinfer))
    };
    canonical_existing(Path::new(command.get_program()), false);
    let args: Vec<&OsStr> = command.get_args().collect();
    assert!(
        !args.is_empty() && args.len() + 3 <= 4096,
        "native trace argument inventory exceeded bound"
    );
    assert_eq!(
        args.iter()
            .filter(|value| **value == source.as_os_str())
            .count(),
        1,
        "native trace source must occur once in actual arguments"
    );
    let outputs: Vec<usize> = args
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == OsStr::new("-o")).then_some(index))
        .collect();
    assert_eq!(
        outputs.len(),
        1,
        "native trace needs exactly one output option"
    );
    assert_eq!(
        args.get(outputs[0] + 1).copied(),
        Some(output.as_os_str()),
        "native trace output differs from actual arguments"
    );
    for (index, arg) in args.iter().enumerate() {
        let value = text(arg);
        assert!(
            !value.is_empty() && value.len() <= 8192,
            "native trace argument field exceeded bound"
        );
        let canonical_output =
            index == outputs[0] && (index == 0 || !forwarding_option(text(args[index - 1])));
        assert!(
            canonical_output || !conflicting_option(value),
            "native trace refuses conflicting or hidden dependency/output overrides"
        );
    }
    let depfile = append(output, ".d");
    let record = append(output, ".native-command");
    let job = append(output, ".native-job.json");
    let lock = append(output, ".native-trace.lock");
    let claim = exclusive_file(&lock);
    let trace = Trace {
        source,
        output: output.to_owned(),
        depfile_relative: relative(&depfile, &target_root),
        record_relative: relative(&record, &target_root),
        output_relative: relative(output, &target_root),
        depfile,
        record,
        job,
        lock,
        source_root,
        source_relative,
        role,
        _claim: claim,
    };
    for path in [&trace.record, &trace.job, &trace.depfile] {
        remove_previous(path);
    }
    command.arg("-MD").arg("-MF").arg(&trace.depfile);
    Some(trace)
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for c in value.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            c if c < ' ' => result.push_str(&format!("\\u{:04x}", c as u32)),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

pub fn finish(trace: Option<Trace>, command: &Command, success: bool) {
    let Some(trace) = trace else {
        return;
    };
    if !success {
        return;
    }
    let metadata =
        fs::symlink_metadata(&trace.depfile).expect("compiler did not emit dependency file");
    assert!(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= MAX_DEPFILE_BYTES,
        "compiler dependency file must be real, regular and bounded"
    );
    assert!(
        fs::symlink_metadata(&trace.output)
            .expect("native output missing")
            .is_file(),
        "native output is not regular"
    );
    let fields = [
        SCHEMA,
        text(command.get_program()),
        text(trace.source.as_os_str()),
        text(trace.output.as_os_str()),
        text(trace.depfile.as_os_str()),
        "0",
    ];
    let mut record = Vec::new();
    for field in fields.into_iter().chain(command.get_args().map(text)) {
        record.extend_from_slice(field.as_bytes());
        record.push(0);
        assert!(
            record.len() <= MAX_RECORD_BYTES,
            "native command record exceeded bound"
        );
    }
    let job = format!(
        "{{\"schema\":\"flux.native-job/1\",\"output\":{},\"source\":{{\"root\":{},\"path\":{}}},\"depfile\":{},\"tool_role\":{},\"command_record\":{}}}\n",
        json_string(&trace.output_relative),
        json_string(trace.source_root),
        json_string(&trace.source_relative),
        json_string(&trace.depfile_relative),
        json_string(trace.role),
        json_string(&trace.record_relative)
    );
    sealed_write(&trace.job, job.as_bytes());
    sealed_write(&trace.record, &record);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};

    pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Environment<'a> {
        _guard: &'a MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<OsString>)>,
    }

    impl<'a> Environment<'a> {
        fn set(
            guard: &'a MutexGuard<'static, ()>,
            values: &[(&'static str, Option<&OsStr>)],
        ) -> Self {
            let saved = values
                .iter()
                .map(|(key, _)| (*key, std::env::var_os(key)))
                .collect();
            for (key, value) in values {
                match value {
                    Some(value) => {
                        // SAFETY: Caller holds `ENV_LOCK` via `guard` for the entire lifetime
                        // of `Environment`, serializing all process-env mutations in tests.
                        unsafe { std::env::set_var(key, value) };
                    }
                    None => {
                        // SAFETY: Caller holds `ENV_LOCK` via `guard` for the entire lifetime
                        // of `Environment`, serializing all process-env mutations in tests.
                        unsafe { std::env::remove_var(key) };
                    }
                }
            }
            Self {
                _guard: guard,
                saved,
            }
        }
    }

    impl Drop for Environment<'_> {
        fn drop(&mut self) {
            for (key, value) in &self.saved {
                match value {
                    Some(value) => {
                        // SAFETY: `_guard` proves `ENV_LOCK` is still held until after
                        // `Environment::drop` finishes restoring the saved process env.
                        unsafe { std::env::set_var(key, value) };
                    }
                    None => {
                        // SAFETY: `_guard` proves `ENV_LOCK` is still held until after
                        // `Environment::drop` finishes restoring the saved process env.
                        unsafe { std::env::remove_var(key) };
                    }
                }
            }
        }
    }

    pub(crate) fn with_policy(value: Option<&str>, test: impl FnOnce()) {
        let guard = ENV_LOCK.lock().unwrap();
        let _environment = Environment::set(&guard, &[(POLICY, value.map(OsStr::new))]);
        test();
    }

    struct Fixture<'a> {
        root: PathBuf,
        source: PathBuf,
        output: PathBuf,
        _environment: Environment<'a>,
    }

    impl<'a> Fixture<'a> {
        fn new(guard: &'a MutexGuard<'static, ()>) -> Self {
            let root = std::env::temp_dir().join(format!(
                "flux-native-source-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let root = root.canonicalize().unwrap();
            let manifest = root.join("source/crates/fixture");
            let target = root.join("target");
            fs::create_dir_all(&manifest).unwrap();
            fs::create_dir(&target).unwrap();
            let source = manifest.join("fixture.cu");
            fs::write(&source, b"// Synthetic unit-test source, never compiled.\n").unwrap();
            let environment = Environment::set(
                guard,
                &[
                    (POLICY, Some(OsStr::new(SCHEMA))),
                    ("CARGO_TARGET_DIR", Some(target.as_os_str())),
                    ("CARGO_MANIFEST_DIR", Some(manifest.as_os_str())),
                ],
            );
            Self {
                source,
                output: target.join("fixture.ptx"),
                root,
                _environment: environment,
            }
        }

        fn command(&self) -> Command {
            // The test process path is a canonical regular file; it is never executed.
            let mut command =
                Command::new(std::env::current_exe().unwrap().canonicalize().unwrap());
            command
                .arg("--ptx")
                .arg(&self.source)
                .arg("-o")
                .arg(&self.output);
            command
        }
    }

    impl Drop for Fixture<'_> {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn identities_append_without_collapsing_output_suffixes() {
        assert_eq!(
            append(Path::new("/target/kernel.ptx"), ".d"),
            Path::new("/target/kernel.ptx.d")
        );
        assert_eq!(
            append(Path::new("/target/kernel.o"), ".d"),
            Path::new("/target/kernel.o.d")
        );
        assert_ne!(
            append(Path::new("/target/kernel.ptx"), ".native-command"),
            append(Path::new("/target/kernel.o"), ".native-command")
        );
    }

    #[test]
    fn inventory_strings_escape_json_controls() {
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\u000a\"");
    }

    #[test]
    fn disabled_prepare_does_not_inspect_paths_or_change_real_command() {
        with_policy(None, || {
            let mut command = Command::new("not-a-real-tool");
            command.arg("unchanged");
            assert!(
                prepare(
                    &mut command,
                    Path::new("missing"),
                    Path::new("missing"),
                    "nvcc"
                )
                .is_none()
            );
            assert_eq!(
                command.get_args().collect::<Vec<_>>(),
                vec![OsStr::new("unchanged")]
            );
        });
    }

    #[test]
    fn real_prepare_and_finish_own_stale_failure_and_success_lifecycle() {
        let guard = ENV_LOCK.lock().unwrap();
        let fixture = Fixture::new(&guard);
        let depfile = append(&fixture.output, ".d");
        let record = append(&fixture.output, ".native-command");
        let job = append(&fixture.output, ".native-job.json");
        let lock = append(&fixture.output, ".native-trace.lock");
        for path in [&depfile, &record, &job] {
            fs::write(path, b"stale synthetic unit-test artifact").unwrap();
        }
        let mut command = fixture.command();
        let trace = prepare(&mut command, &fixture.source, &fixture.output, "nvcc");
        assert!(lock.is_file());
        for path in [&depfile, &record, &job] {
            assert!(!path.exists());
        }
        let mut duplicate = fixture.command();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prepare(&mut duplicate, &fixture.source, &fixture.output, "nvcc")
            }))
            .is_err()
        );
        assert!(lock.is_file());
        finish(trace, &command, false);
        for path in [&record, &job, &lock] {
            assert!(!path.exists());
        }

        let mut command = fixture.command();
        let trace = prepare(&mut command, &fixture.source, &fixture.output, "nvcc");
        // Synthetic test payloads exercise the wrapper; they are never compiler evidence.
        fs::write(&depfile, b"synthetic: synthetic.cu\n").unwrap();
        fs::write(&fixture.output, b"synthetic PTX placeholder\n").unwrap();
        finish(trace, &command, true);
        assert!(!lock.exists());
        let bytes = fs::read(&record).unwrap();
        let fields: Vec<_> = bytes.split(|value| *value == 0).collect();
        assert_eq!(fields[0], SCHEMA.as_bytes());
        let actual: Vec<_> = command.get_args().map(|arg| text(arg).as_bytes()).collect();
        assert_eq!(&fields[6..fields.len() - 1], actual.as_slice());
        assert!(
            fs::read_to_string(&job)
                .unwrap()
                .contains("fixture.ptx.native-command")
        );
    }

    #[test]
    fn real_finish_rejects_missing_or_oversized_depfile_without_success_records() {
        let guard = ENV_LOCK.lock().unwrap();
        let fixture = Fixture::new(&guard);
        for oversized in [false, true] {
            let mut command = fixture.command();
            let trace = prepare(&mut command, &fixture.source, &fixture.output, "nvcc");
            fs::write(&fixture.output, b"synthetic PTX placeholder\n").unwrap();
            if oversized {
                let file = File::create(append(&fixture.output, ".d")).unwrap();
                file.set_len(MAX_DEPFILE_BYTES + 1).unwrap();
            }
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    finish(trace, &command, true)
                }))
                .is_err()
            );
            for suffix in [".native-command", ".native-job.json", ".native-trace.lock"] {
                assert!(!append(&fixture.output, suffix).exists());
            }
        }
    }

    #[test]
    fn real_prepare_rejects_forwarded_hidden_and_alias_overrides_before_claim() {
        let guard = ENV_LOCK.lock().unwrap();
        let fixture = Fixture::new(&guard);
        for value in [
            "-M",
            "-MM",
            "-MD",
            "-MMD",
            "-MG",
            "-MP",
            "-MTother",
            "-MQother",
            "-MFother",
            "--generate-dependencies-with-compile",
            "--dependency-output=other",
            "--generate-nonsystem-dependencies",
            "--dependency-target-name=other",
            "-Xcompiler=-MF,other",
            "-Wp,-MD,other",
            "@options",
            "-optf=options",
            "--options-file=options",
            "--output-file=other",
            "-oother",
            "-Xcompiler='-MF,other'",
            "-Wp,\"-MD\",other",
            "-Xcompiler=-fPIC -MF other",
            "-Wp,-fPIC\t-MD,other",
            "-Xcompiler=-fPIC\u{001c}-MF other",
        ] {
            let mut command = fixture.command();
            command.arg(value);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prepare(&mut command, &fixture.source, &fixture.output, "nvcc")
                }))
                .is_err(),
                "accepted conflicting argument {value}"
            );
            assert!(!append(&fixture.output, ".native-trace.lock").exists());
        }
    }

    #[test]
    fn real_prepare_rejects_output_option_used_only_as_forwarder_operand() {
        let guard = ENV_LOCK.lock().unwrap();
        let fixture = Fixture::new(&guard);
        for forwarder in [
            "-Xcompiler",
            "--compiler-options",
            "-Xlinker",
            "--linker-options",
            "-Xptxas",
            "--ptxas-options",
            "-Xcudafe",
            "--cudafe-options",
            "-Xnvlink",
            "--nvlink-options",
            "-Xpreprocessor",
            "--preprocessor-options",
            "-Xassembler",
            "-Xarchive",
            "--archive-options",
        ] {
            let original = fixture.command();
            let mut args: Vec<_> = original.get_args().map(OsStr::to_os_string).collect();
            let output_index = args
                .iter()
                .position(|arg| arg.as_os_str() == OsStr::new("-o"))
                .unwrap();
            args.insert(output_index, OsString::from(forwarder));
            let mut command = Command::new(original.get_program());
            command.args(args);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    prepare(&mut command, &fixture.source, &fixture.output, "nvcc")
                }))
                .is_err(),
                "accepted forwarded output option through {forwarder}"
            );
            assert!(!append(&fixture.output, ".native-trace.lock").exists());
        }
    }
}
