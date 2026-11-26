// SPDX-License-Identifier: AGPL-3.0-only

//! Std-only half of [`super`] (`ATLAS_PLE_READ_AFFINITY`): the selector grammar, the CPU
//! classes from `cpu_capacity`, and the per-process plan. `rustc --edition 2024 --test`
//! runs its tests without any dependency.

pub const SELECTOR: &str = "ATLAS_PLE_READ_AFFINITY";

/// A CPU is "big" when its capacity is at least this share of the maximum.
const BIG_SHARE_NUM: u64 = 9;
const BIG_SHARE_DEN: u64 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Observe,
    Big,
    Little,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Observe => "observe",
            Self::Big => "big",
            Self::Little => "little",
        }
    }
}

pub fn parse_mode(value: Option<&str>) -> Result<Mode, String> {
    match value.map(str::trim) {
        None | Some("") | Some("0") | Some("off") => Ok(Mode::Off),
        Some("observe") => Ok(Mode::Observe),
        Some("big") | Some("1") => Ok(Mode::Big),
        Some("little") => Ok(Mode::Little),
        Some(other) => Err(format!(
            "{SELECTOR} must be off, observe, big or little; got {other:?}"
        )),
    }
}

/// Split CPUs into (big, little) by capacity. `capacities[i]` is CPU i's
/// `cpu_capacity` (None = not readable). None when any capacity is missing
/// or the machine is symmetric (no little CPU): there is nothing to choose.
pub fn classes(capacities: &[Option<u64>]) -> Option<(Vec<usize>, Vec<usize>)> {
    if capacities.is_empty() || capacities.iter().any(Option::is_none) {
        return None;
    }
    let caps: Vec<u64> = capacities.iter().map(|c| c.unwrap_or(0)).collect();
    let max = *caps.iter().max()?;
    if max == 0 {
        return None;
    }
    let (mut big, mut little) = (Vec::new(), Vec::new());
    for (cpu, &cap) in caps.iter().enumerate() {
        if cap * BIG_SHARE_DEN >= max * BIG_SHARE_NUM {
            big.push(cpu);
        } else {
            little.push(cpu);
        }
    }
    (!little.is_empty()).then_some((big, little))
}

/// `a,b-c,...` for a receipt.
pub fn cpu_list(cpus: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let start = cpus[i];
        let mut end = start;
        while i + 1 < cpus.len() && cpus[i + 1] == end + 1 {
            i += 1;
            end = cpus[i];
        }
        out.push(if start == end {
            start.to_string()
        } else {
            format!("{start}-{end}")
        });
        i += 1;
    }
    out.join(",")
}

/// What a read thread does, resolved once per process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    Off,
    Observe,
    /// Pin to these CPUs (`mode` is Big or Little).
    Pin {
        mode: Mode,
        cpus: Vec<usize>,
    },
    /// The selector asked for a pin that cannot be honoured; the read runs unpinned.
    Refused {
        mode: Mode,
        reason: String,
    },
}

impl Plan {
    /// The one receipt line for this plan (None when off).
    pub fn receipt(&self) -> Option<String> {
        match self {
            Self::Off => None,
            Self::Observe => Some(format!(
                "PLE_READ_AFFINITY_ENGAGED selector={SELECTOR} mode=observe cpus=unchanged"
            )),
            Self::Pin { mode, cpus } => Some(format!(
                "PLE_READ_AFFINITY_ENGAGED selector={SELECTOR} mode={} cpus={} n={}",
                mode.name(),
                cpu_list(cpus),
                cpus.len()
            )),
            Self::Refused { mode, reason } => Some(format!(
                "PLE_READ_AFFINITY_REFUSED selector={SELECTOR} mode={} reason={reason}",
                mode.name()
            )),
        }
    }

    /// True when per-read placement should be logged.
    pub fn observing(&self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Resolve a plan from the mode, the CPU capacities and the CPUs the process
/// may run on (its affinity mask at resolution time).
pub fn resolve(mode: Mode, capacities: &[Option<u64>], allowed: &[usize]) -> Plan {
    match mode {
        Mode::Off => return Plan::Off,
        Mode::Observe => return Plan::Observe,
        Mode::Big | Mode::Little => {}
    }
    let Some((big, little)) = classes(capacities) else {
        return Plan::Refused {
            mode,
            reason: "no_asymmetric_capacity".into(),
        };
    };
    let class = if mode == Mode::Big { big } else { little };
    let cpus: Vec<usize> = class.into_iter().filter(|c| allowed.contains(c)).collect();
    if cpus.is_empty() {
        return Plan::Refused {
            mode,
            reason: "class_outside_affinity_mask".into(),
        };
    }
    Plan::Pin { mode, cpus }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GB10 as measured (`/sys/devices/system/cpu/cpu*/cpu_capacity`, 2026-10-15).
    const GB10: [u64; 20] = [
        718, 718, 718, 718, 718, 997, 997, 997, 997, 997, 731, 731, 731, 731, 731, 1017, 1017,
        1017, 1017, 1024,
    ];

    fn gb10() -> Vec<Option<u64>> {
        GB10.iter().map(|&c| Some(c)).collect()
    }

    #[test]
    fn parses_modes_strictly() {
        assert_eq!(parse_mode(None), Ok(Mode::Off));
        assert_eq!(parse_mode(Some("")), Ok(Mode::Off));
        assert_eq!(parse_mode(Some("off")), Ok(Mode::Off));
        assert_eq!(parse_mode(Some("0")), Ok(Mode::Off));
        assert_eq!(parse_mode(Some("observe")), Ok(Mode::Observe));
        assert_eq!(parse_mode(Some("big")), Ok(Mode::Big));
        assert_eq!(parse_mode(Some("1")), Ok(Mode::Big));
        assert_eq!(parse_mode(Some("little")), Ok(Mode::Little));
        assert!(parse_mode(Some("fast")).is_err());
        assert!(parse_mode(Some("2")).is_err());
    }

    #[test]
    fn gb10_classes() {
        // X925 (3.9 GHz) = cpu 5-9 and 15-19; A725 (2.8 GHz) = cpu 0-4 and 10-14.
        let (big, little) = classes(&gb10()).unwrap();
        assert_eq!(big, vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]);
        assert_eq!(little, vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
        assert_eq!(cpu_list(&big), "5-9,15-19");
        assert_eq!(cpu_list(&little), "0-4,10-14");
    }

    #[test]
    fn symmetric_or_unreadable_refuses() {
        assert_eq!(classes(&[Some(1024); 8]), None);
        assert_eq!(classes(&[Some(1024), None]), None);
        assert_eq!(classes(&[]), None);
        let all: Vec<usize> = (0..8).collect();
        assert!(matches!(
            resolve(Mode::Big, &[Some(1024); 8], &all),
            Plan::Refused { .. }
        ));
    }

    #[test]
    fn resolve_respects_the_affinity_mask() {
        let all: Vec<usize> = (0..20).collect();
        assert_eq!(
            resolve(Mode::Big, &gb10(), &all),
            Plan::Pin {
                mode: Mode::Big,
                cpus: vec![5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
            }
        );
        // A server started under `taskset -c 0-9` keeps only the big CPUs inside it.
        let first_ten: Vec<usize> = (0..10).collect();
        assert_eq!(
            resolve(Mode::Big, &gb10(), &first_ten),
            Plan::Pin {
                mode: Mode::Big,
                cpus: vec![5, 6, 7, 8, 9]
            }
        );
        // No big CPU allowed: refuse rather than widen the mask.
        let littles: Vec<usize> = vec![0, 1, 2];
        assert!(matches!(
            resolve(Mode::Big, &gb10(), &littles),
            Plan::Refused { .. }
        ));
        assert_eq!(resolve(Mode::Off, &gb10(), &all), Plan::Off);
        assert_eq!(resolve(Mode::Observe, &gb10(), &all), Plan::Observe);
    }

    #[test]
    fn receipts() {
        let all: Vec<usize> = (0..20).collect();
        assert_eq!(resolve(Mode::Off, &gb10(), &all).receipt(), None);
        assert_eq!(
            resolve(Mode::Big, &gb10(), &all).receipt().unwrap(),
            "PLE_READ_AFFINITY_ENGAGED selector=ATLAS_PLE_READ_AFFINITY mode=big cpus=5-9,15-19 n=10"
        );
        assert!(
            resolve(Mode::Big, &[Some(5); 4], &all)
                .receipt()
                .unwrap()
                .starts_with("PLE_READ_AFFINITY_REFUSED")
        );
        assert!(Plan::Observe.observing() && !Plan::Off.observing());
    }

    #[test]
    fn cpu_list_ranges() {
        assert_eq!(cpu_list(&[]), "");
        assert_eq!(cpu_list(&[3]), "3");
        assert_eq!(cpu_list(&[0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
    }
}
