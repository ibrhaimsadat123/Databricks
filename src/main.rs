// Shopping Sessions pipeline (v45). Lock-free general path + 192B SIMD + MonTab + pq.rs.
use memchr::memchr;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::{LazyLock, Mutex};

mod fast;
mod pq;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


/// Tuning knobs (environment overrides, used by on-scorer experiments).
fn knob(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
static K_MMAP: LazyLock<bool> = LazyLock::new(|| knob("SESS_MMAP", 0) != 0);
static K_POP: LazyLock<bool> = LazyLock::new(|| knob("SESS_POP", 1) != 0);
static K_NBA: LazyLock<usize> = LazyLock::new(|| knob("SESS_NBA", 512).next_power_of_two().clamp(16, 4096));
static K_NBD: LazyLock<usize> = LazyLock::new(|| knob("SESS_NBD", 512).next_power_of_two().clamp(16, 4096));
static K_NOCLK: LazyLock<bool> = LazyLock::new(|| knob("SESS_NOCLK", 0) != 0);

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 256;

// ---------- keys & lock-free long ID interning ----------
// Strings of at most 9 bytes, every byte in 1..=127, are packed 7 bits per byte
// (big-endian, 0 = padding) into 63 bits (bit 63 == 0), which preserves byte-wise order.
// Long/non-ASCII strings get FALLBACK (bit 63 == 1) and are stored in lock-free bump arenas.
const FALLBACK: u64 = 1 << 63;
const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const HI8: u64 = 0x8080_8080_8080_8080;
const ONES: u64 = 0x0101_0101_0101_0101;

#[inline(always)]
fn pext7(x: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        return std::arch::x86_64::_pext_u64(x, LOW7);
    }
    #[allow(unreachable_code)]
    {
        let mut v = 0u64;
        for i in (0..8).rev() {
            v = (v << 7) | ((x >> (8 * i)) & 0x7F);
        }
        v
    }
}

/// x = first 8 bytes little-endian with bytes past len zeroed; c9 = 9th byte or 0.
#[inline(always)]
pub(crate) fn pack(x: u64, c9: u8, len: usize) -> Option<u64> {
    // valid bytes: 1..=127 within len
    let lm = if len >= 8 { u64::MAX } else { (1u64 << (8 * len)) - 1 };
    let t = x | (!lm & ONES);
    let zero = (t.wrapping_sub(ONES)) & !t & HI8;
    if (x & HI8) != 0 || zero != 0 || c9 >= 128 || (len == 9 && c9 == 0) {
        return None;
    }
    Some((pext7(x.swap_bytes()) << 7) | c9 as u64)
}

#[inline(always)]
fn key_short(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n <= 9 {
        let mut b = [0u8; 9];
        b[..n].copy_from_slice(s);
        let x = u64::from_le_bytes(b[..8].try_into().unwrap());
        return pack(x, b[8], n);
    }
    None
}

/// Same as key_short, but may read up to 9 bytes past the start of s (caller guarantees padding).
#[inline(always)]
unsafe fn key_short_fast(s: &[u8]) -> Option<u64> {
    let n = s.len();
    if n <= 9 {
        let p = s.as_ptr();
        let raw = unsafe { std::ptr::read_unaligned(p as *const u64) };
        let lm = if n >= 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 };
        let c9 = if n == 9 { unsafe { *p.add(8) } } else { 0 };
        return pack(raw & lm, c9, n);
    }
    None
}

#[inline(always)]
fn decode_short(k: u64, out: &mut [u8; 9]) -> &[u8] {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        let c0 = ((k >> 56) & 0x7F) as u8;
        if c0 == 0 {
            return &out[..0];
        }
        let rest = std::arch::x86_64::_pdep_u64(k, LOW7).swap_bytes();
        let z = rest.wrapping_sub(ONES) & !rest & HI8;
        let len = 1 + (z.trailing_zeros() as usize >> 3).min(8);
        std::ptr::write_unaligned(out.as_mut_ptr() as *mut u64, (c0 as u64) | (rest << 8));
        out[8] = (rest >> 56) as u8;
        return &out[..len];
    }
    #[allow(unreachable_code)]
    {
        let mut n = 0;
        for i in 0..9 {
            let c = ((k >> (7 * (8 - i))) & 0x7F) as u8;
            if c == 0 {
                break;
            }
            out[n] = c;
            n += 1;
        }
        &out[..n]
    }
}

#[inline(always)]
fn hash_bytes(s: &[u8]) -> u64 {
    let mut h = (s.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F;
    let n = s.len();
    let mut i = 0;
    while i + 8 <= n {
        let w = unsafe { std::ptr::read_unaligned(s.as_ptr().add(i) as *const u64) };
        h = (h ^ w).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        i += 8;
    }
    if i < n {
        let mut tail = 0u64;
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr().add(i), &mut tail as *mut u64 as *mut u8, n - i);
        }
        h = (h ^ tail).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
    mix(h)
}

#[inline(always)]
fn arena_push(arena: &mut Vec<u8>, s: &[u8]) -> u32 {
    let off = arena.len() as u32;
    let n = s.len();
    if n < 255 {
        arena.push(n as u8);
    } else {
        arena.push(255);
        arena.extend_from_slice(&(n as u32).to_le_bytes());
    }
    arena.extend_from_slice(s);
    off
}

#[inline(always)]
fn arena_get(arena: &[u8], off: u32) -> &[u8] {
    let i = off as usize;
    let b0 = unsafe { *arena.get_unchecked(i) };
    if b0 < 255 {
        let len = b0 as usize;
        unsafe { arena.get_unchecked(i + 1..i + 1 + len) }
    } else {
        let len = u32::from_le_bytes(arena[i + 1..i + 5].try_into().unwrap()) as usize;
        &arena[i + 5..i + 5 + len]
    }
}

struct LocalIntern {
    arena: Vec<u8>,
    offs: Vec<u32>,
    hashes: Vec<u64>,
    tab: Vec<(u64, u32)>, // (hash, local_id + 1); 0 = empty
    mask: usize,
}
impl LocalIntern {
    fn new() -> Self {
        LocalIntern { arena: vec![], offs: vec![], hashes: vec![], tab: vec![], mask: 0 }
    }
    fn grow(&mut self) {
        let ncap = (self.tab.len() * 2).max(1024);
        let mut nt = vec![(0u64, 0u32); ncap];
        let mask = ncap - 1;
        for (id, &h) in self.hashes.iter().enumerate() {
            let mut idx = (h as usize) & mask;
            while nt[idx].1 != 0 {
                idx = (idx + 1) & mask;
            }
            nt[idx] = (h, id as u32 + 1);
        }
        self.tab = nt;
        self.mask = mask;
    }
    #[cold]
    #[inline(never)]
    fn intern(&mut self, s: &[u8]) -> u32 {
        if self.offs.len() * 2 >= self.tab.len() {
            self.grow();
        }
        let h = hash_bytes(s);
        let mask = self.mask;
        let mut idx = (h as usize) & mask;
        loop {
            let slot = self.tab[idx];
            if slot.1 == 0 {
                let id = self.offs.len() as u32;
                let off = arena_push(&mut self.arena, s);
                self.offs.push(off);
                self.hashes.push(h);
                self.tab[idx] = (h, id + 1);
                return id;
            }
            if slot.0 == h && arena_get(&self.arena, self.offs[(slot.1 - 1) as usize]) == s {
                return slot.1 - 1;
            }
            idx = (idx + 1) & mask;
        }
    }
    #[inline(always)]
    fn get(&self, id: u32) -> &[u8] {
        arena_get(&self.arena, self.offs[id as usize])
    }
}

#[derive(Default)]
struct GlobalIntern {
    arena: Vec<u8>,
    offs: Vec<u32>,
    hashes: Vec<u64>,
}
impl GlobalIntern {
    #[inline(always)]
    fn get(&self, id: u32) -> &[u8] {
        arena_get(&self.arena, self.offs[id as usize])
    }
}

fn merge_links_intern(lparts: &mut [(Vec<u8>, Vec<(u64, u64)>, LocalIntern)]) -> GlobalIntern {
    let total_local: usize = lparts.iter().map(|s| s.2.offs.len()).sum();
    let mut gi = GlobalIntern::default();
    if total_local == 0 {
        return gi;
    }
    let tcap = (total_local * 2).next_power_of_two().max(64);
    let mask = tcap - 1;
    let mut tab = vec![(0u64, 0u32); tcap];
    let mut remap: Vec<Vec<u64>> = Vec::with_capacity(lparts.len());
    for lp in lparts.iter() {
        let mut r = Vec::with_capacity(lp.2.offs.len());
        for lid in 0..lp.2.offs.len() {
            let h = lp.2.hashes[lid];
            let s = lp.2.get(lid as u32);
            let mut idx = (h as usize) & mask;
            let gid = loop {
                let slot = tab[idx];
                if slot.1 == 0 {
                    let g = gi.offs.len() as u32;
                    let off = arena_push(&mut gi.arena, s);
                    gi.offs.push(off);
                    gi.hashes.push(h);
                    tab[idx] = (h, g + 1);
                    break g;
                }
                if slot.0 == h && gi.get(slot.1 - 1) == s {
                    break slot.1 - 1;
                }
                idx = (idx + 1) & mask;
            };
            r.push(FALLBACK | gid as u64);
        }
        remap.push(r);
    }
    for lp in lparts.iter_mut() {
        for pair in lp.1.iter_mut() {
            if pair.0 & FALLBACK != 0 {
                let tid = ((pair.0 & !FALLBACK) >> 32) as usize;
                let lid = (pair.0 & 0xFFFF_FFFF) as usize;
                pair.0 = remap[tid][lid];
            }
            if pair.1 & FALLBACK != 0 {
                let tid = ((pair.1 & !FALLBACK) >> 32) as usize;
                let lid = (pair.1 & 0xFFFF_FFFF) as usize;
                pair.1 = remap[tid][lid];
            }
        }
        lp.2 = LocalIntern::new();
    }
    gi
}

fn merge_phase_a_intern(mut gi: GlobalIntern, sa_interns: &[LocalIntern], nt: usize) -> (GlobalIntern, Vec<Vec<u64>>) {
    let mut uremap: Vec<Vec<u64>> = sa_interns.iter().map(|li| vec![0u64; li.offs.len()]).collect();
    if sa_interns.iter().all(|li| li.offs.is_empty()) {
        return (gi, uremap);
    }
    const SHARDS: usize = 64;
    let mut lb: Vec<Vec<u32>> = vec![vec![]; SHARDS];
    for (id, &h) in gi.hashes.iter().enumerate() {
        lb[(h >> 58) as usize].push(id as u32);
    }
    let mut sb: Vec<Vec<(u16, u32)>> = vec![vec![]; SHARDS];
    for (tid, li) in sa_interns.iter().enumerate() {
        for (lid, &h) in li.hashes.iter().enumerate() {
            sb[(h >> 58) as usize].push((tid as u16, lid as u32));
        }
    }
    #[derive(Default)]
    struct ShardOut {
        new_items: Vec<(u64, u16, u32)>,
        hits: Vec<(u16, u32, u32)>, // bit 31 set => index into new_items; clear => existing global id
    }
    let outs: Vec<Mutex<ShardOut>> = (0..SHARDS).map(|_| Mutex::new(ShardOut::default())).collect();
    let gi_ref = &gi;
    run(
        nt,
        SHARDS,
        |_| Vec::<(u64, u32)>::new(),
        |tab, sh| {
            let llist = &lb[sh];
            let slist = &sb[sh];
            if slist.is_empty() {
                return;
            }
            let need = (llist.len() + slist.len()) * 2;
            let cap = need.next_power_of_two().max(16);
            tab.clear();
            tab.resize(cap, (0, 0));
            let mask = cap - 1;
            for &gid in llist {
                let h = gi_ref.hashes[gid as usize];
                let mut idx = (h as usize) & mask;
                while tab[idx].1 != 0 {
                    idx = (idx + 1) & mask;
                }
                tab[idx] = (h, gid + 1);
            }
            let mut so = ShardOut { new_items: Vec::new(), hits: Vec::with_capacity(slist.len()) };
            for &(tid, lid) in slist {
                let h = sa_interns[tid as usize].hashes[lid as usize];
                let s = sa_interns[tid as usize].get(lid);
                let mut idx = (h as usize) & mask;
                let tag = loop {
                    let slot = tab[idx];
                    if slot.1 == 0 {
                        let ni = so.new_items.len() as u32;
                        so.new_items.push((h, tid, lid));
                        let t = 0x8000_0000u32 | ni;
                        tab[idx] = (h, t + 1);
                        break t;
                    }
                    let id = slot.1 - 1;
                    let existing = if id & 0x8000_0000 != 0 {
                        let (_, t2, l2) = so.new_items[(id & 0x7FFF_FFFF) as usize];
                        sa_interns[t2 as usize].get(l2)
                    } else {
                        gi_ref.get(id)
                    };
                    if slot.0 == h && existing == s {
                        break id;
                    }
                    idx = (idx + 1) & mask;
                };
                so.hits.push((tid, lid, tag));
            }
            *outs[sh].lock().unwrap() = so;
        },
    );
    let outs: Vec<ShardOut> = outs.into_iter().map(|m| m.into_inner().unwrap()).collect();
    let mut bases = [0u32; SHARDS];
    let mut cur = gi.offs.len() as u32;
    let mut extra_bytes = 0usize;
    let mut extra_items = 0usize;
    for sh in 0..SHARDS {
        bases[sh] = cur;
        cur += outs[sh].new_items.len() as u32;
        extra_items += outs[sh].new_items.len();
        for &(_, tid, lid) in &outs[sh].new_items {
            let n = sa_interns[tid as usize].get(lid).len();
            extra_bytes += n + if n < 255 { 1 } else { 5 };
        }
    }
    gi.arena.reserve(extra_bytes);
    gi.offs.reserve(extra_items);
    gi.hashes.reserve(extra_items);
    for sh in 0..SHARDS {
        let base = bases[sh];
        for &(h, tid, lid) in &outs[sh].new_items {
            let s = sa_interns[tid as usize].get(lid);
            let off = arena_push(&mut gi.arena, s);
            gi.offs.push(off);
            gi.hashes.push(h);
        }
        for &(tid, lid, tag) in &outs[sh].hits {
            let gid = if tag & 0x8000_0000 != 0 { base + (tag & 0x7FFF_FFFF) } else { tag };
            uremap[tid as usize][lid as usize] = FALLBACK | gid as u64;
        }
    }
    (gi, uremap)
}

#[inline(always)]
fn resolve_uid(u: u64, uremap: &[Vec<u64>]) -> u64 {
    if u & FALLBACK == 0 {
        u
    } else {
        let tid = ((u & !FALLBACK) >> 32) as usize;
        let lid = (u & 0xFFFF_FFFF) as usize;
        unsafe { *uremap.get_unchecked(tid).get_unchecked(lid) }
    }
}

#[inline]
fn cmp_key_gi(a: u64, b: u64, gi: &GlobalIntern) -> Ordering {
    if (a | b) & FALLBACK == 0 {
        a.cmp(&b)
    } else {
        let mut ba = [0u8; 9];
        let mut bb = [0u8; 9];
        let sa = if a & FALLBACK == 0 { decode_short(a, &mut ba) } else { gi.get((a & !FALLBACK) as u32) };
        let sb = if b & FALLBACK == 0 { decode_short(b, &mut bb) } else { gi.get((b & !FALLBACK) as u32) };
        sa.cmp(sb)
    }
}

#[inline]
fn cmp_cid(a: u64, b: u64, cid_arenas: &[Vec<u8>]) -> Ordering {
    if (a | b) & FALLBACK == 0 {
        a.cmp(&b)
    } else {
        let mut ba = [0u8; 9];
        let mut bb = [0u8; 9];
        let sa = if a & FALLBACK == 0 {
            decode_short(a, &mut ba)
        } else {
            let tid = ((a & !FALLBACK) >> 32) as usize;
            let off = (a & 0xFFFF_FFFF) as u32;
            arena_get(&cid_arenas[tid], off)
        };
        let sb = if b & FALLBACK == 0 {
            decode_short(b, &mut bb)
        } else {
            let tid = ((b & !FALLBACK) >> 32) as usize;
            let off = (b & 0xFFFF_FFFF) as u32;
            arena_get(&cid_arenas[tid], off)
        };
        sa.cmp(sb)
    }
}

#[inline(always)]
fn mix(k: u64) -> u64 {
    let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}
#[inline(always)]
fn bucket(k: u64) -> usize {
    (mix(k) >> 56) as usize
}
/// bucket among 2^(64 - sh) buckets
#[inline(always)]
fn bucket_sh(k: u64, sh: u32) -> usize {
    (mix(k) >> sh) as usize
}

// ---------- timestamps ----------
#[inline(always)]
fn d2(b: &[u8], i: usize) -> Option<i64> {
    let a = b[i].wrapping_sub(b'0');
    let c = b[i + 1].wrapping_sub(b'0');
    if a > 9 || c > 9 {
        None
    } else {
        Some((a * 10 + c) as i64)
    }
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// None = bad format. F3 values beyond i64 return Some(i64::MAX) (outside any window).
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    if (n != 20 && n != 25) || b[4] != b'-' {
        if !b.iter().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let mut i = 0;
        while i < n && b[i] == b'0' {
            i += 1;
        }
        let r = &b[i..];
        if r.len() > 19 {
            return Some(i64::MAX);
        }
        let mut v: u64 = 0;
        for &c in r {
            v = v * 10 + (c - b'0') as u64;
        }
        return Some(if v > i64::MAX as u64 { i64::MAX } else { v as i64 });
    }
    if b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let y = d2(b, 0)? * 100 + d2(b, 2)?;
    let mo = d2(b, 5)?;
    let d = d2(b, 8)?;
    let h = d2(b, 11)?;
    let mi = d2(b, 14)?;
    let s = d2(b, 17)?;
    if !(1..=12).contains(&mo) || h > 23 || mi > 59 || s > 59 || d < 1 {
        return None;
    }
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if leap(y) {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    let mut ms = days_from_civil(y, mo, d) * DAY + (h * 3600 + mi * 60 + s) * 1000;
    if n == 20 {
        if b[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        ms -= sign * (oh * 60 + om) * 60_000;
    }
    Some(ms)
}


/// Date cache: remembers the last validated YYYY-MM-DD prefix.
#[derive(Clone, Copy)]
struct DC {
    k: u64,
    k2: u16,
    days: i64,
}
impl DC {
    fn new() -> Self {
        DC { k: 0, k2: 0, days: 0 }
    }
}

#[inline(always)]
fn parse_ts_c(b: &[u8], dc: &mut DC) -> Option<i64> {
    let n = b.len();
    if (n == 20 || n == 25) && b[4] == b'-' {
        let k = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let k2 = u16::from_le_bytes(b[8..10].try_into().unwrap());
        let days = if k == dc.k && k2 == dc.k2 {
            dc.days
        } else {
            if b[7] != b'-' {
                return None;
            }
            let y = d2(b, 0)? * 100 + d2(b, 2)?;
            let mo = d2(b, 5)?;
            let d = d2(b, 8)?;
            if !(1..=12).contains(&mo) || d < 1 {
                return None;
            }
            let dim = match mo {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                _ => {
                    if leap(y) {
                        29
                    } else {
                        28
                    }
                }
            };
            if d > dim {
                return None;
            }
            let days = days_from_civil(y, mo, d);
            *dc = DC { k, k2, days };
            days
        };
        if b[10] != b'T' || b[13] != b':' || b[16] != b':' {
            return None;
        }
        let h = d2(b, 11)?;
        let mi = d2(b, 14)?;
        let s = d2(b, 17)?;
        if h > 23 || mi > 59 || s > 59 {
            return None;
        }
        let mut ms = days * DAY + (h * 3600 + mi * 60 + s) * 1000;
        if n == 20 {
            if b[19] != b'Z' {
                return None;
            }
        } else {
            let sign = match b[19] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            if b[22] != b':' {
                return None;
            }
            let oh = d2(b, 20)?;
            let om = d2(b, 23)?;
            if oh > 23 || om > 59 {
                return None;
            }
            ms -= sign * (oh * 60 + om) * 60_000;
        }
        return Some(ms);
    }
    // F3: digits only
    if n == 0 {
        return None;
    }
    let mut v: u64 = 0;
    let mut over = false;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        if v >= 100_000_000_000_000_000 {
            over = true;
        }
        v = v.wrapping_mul(10).wrapping_add(d as u64);
    }
    if over || v > i64::MAX as u64 {
        return Some(i64::MAX);
    }
    Some(v as i64)
}

// ---------- fast timestamps (SWAR) ----------
const DIGIT_LO: u64 = 0x3030_3030_3030_3030;

/// All 8 bytes of x (each already minus '0') are 0..=9.
#[inline(always)]
fn all_digits(d: u64) -> bool {
    // a byte b is 0..=9 iff b < 10, i.e. (b + 0x76) has high bit 0 and b has high bit 0
    ((d.wrapping_add(0x7676_7676_7676_7676) | d) & HI8) == 0
}

/// 8 digit values (LE, first char most significant) -> number.
#[inline(always)]
fn swar8(d: u64) -> u64 {
    let x = (d.wrapping_mul(10)).wrapping_add(d >> 8) & 0x00FF_00FF_00FF_00FF;
    let x = (x.wrapping_mul(100)).wrapping_add(x >> 16) & 0x0000_FFFF_0000_FFFF;
    (x.wrapping_mul(10000)).wrapping_add(x >> 32) & 0xFFFF_FFFF
}

struct YearTab {
    y0: i64,
    start: [i64; 8], // days_from_civil(y0 + i, 1, 1)
    ystart: [[i64; 16]; 8], // days of (y0+i, mo, 1), mo 1..=12
    dim: [[i64; 16]; 8],
}
static YT: std::sync::OnceLock<YearTab> = std::sync::OnceLock::new();
fn year_tab(y0: i64) -> YearTab {
    let mut t = YearTab { y0, start: [0; 8], ystart: [[0; 16]; 8], dim: [[0; 16]; 8] };
    for i in 0..8 {
        let y = y0 + i as i64;
        t.start[i] = days_from_civil(y, 1, 1);
        let lp = leap(y) as usize;
        for mo in 1..=12 {
            t.ystart[i][mo] = t.start[i] + MCUM[lp][mo];
            t.dim[i][mo] = MDIM[lp][mo];
        }
    }
    t
}
const MCUM: [[i64; 13]; 2] = [
    [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    [0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
];
const MDIM: [[i64; 13]; 2] = [[0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]];

#[inline(always)]
fn civil_days(y: i64, mo: i64, d: i64) -> Option<i64> {
    if !(1..=12).contains(&mo) || d < 1 {
        return None;
    }
    let lp = leap(y) as usize;
    if d > MDIM[lp][mo as usize] {
        return None;
    }
    if let Some(t) = YT.get() {
        let i = y - t.y0;
        if (0..8).contains(&i) {
            return Some(t.start[i as usize] + MCUM[lp][mo as usize] + d - 1);
        }
    }
    Some(days_from_civil(y, mo, d))
}

/// Timestamp parse; may read up to 32 bytes from b's start (caller guarantees padding).
#[inline(always)]
unsafe fn parse_ts_fast(b: &[u8]) -> Option<i64> {
    let n = b.len();
    let p = b.as_ptr();
    if (n == 20 || n == 25) && unsafe { *p.add(4) } == b'-' {
        let v0 = unsafe { std::ptr::read_unaligned(p as *const u64) }; // YYYY-MM-
        let v1 = unsafe { std::ptr::read_unaligned(p.add(8) as *const u64) }; // DDTHH:MM
        let v2 = unsafe { std::ptr::read_unaligned(p.add(16) as *const u32) } as u64; // :SS?
        const M0: u64 = 0xFF00_00FF_0000_0000;
        const P0: u64 = 0x2D00_002D_0000_0000;
        const M1: u64 = 0x0000_FF00_00FF_0000;
        const P1: u64 = 0x0000_3A00_0054_0000;
        const D0: u64 = 0x00FF_FF00_FFFF_FFFF;
        const D1: u64 = 0xFFFF_00FF_FF00_FFFF;
        let d0 = (v0 & D0).wrapping_sub(DIGIT_LO & D0);
        let d1 = (v1 & D1).wrapping_sub(DIGIT_LO & D1);
        let d2v = (v2 & 0x00FF_FF00).wrapping_sub(0x0030_3000);
        let bad_sep = ((v0 & M0) ^ P0) | ((v1 & M1) ^ P1) | ((v2 & 0xFF) ^ 0x3A);
        let bad_dig = ((d0.wrapping_add(0x7676_7676_7676_7676) | d0)
            | (d1.wrapping_add(0x7676_7676_7676_7676) | d1)
            | (d2v.wrapping_add(0x7676_7676_7676_7676) | d2v))
            & HI8;
        if (bad_sep | bad_dig) != 0 {
            return None;
        }
        // pair trick: byte k of p = 10 * digit_k + digit_{k+1}
        let p0 = d0.wrapping_mul(10).wrapping_add(d0 >> 8);
        let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
        let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
        let pb = |x: u64, i: u32| ((x >> (8 * i)) & 0xFF) as i64;
        let y = pb(p0, 0) * 100 + pb(p0, 2);
        let mo = pb(p0, 5);
        let d = pb(p1, 0);
        let h = pb(p1, 3);
        let mi = pb(p1, 6);
        let s = pb(p2, 1);
        let t = unsafe { YT.get().unwrap_unchecked() };
        let yi = (y - t.y0) as u64;
        let days;
        if yi < 8 && (1..=12).contains(&mo) {
            let yi = yi as usize;
            let mo = mo as usize;
            if (h > 23) | (mi > 59) | (s > 59) | (d < 1) | (d > t.dim[yi][mo]) {
                return None;
            }
            days = t.ystart[yi][mo] + d - 1;
        } else {
            if h > 23 || mi > 59 || s > 59 {
                return None;
            }
            days = civil_days(y, mo, d)?;
        }
        let mut ms = days * DAY + (h * 3600 + mi * 60 + s) * 1000;
        let c19 = (v2 >> 24) as u8;
        if n == 20 {
            if c19 != b'Z' {
                return None;
            }
        } else {
            let sign = match c19 {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            if unsafe { *p.add(22) } != b':' {
                return None;
            }
            let oh = d2(b, 20)?;
            let om = d2(b, 23)?;
            if oh > 23 || om > 59 {
                return None;
            }
            ms -= sign * (oh * 60 + om) * 60_000;
        }
        return Some(ms);
    }
    if n == 0 {
        return None;
    }
    if n <= 16 {
        // right-align the digits in 16 bytes padded with '0'
        let raw = unsafe { std::ptr::read_unaligned(p as *const u128) };
        let sh = 8 * (16 - n) as u32;
        let v = if sh == 0 { raw } else { (raw << sh) | (0x3030_3030_3030_3030_3030_3030_3030_3030u128 & ((1u128 << sh) - 1)) };
        let lo = (v as u64).wrapping_sub(DIGIT_LO);
        let hi = ((v >> 64) as u64).wrapping_sub(DIGIT_LO);
        // bytes below '0' wrap to >= 0xD0 and are caught by the high-bit test
        if ((v as u64) & HI8) != 0 || (((v >> 64) as u64) & HI8) != 0 || !all_digits(lo) || !all_digits(hi) {
            return None;
        }
        // reject bytes < '0' (subtraction borrowed): check each original byte >= '0'
        let ge = |x: u64| ((x | HI8).wrapping_sub(DIGIT_LO) & HI8) == HI8;
        if !ge(v as u64) || !ge((v >> 64) as u64) {
            return None;
        }
        return Some((swar8(lo) * 100_000_000 + swar8(hi)) as i64);
    }
    parse_ts(b)
}


/// ingest_ts is always valid F1 (rules.md), so it is decoded without validation.
/// Any other length falls back to the validating parser.
#[inline(always)]
unsafe fn parse_ingest(b: &[u8]) -> Option<i64> {
    if b.len() != 20 {
        return unsafe { parse_ts_fast(b) };
    }
    let p = b.as_ptr();
    let v0 = unsafe { std::ptr::read_unaligned(p as *const u64) }; // YYYY-MM-
    let v1 = unsafe { std::ptr::read_unaligned(p.add(8) as *const u64) }; // DDTHH:MM
    let v2 = unsafe { std::ptr::read_unaligned(p.add(16) as *const u32) } as u64; // :SSZ
    const D0: u64 = 0x00FF_FF00_FFFF_FFFF;
    const D1: u64 = 0xFFFF_00FF_FF00_FFFF;
    let d0 = (v0 & D0).wrapping_sub(DIGIT_LO & D0);
    let d1 = (v1 & D1).wrapping_sub(DIGIT_LO & D1);
    let d2v = (v2 & 0x00FF_FF00).wrapping_sub(0x0030_3000);
    let p0 = d0.wrapping_mul(10).wrapping_add(d0 >> 8);
    let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
    let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
    let pb = |x: u64, i: u32| ((x >> (8 * i)) & 0xFF) as i64;
    let y = pb(p0, 0) * 100 + pb(p0, 2);
    let mo = pb(p0, 5);
    let d = pb(p1, 0);
    let t = unsafe { YT.get().unwrap_unchecked() };
    let yi = (y - t.y0) as u64;
    let days = if yi < 8 && (1..=12).contains(&mo) { t.ystart[yi as usize][mo as usize] + d - 1 } else { days_from_civil(y, mo, d) };
    Some(days * DAY + (pb(p1, 3) * 3600 + pb(p1, 6) * 60 + pb(p2, 1)) * 1000)
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    let r = &b[i..];
    if r.len() > 8 {
        return None;
    }
    let mut v: u32 = 0;
    for &c in r {
        v = v * 10 + (c - b'0') as u32;
    }
    if (1..=10_000_000).contains(&v) {
        Some(v)
    } else {
        None
    }
}


/// Parses "<prefix><canonical decimal>" (no leading zero unless "0"), up to 7 digits.
#[inline(always)]
fn num_id(s: &[u8], prefix: &[u8]) -> Option<usize> {
    let pl = prefix.len();
    if s.len() <= pl || s.len() > pl + 7 || &s[..pl] != prefix {
        return None;
    }
    let d = &s[pl..];
    if d.len() > 1 && d[0] == b'0' {
        return None;
    }
    let mut v = 0usize;
    for &c in d {
        let x = c.wrapping_sub(b'0');
        if x > 9 {
            return None;
        }
        v = v * 10 + x as usize;
    }
    Some(v)
}

/// Lookup table for ids: canonical "<prefix><number>" ids in a direct array,
/// everything else in a hash map.
struct IdTable {
    prefix: &'static [u8],
    arr: Vec<u16>,
    map: FxHashMap<Vec<u8>, u16>,
}
const NONE16: u16 = u16::MAX;
impl IdTable {
    fn new(prefix: &'static [u8]) -> Self {
        IdTable { prefix, arr: vec![], map: FxHashMap::default() }
    }
    fn insert_if_absent(&mut self, s: &[u8], v: u16) -> bool {
        if let Some(n) = num_id(s, self.prefix) {
            if n >= self.arr.len() {
                self.arr.resize(n + 1, NONE16);
            }
            if self.arr[n] != NONE16 {
                return false;
            }
            self.arr[n] = v;
            true
        } else {
            if self.map.contains_key(s) {
                return false;
            }
            self.map.insert(s.to_vec(), v);
            true
        }
    }
    /// get() for fields inside a padded buffer (reads 8 bytes past the prefix)
    #[inline(always)]
    unsafe fn get_fast(&self, s: &[u8]) -> Option<u16> {
        let pl = self.prefix.len();
        let n = s.len();
        if n > pl && n <= pl + 7 && pl <= 2 && s[0] == self.prefix[0] && (pl == 1 || s[1] == self.prefix[1]) {
            let k = n - pl;
            let p = unsafe { s.as_ptr().add(pl) };
            let raw = unsafe { std::ptr::read_unaligned(p as *const u64) };
            let lm = (1u64 << (8 * k)) - 1;
            let x = (raw & lm) | (!lm & DIGIT_LO);
            let d = x.wrapping_sub(DIGIT_LO);
            if ((x & HI8) | ((d.wrapping_add(0x7676_7676_7676_7676) | d) & HI8)) == 0 && !(k > 1 && (raw & 0xFF) == 0x30) {
                let v = swar8(d << (8 * (8 - k))) as usize;
                return match self.arr.get(v) {
                    Some(&r) if r != NONE16 => Some(r),
                    _ => None,
                };
            }
        }
        self.get(s)
    }
    #[inline(always)]
    fn get(&self, s: &[u8]) -> Option<u16> {
        if let Some(n) = num_id(s, self.prefix) {
            match self.arr.get(n) {
                Some(&v) if v != NONE16 => Some(v),
                _ => None,
            }
        } else if self.map.is_empty() {
            None
        } else {
            self.map.get(s).copied()
        }
    }
}

// ---------- CSV ----------
#[inline]
fn skip_line(b: &[u8], i: usize) -> usize {
    let i = i.min(b.len());
    match memchr(b'\n', &b[i..]) {
        Some(p) => i + p + 1,
        None => b.len(),
    }
}

/// Parses the line at b[i] into exactly N fields (quotes removed).
/// Returns (next line index, well-formed).
#[inline(always)]
fn next_line<const N: usize>(b: &[u8], mut i: usize, f: &mut [(usize, usize); N]) -> (usize, bool) {
    let n = b.len();
    let mut k = 0;
    loop {
        let (s, e);
        if i < n && b[i] == b'"' {
            let mut j = i + 1;
            while j < n && b[j] != b'"' && b[j] != b'\n' && b[j] != b',' {
                j += 1;
            }
            if j >= n || b[j] != b'"' {
                return (skip_line(b, j), false);
            }
            s = i + 1;
            e = j;
            i = j + 1;
            if i < n && b[i] != b',' && b[i] != b'\n' {
                return (skip_line(b, i), false);
            }
        } else {
            let mut j = i;
            while j < n {
                let c = b[j];
                if c == b',' || c == b'\n' || c == b'"' {
                    break;
                }
                j += 1;
            }
            if j < n && b[j] == b'"' {
                return (skip_line(b, j), false);
            }
            s = i;
            e = j;
            i = j;
        }
        if k == N {
            return (skip_line(b, i), false);
        }
        f[k] = (s, e);
        k += 1;
        if i >= n {
            return (n, k == N);
        }
        if b[i] == b'\n' {
            return (i + 1, k == N);
        }
        i += 1;
    }
}

#[inline(always)]
fn for_rows<const N: usize>(c: &[u8], mut g: impl FnMut([&[u8]; N])) {
    let mut f = [(0usize, 0usize); N];
    let mut i = 0;
    while i < c.len() {
        let (ni, ok) = next_line::<N>(c, i, &mut f);
        i = ni;
        if ok {
            let mut a: [&[u8]; N] = [&[]; N];
            for k in 0..N {
                a[k] = &c[f[k].0..f[k].1];
            }
            g(a);
        }
    }
}

// ---------- SIMD structural scan ----------
/// Collects positions of ',' and '\n' (seps) and '"' (quotes) in b.
fn scan(b: &[u8], seps: &mut Vec<u32>, quotes: &mut Vec<u32>) {
    seps.clear();
    quotes.clear();
    let n = b.len();
    seps.reserve(n / 6 + 64);
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let vc = _mm256_set1_epi8(b',' as i8);
        let vn = _mm256_set1_epi8(b'\n' as i8);
        let vq = _mm256_set1_epi8(b'"' as i8);
        let p = b.as_ptr();
        while i + 64 <= n {
            let a = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let c = _mm256_loadu_si256(p.add(i + 32) as *const __m256i);
            let sa = _mm256_or_si256(_mm256_cmpeq_epi8(a, vc), _mm256_cmpeq_epi8(a, vn));
            let sc = _mm256_or_si256(_mm256_cmpeq_epi8(c, vc), _mm256_cmpeq_epi8(c, vn));
            let mut ms = (_mm256_movemask_epi8(sa) as u32 as u64) | ((_mm256_movemask_epi8(sc) as u32 as u64) << 32);
            let mut mq = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, vq)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, vq)) as u32 as u64) << 32);
            if seps.capacity() - seps.len() < 64 {
                seps.reserve(4096);
            }
            let mut len = seps.len();
            let sp = seps.as_mut_ptr();
            while ms != 0 {
                *sp.add(len) = (i as u32) + ms.trailing_zeros();
                len += 1;
                ms &= ms - 1;
            }
            seps.set_len(len);
            while mq != 0 {
                quotes.push((i as u32) + mq.trailing_zeros());
                mq &= mq - 1;
            }
            i += 64;
        }
    }
    while i < n {
        let c = b[i];
        if c == b',' || c == b'\n' {
            seps.push(i as u32);
        } else if c == b'"' {
            quotes.push(i as u32);
        }
        i += 1;
    }
}

// ---------- per-line SIMD row parser ----------
#[inline(always)]
fn masks128(p: &[u8]) -> ([u64; 2], [u64; 2], [u64; 2]) {
    debug_assert!(p.len() >= 128);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let vc = _mm256_set1_epi8(b',' as i8);
        let vn = _mm256_set1_epi8(b'\n' as i8);
        let vq = _mm256_set1_epi8(b'"' as i8);
        let ptr = p.as_ptr();
        let a0 = _mm256_loadu_si256(ptr as *const __m256i);
        let a1 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
        let a2 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);
        let a3 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);
        let m = |v: __m256i, t: __m256i| _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, t)) as u32 as u64;
        let c = [m(a0, vc) | (m(a1, vc) << 32), m(a2, vc) | (m(a3, vc) << 32)];
        let n = [m(a0, vn) | (m(a1, vn) << 32), m(a2, vn) | (m(a3, vn) << 32)];
        let q = [m(a0, vq) | (m(a1, vq) << 32), m(a2, vq) | (m(a3, vq) << 32)];
        return (c, n, q);
    }
    #[allow(unreachable_code)]
    {
        let (mut c, mut n, mut q) = ([0u64; 2], [0u64; 2], [0u64; 2]);
        for i in 0..128 {
            match p[i] {
                b',' => c[i >> 6] |= 1 << (i & 63),
                b'\n' => n[i >> 6] |= 1 << (i & 63),
                b'"' => q[i >> 6] |= 1 << (i & 63),
                _ => {}
            }
        }
        (c, n, q)
    }
}


#[inline(always)]
fn pext64(x: u64, m: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        return std::arch::x86_64::_pext_u64(x, m);
    }
    #[allow(unreachable_code)]
    {
        let (mut r, mut bit, mut mm) = (0u64, 0u32, m);
        while mm != 0 {
            let t = mm.trailing_zeros();
            r |= ((x >> t) & 1) << bit;
            bit += 1;
            mm &= mm - 1;
        }
        r
    }
}

#[inline(always)]
fn low_mask(k: usize) -> u64 {
    if k >= 64 { u64::MAX } else { (1u64 << k) - 1 }
}

/// Iterates rows of buf[s..e]. buf must have at least 128 readable bytes past e.
#[inline(always)]
fn for_rows_fast<const N: usize>(buf: &[u8], s: usize, e: usize, g: impl FnMut([&[u8]; N])) {
    for_rows_fast_v::<N, 0>(buf, s, e, g)
}

/// V bit 1: commas extracted without data-dependent loop exits; V bit 2: quote validation also
/// for lines without quotes (no branch on the quote count)
#[inline(always)]
fn for_rows_fast_v<const N: usize, const V: u32>(buf: &[u8], s: usize, e: usize, mut g: impl FnMut([&[u8]; N])) {
    assert!(buf.len() >= e + 128);
    let mut ls = s;
    let mut f = [(0usize, 0usize); N];
    let base = buf.as_ptr();
    while ls < e {
        let lim = e - ls;
        let (mut mc, mut mn, mut mq) = masks128(unsafe { buf.get_unchecked(ls..ls + 128) });
        if lim < 128 {
            let v0 = low_mask(lim);
            let v1 = if lim > 64 { low_mask(lim - 64) } else { 0 };
            mn[0] &= v0;
            mn[1] &= v1;
        }
        let le_off = if mn[0] != 0 {
            mn[0].trailing_zeros() as usize
        } else if mn[1] != 0 {
            64 + mn[1].trailing_zeros() as usize
        } else if lim <= 128 {
            lim
        } else {
            // long line: slow path
            let (ni, ok) = next_line::<N>(&buf[..e], ls, &mut f);
            if ok {
                let mut a: [&[u8]; N] = [&[]; N];
                for k in 0..N {
                    a[k] = &buf[f[k].0..f[k].1];
                }
                if V & 128 != 0 {
                    let gd: &mut dyn FnMut([&[u8]; N]) = &mut g;
                    std::hint::black_box(&gd);
                    gd(a);
                } else {
                    g(a);
                }
            }
            ls = ni;
            continue;
        };
        if V & 64 != 0 {
            // framing only (measurement)
            std::hint::black_box(mc[0] ^ mc[1] ^ mq[0] ^ mq[1]);
            ls += le_off + 1;
            continue;
        }
        // restrict masks to the line
        let m0 = low_mask(le_off);
        let m1 = if le_off > 64 { low_mask(le_off - 64) } else { 0 };
        mc[0] &= m0;
        mc[1] &= m1;
        mq[0] &= m0;
        mq[1] &= m1;
        if (mc[0].count_ones() + mc[1].count_ones()) as usize == N - 1 {
            // field end offsets
            let mut ends = [0usize; N];
            if V & 1 != 0 {
                // exactly N-1 commas: extract them without data-dependent loop exits
                let (mut w0, mut w1) = (mc[0], mc[1]);
                for k in 0..N - 1 {
                    let p0 = w0.trailing_zeros() as usize;
                    let p1 = 64 + w1.trailing_zeros() as usize;
                    let use0 = w0 != 0;
                    ends[k] = if use0 { p0 } else { p1 };
                    let b0 = w0 & w0.wrapping_sub(1);
                    let b1 = w1 & w1.wrapping_sub(1);
                    w0 = if use0 { b0 } else { w0 };
                    w1 = if use0 { w1 } else { b1 };
                }
            } else {
                let mut k = 0;
                let mut c0 = mc[0];
                while c0 != 0 {
                    ends[k] = c0.trailing_zeros() as usize;
                    c0 &= c0 - 1;
                    k += 1;
                }
                let mut c1 = mc[1];
                while c1 != 0 {
                    ends[k] = 64 + c1.trailing_zeros() as usize;
                    c1 &= c1 - 1;
                    k += 1;
                }
            }
            ends[N - 1] = le_off;
            let lp = unsafe { base.add(ls) };
            let mut a: [&[u8]; N] = [&[]; N];
            let nq = mq[0].count_ones() + mq[1].count_ones();
            if V & 128 != 0 {
                // one call site for the row callback: compute the quote flags, then the slices
                let (fsb, ok): (u64, bool) = if nq == 0 {
                    (0, true)
                } else if le_off < 127 {
                    let sm = [(mc[0] << 1) | 1, (mc[1] << 1) | (mc[0] >> 63)];
                    let em = {
                        let mut x = [mc[0], mc[1]];
                        x[le_off >> 6] |= 1u64 << (le_off & 63);
                        x
                    };
                    let last = [(em[0] >> 1) | (em[1] << 63), em[1] >> 1];
                    let qs = [mq[0] & sm[0], mq[1] & sm[1]];
                    let ql = [mq[0] & last[0], mq[1] & last[1]];
                    let qe = [(mq[0] << 1) & em[0], ((mq[1] << 1) | (mq[0] >> 63)) & em[1]];
                    let fsb = pext64(qs[0], sm[0]) | (pext64(qs[1], sm[1]) << sm[0].count_ones());
                    let feb = pext64(qe[0], em[0]) | (pext64(qe[1], em[1]) << em[0].count_ones());
                    (fsb, (qs[0] | ql[0]) == mq[0] && (qs[1] | ql[1]) == mq[1] && fsb == feb && nq == 2 * fsb.count_ones())
                } else {
                    let mut fs = 0usize;
                    let mut bits = 0u64;
                    for k in 0..N {
                        let fe = ends[k];
                        let len = fe - fs;
                        let last = if len > 0 { fe - 1 } else { fs };
                        let qa = unsafe { *lp.add(fs) } == b'"';
                        let qb = unsafe { *lp.add(last) } == b'"';
                        bits |= ((qa & qb & (len >= 2)) as u64) << k;
                        fs = fe + 1;
                    }
                    (bits, nq == 2 * bits.count_ones())
                };
                if ok {
                    let mut fs = 0usize;
                    for k in 0..N {
                        let fe = ends[k];
                        let q = ((fsb >> k) & 1) as usize;
                        a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), fe - fs - 2 * q) };
                        fs = fe + 1;
                    }
                    g(a);
                }
            } else if V & 2 == 0 && nq == 0 {
                let mut fs = 0usize;
                for k in 0..N {
                    let fe = ends[k];
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs), fe - fs) };
                    fs = fe + 1;
                }
                g(a);
            } else if le_off < 127 {
                // Whole-line quote validation with bitmasks. A line is valid iff every quote is the
                // first or last char of its field, a field starts with a quote iff it ends with one,
                // and quoted fields have length >= 2 (so their two quotes are distinct).
                let sm = [(mc[0] << 1) | 1, (mc[1] << 1) | (mc[0] >> 63)]; // field starts
                let em = {
                    // separator after each field (commas and the line end)
                    let mut x = [mc[0], mc[1]];
                    x[le_off >> 6] |= 1u64 << (le_off & 63);
                    x
                };
                let last = [(em[0] >> 1) | (em[1] << 63), em[1] >> 1]; // last char of each field
                let qs = [mq[0] & sm[0], mq[1] & sm[1]];
                let ql = [mq[0] & last[0], mq[1] & last[1]];
                let qe = [(mq[0] << 1) & em[0], ((mq[1] << 1) | (mq[0] >> 63)) & em[1]];
                let fsb = pext64(qs[0], sm[0]) | (pext64(qs[1], sm[1]) << sm[0].count_ones());
                let feb = pext64(qe[0], em[0]) | (pext64(qe[1], em[1]) << em[0].count_ones());
                let ok = (qs[0] | ql[0]) == mq[0] && (qs[1] | ql[1]) == mq[1] && fsb == feb && nq == 2 * fsb.count_ones();
                if ok {
                    let mut fs = 0usize;
                    for k in 0..N {
                        let fe = ends[k];
                        let q = ((fsb >> k) & 1) as usize;
                        a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), fe - fs - 2 * q) };
                        fs = fe + 1;
                    }
                    g(a);
                }
            } else {
                let mut fs = 0usize;
                let mut cand = 0u32;
                for k in 0..N {
                    let fe = ends[k];
                    let len = fe - fs;
                    let last = if len > 0 { fe - 1 } else { fs };
                    let qa = unsafe { *lp.add(fs) } == b'"';
                    let qb = unsafe { *lp.add(last) } == b'"';
                    let q = (qa & qb & (len >= 2)) as usize;
                    cand += q as u32;
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), len - 2 * q) };
                    fs = fe + 1;
                }
                if nq == 2 * cand {
                    g(a);
                }
            }
        }
        ls += le_off + 1;
    }
}

/// masks over 128 bytes: separators (',' or '\n'), newlines, quotes
#[inline(always)]
fn masks128s(p: *const u8) -> ([u64; 2], [u64; 2], [u64; 2]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let vc = _mm256_set1_epi8(b',' as i8);
        let vn = _mm256_set1_epi8(b'\n' as i8);
        let vq = _mm256_set1_epi8(b'"' as i8);
        let a0 = _mm256_loadu_si256(p as *const __m256i);
        let a1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let a2 = _mm256_loadu_si256(p.add(64) as *const __m256i);
        let a3 = _mm256_loadu_si256(p.add(96) as *const __m256i);
        let n0 = _mm256_cmpeq_epi8(a0, vn);
        let n1 = _mm256_cmpeq_epi8(a1, vn);
        let n2 = _mm256_cmpeq_epi8(a2, vn);
        let n3 = _mm256_cmpeq_epi8(a3, vn);
        let s0 = _mm256_or_si256(_mm256_cmpeq_epi8(a0, vc), n0);
        let s1 = _mm256_or_si256(_mm256_cmpeq_epi8(a1, vc), n1);
        let s2 = _mm256_or_si256(_mm256_cmpeq_epi8(a2, vc), n2);
        let s3 = _mm256_or_si256(_mm256_cmpeq_epi8(a3, vc), n3);
        let mm = |v: __m256i| _mm256_movemask_epi8(v) as u32 as u64;
        let sep = [mm(s0) | (mm(s1) << 32), mm(s2) | (mm(s3) << 32)];
        let nl = [mm(n0) | (mm(n1) << 32), mm(n2) | (mm(n3) << 32)];
        let q = [mm(_mm256_cmpeq_epi8(a0, vq)) | (mm(_mm256_cmpeq_epi8(a1, vq)) << 32), mm(_mm256_cmpeq_epi8(a2, vq)) | (mm(_mm256_cmpeq_epi8(a3, vq)) << 32)];
        return (sep, nl, q);
    }
    #[allow(unreachable_code)]
    {
        let s = unsafe { std::slice::from_raw_parts(p, 128) };
        let (c, n, q) = masks128(s);
        ([c[0] | n[0], c[1] | n[1]], n, q)
    }
}

/// Same contract as for_rows_fast (identical results), leaner per-line work.
#[inline(always)]
fn for_rows_fast2<const N: usize>(buf: &[u8], s: usize, e: usize, mut g: impl FnMut([&[u8]; N])) {
    assert!(buf.len() >= e + 128);
    let base = buf.as_ptr();
    let mut ls = s;
    let mut f = [(0usize, 0usize); N];
    while ls < e {
        let lp = unsafe { base.add(ls) };
        let (mut sep, mut nl, mut q) = masks128s(lp);
        let lim = e - ls;
        if lim < 128 {
            let v0 = low_mask(lim);
            let v1 = if lim > 64 { low_mask(lim - 64) } else { 0 };
            sep[0] &= v0;
            sep[1] &= v1;
            nl[0] &= v0;
            nl[1] &= v1;
            q[0] &= v0;
            q[1] &= v1;
        }
        let le_off = if nl[0] != 0 {
            nl[0].trailing_zeros() as usize
        } else if nl[1] != 0 {
            64 + nl[1].trailing_zeros() as usize
        } else if lim <= 128 {
            lim
        } else {
            let (ni, ok) = next_line::<N>(&buf[..e], ls, &mut f);
            if ok {
                let mut a: [&[u8]; N] = [&[]; N];
                for k in 0..N {
                    a[k] = &buf[f[k].0..f[k].1];
                }
                g(a);
            }
            ls = ni;
            continue;
        };
        // separators strictly before the line end are commas
        let m0 = low_mask(le_off);
        let m1 = if le_off > 64 { low_mask(le_off - 64) } else { 0 };
        let c0 = sep[0] & m0;
        let c1 = sep[1] & m1;
        if (c0.count_ones() + c1.count_ones()) as usize == N - 1 {
            let mut ends = [0usize; N];
            let mut k = 0;
            let mut w = c0;
            while w != 0 {
                ends[k] = w.trailing_zeros() as usize;
                w &= w - 1;
                k += 1;
            }
            let mut w = c1;
            while w != 0 {
                ends[k] = 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                k += 1;
            }
            ends[N - 1] = le_off;
            let q0 = q[0] & m0;
            let q1 = q[1] & m1;
            let mut a: [&[u8]; N] = [&[]; N];
            if (q0 | q1) == 0 {
                let mut fs = 0usize;
                for k in 0..N {
                    let fe = ends[k];
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs), fe - fs) };
                    fs = fe + 1;
                }
                g(a);
            } else if le_off < 127 {
                let nq = q0.count_ones() + q1.count_ones();
                let sm = [(c0 << 1) | 1, (c1 << 1) | (c0 >> 63)]; // field starts
                let em = {
                    let mut x = [c0, c1];
                    x[le_off >> 6] |= 1u64 << (le_off & 63);
                    x
                };
                let last = [(em[0] >> 1) | (em[1] << 63), em[1] >> 1]; // last char of each field
                let qs = [q0 & sm[0], q1 & sm[1]];
                let ql = [q0 & last[0], q1 & last[1]];
                let qe = [(q0 << 1) & em[0], ((q1 << 1) | (q0 >> 63)) & em[1]];
                let fsb = pext64(qs[0], sm[0]) | (pext64(qs[1], sm[1]) << sm[0].count_ones());
                let feb = pext64(qe[0], em[0]) | (pext64(qe[1], em[1]) << em[0].count_ones());
                let ok = (qs[0] | ql[0]) == q0 && (qs[1] | ql[1]) == q1 && fsb == feb && nq == 2 * fsb.count_ones();
                if ok {
                    let mut fs = 0usize;
                    for k in 0..N {
                        let fe = ends[k];
                        let qf = ((fsb >> k) & 1) as usize;
                        a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + qf), fe - fs - 2 * qf) };
                        fs = fe + 1;
                    }
                    g(a);
                }
            } else {
                let nq = q0.count_ones() + q1.count_ones();
                let mut fs = 0usize;
                let mut cand = 0u32;
                for k in 0..N {
                    let fe = ends[k];
                    let len = fe - fs;
                    let qa = unsafe { *lp.add(fs) } == b'"';
                    let last = if fe > fs { fe - 1 } else { fs };
                    let qb = unsafe { *lp.add(last) } == b'"';
                    let qf = (qa & qb & (len >= 2)) as usize;
                    cand += qf as u32;
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + qf), len - 2 * qf) };
                    fs = fe + 1;
                }
                if nq == 2 * cand {
                    g(a);
                }
            }
        }
        ls += le_off + 1;
    }
}

#[inline(always)]
fn masks64(p: *const u8) -> (u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let vc = _mm256_set1_epi8(b',' as i8);
        let vn = _mm256_set1_epi8(b'\n' as i8);
        let vq = _mm256_set1_epi8(b'"' as i8);
        let a0 = _mm256_loadu_si256(p as *const __m256i);
        let a1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let mm = |v: __m256i, t: __m256i| _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, t)) as u32 as u64;
        return (
            mm(a0, vc) | (mm(a1, vc) << 32),
            mm(a0, vn) | (mm(a1, vn) << 32),
            mm(a0, vq) | (mm(a1, vq) << 32),
        );
    }
    #[allow(unreachable_code)]
    {
        let (mut c, mut n, mut q) = (0u64, 0u64, 0u64);
        for i in 0..64 {
            match unsafe { *p.add(i) } {
                b',' => c |= 1 << i,
                b'\n' => n |= 1 << i,
                b'"' => q |= 1 << i,
                _ => {}
            }
        }
        (c, n, q)
    }
}

/// Iterates rows of buf[s..e] using 128B+64B (192B) AVX2 framing. buf must have >= 256 readable bytes past e.
#[inline(always)]
fn for_rows_fast_192<const N: usize>(buf: &[u8], s: usize, e: usize, mut g: impl FnMut([&[u8]; N])) {
    assert!(buf.len() >= e + 256);
    let base = buf.as_ptr();
    let mut ls = s;
    let mut f = [(0usize, 0usize); N];
    while ls < e {
        let lp = unsafe { base.add(ls) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(lp.add(1024) as *const i8);
        }
        let lim = e - ls;
        let (mut mc, mut mn, mut mq) = masks128(unsafe { buf.get_unchecked(ls..ls + 128) });
        if lim < 128 {
            let v0 = low_mask(lim);
            let v1 = if lim > 64 { low_mask(lim - 64) } else { 0 };
            mn[0] &= v0;
            mn[1] &= v1;
        }
        if mn[0] != 0 || mn[1] != 0 || lim <= 128 {
            let le_off = if mn[0] != 0 {
                mn[0].trailing_zeros() as usize
            } else if mn[1] != 0 {
                64 + mn[1].trailing_zeros() as usize
            } else {
                lim
            };
            let m0 = low_mask(le_off);
            let m1 = if le_off > 64 { low_mask(le_off - 64) } else { 0 };
            mc[0] &= m0;
            mc[1] &= m1;
            mq[0] &= m0;
            mq[1] &= m1;
            if (mc[0].count_ones() + mc[1].count_ones()) as usize == N - 1 {
                let mut ends = [0usize; N];
                let mut k = 0;
                let mut c0 = mc[0];
                while c0 != 0 {
                    ends[k] = c0.trailing_zeros() as usize;
                    c0 &= c0 - 1;
                    k += 1;
                }
                let mut c1 = mc[1];
                while c1 != 0 {
                    ends[k] = 64 + c1.trailing_zeros() as usize;
                    c1 &= c1 - 1;
                    k += 1;
                }
                ends[N - 1] = le_off;
                let mut a: [&[u8]; N] = [&[]; N];
                let nq = mq[0].count_ones() + mq[1].count_ones();
                if nq == 0 {
                    let mut fs = 0usize;
                    for k in 0..N {
                        let fe = ends[k];
                        a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs), fe - fs) };
                        fs = fe + 1;
                    }
                    g(a);
                } else if le_off < 127 {
                    let sm = [(mc[0] << 1) | 1, (mc[1] << 1) | (mc[0] >> 63)];
                    let em = {
                        let mut x = [mc[0], mc[1]];
                        x[le_off >> 6] |= 1u64 << (le_off & 63);
                        x
                    };
                    let last = [(em[0] >> 1) | (em[1] << 63), em[1] >> 1];
                    let qs = [mq[0] & sm[0], mq[1] & sm[1]];
                    let ql = [mq[0] & last[0], mq[1] & last[1]];
                    let qe = [(mq[0] << 1) & em[0], ((mq[1] << 1) | (mq[0] >> 63)) & em[1]];
                    let fsb = pext64(qs[0], sm[0]) | (pext64(qs[1], sm[1]) << sm[0].count_ones());
                    let feb = pext64(qe[0], em[0]) | (pext64(qe[1], em[1]) << em[0].count_ones());
                    if (qs[0] | ql[0]) == mq[0] && (qs[1] | ql[1]) == mq[1] && fsb == feb && nq == 2 * fsb.count_ones() {
                        let mut fs = 0usize;
                        for k in 0..N {
                            let fe = ends[k];
                            let q = ((fsb >> k) & 1) as usize;
                            a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), fe - fs - 2 * q) };
                            fs = fe + 1;
                        }
                        g(a);
                    }
                } else {
                    let mut fs = 0usize;
                    let mut cand = 0u32;
                    for k in 0..N {
                        let fe = ends[k];
                        let len = fe - fs;
                        let q = (len >= 2 && unsafe { *lp.add(fs) == b'"' && *lp.add(fe - 1) == b'"' }) as usize;
                        cand += q as u32;
                        a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), len - 2 * q) };
                        fs = fe + 1;
                    }
                    if nq == 2 * cand {
                        g(a);
                    }
                }
            }
            ls += le_off + 1;
            continue;
        }
        // Check 128..192 bytes for lines containing 33-byte hex IDs
        let (mut mc2, mut mn2, mut mq2) = masks64(unsafe { lp.add(128) });
        if lim < 192 {
            mn2 &= low_mask(lim - 128);
        }
        if mn2 == 0 && lim > 192 {
            let (ni, ok) = next_line::<N>(&buf[..e], ls, &mut f);
            if ok {
                let mut a: [&[u8]; N] = [&[]; N];
                for k in 0..N {
                    a[k] = &buf[f[k].0..f[k].1];
                }
                g(a);
            }
            ls = ni;
            continue;
        }
        let le_off = if mn2 != 0 { 128 + mn2.trailing_zeros() as usize } else { lim };
        let m2 = low_mask(le_off - 128);
        mc2 &= m2;
        mq2 &= m2;
        if (mc[0].count_ones() + mc[1].count_ones() + mc2.count_ones()) as usize == N - 1 {
            let mut ends = [0usize; N];
            let mut k = 0;
            let mut c0 = mc[0];
            while c0 != 0 {
                ends[k] = c0.trailing_zeros() as usize;
                c0 &= c0 - 1;
                k += 1;
            }
            let mut c1 = mc[1];
            while c1 != 0 {
                ends[k] = 64 + c1.trailing_zeros() as usize;
                c1 &= c1 - 1;
                k += 1;
            }
            let mut c2 = mc2;
            while c2 != 0 {
                ends[k] = 128 + c2.trailing_zeros() as usize;
                c2 &= c2 - 1;
                k += 1;
            }
            ends[N - 1] = le_off;
            let mut a: [&[u8]; N] = [&[]; N];
            let nq = mq[0].count_ones() + mq[1].count_ones() + mq2.count_ones();
            if nq == 0 {
                let mut fs = 0usize;
                for k in 0..N {
                    let fe = ends[k];
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs), fe - fs) };
                    fs = fe + 1;
                }
                g(a);
            } else {
                let mut fs = 0usize;
                let mut cand = 0u32;
                for k in 0..N {
                    let fe = ends[k];
                    let len = fe - fs;
                    let q = (len >= 2 && unsafe { *lp.add(fs) == b'"' && *lp.add(fe - 1) == b'"' }) as usize;
                    cand += q as u32;
                    a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), len - 2 * q) };
                    fs = fe + 1;
                }
                if nq == 2 * cand {
                    g(a);
                }
            }
        }
        ls += le_off + 1;
    }
}

// ---------- files ----------
fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "csv").unwrap_or(false) && p.is_file())
            .collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}

fn read_small(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_default()
}

fn manifest_time(text: &str, name: &str) -> i64 {
    let i = text.find(&format!("\"{name}\"")).expect("manifest key");
    let rest = &text[i + name.len() + 2..];
    let a = rest.find('"').unwrap();
    let rest = &rest[a + 1..];
    let b = rest.find('"').unwrap();
    parse_ts(rest[..b].as_bytes()).expect("manifest ts")
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ev,
    Ck,
    Ln,
}

struct Chunk {
    kind: Kind,
    file: usize,
    s: u64,
    e: u64,
}

/// Reads the lines that start in [s, e) (header excluded) into buf.
fn read_chunk(f: &File, len: u64, c: &Chunk, buf: &mut Vec<u8>) -> (usize, usize) {
    use std::os::unix::fs::FileExt;
    let a = c.s.saturating_sub(1);
    let mut want = (c.e + 1024).min(len);
    loop {
        let n = (want - a) as usize;
        if buf.len() < n + 256 {
            buf.resize(n + 256, 0);
        }
        f.read_exact_at(&mut buf[..n], a).unwrap();
        let Some(p) = memchr(b'\n', &buf[..n]) else {
            if want == len {
                return (0, 0);
            }
            want = (want + 65536).min(len);
            continue;
        };
        let start = p + 1;
        if start as u64 + a >= c.e {
            return (0, 0);
        }
        let from = (c.e - 1 - a) as usize;
        match memchr(b'\n', &buf[from..n]) {
            Some(q) => return (start, from + q + 1),
            None => {
                if want == len {
                    return (start, n);
                }
                want = (want + 65536).min(len);
            }
        }
    }
}


/// Line range [start, end) of the data lines that start in [c.s, c.e) of a mapped file.
fn chunk_lines(m: &[u8], c: &Chunk) -> (usize, usize) {
    let (s, e) = (c.s as usize, c.e as usize);
    let a = s.saturating_sub(1);
    let Some(p) = memchr(b'\n', &m[a..]) else { return (0, 0) };
    let start = a + p + 1;
    if start >= e {
        return (0, 0);
    }
    let from = e - 1;
    let end = match memchr(b'\n', &m[from..]) {
        Some(q) => from + q + 1,
        None => m.len(),
    };
    (start, end)
}

#[inline]
fn populate(m: &[u8], s: usize, e: usize) {
    if e <= s {
        return;
    }
    unsafe {
        let base = m.as_ptr() as usize + s;
        let al = base & !4095;
        // MADV_POPULATE_READ = 22 (Linux 5.14+); errors are ignored
        libc::madvise(al as *mut libc::c_void, e - s + (base - al), 22);
    }
}

/// Runs `ntasks` tasks over `nt` threads; each thread owns a state.
fn run<S: Send>(nt: usize, ntasks: usize, init: impl Fn(usize) -> S + Sync, f: impl Fn(&mut S, usize) + Sync) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|tid| {
                let init = &init;
                let f = &f;
                let next = &next;
                sc.spawn(move || {
                    let mut s = init(tid);
                    loop {
                        let t = next.fetch_add(1, AO::Relaxed);
                        if t >= ntasks {
                            break;
                        }
                        f(&mut s, t);
                        // order any non-temporal stores of this task before other threads read them
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_sfence();
                        }
                    }
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// `run`, and then every thread calls `fin(state, tid)` once it finds no task left (while other
/// threads may still be working on theirs)
fn run_fin<S: Send>(nt: usize, ntasks: usize, init: impl Fn(usize) -> S + Sync, f: impl Fn(&mut S, usize) + Sync, fin: impl Fn(&mut S, usize) + Sync) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|tid| {
                let init = &init;
                let f = &f;
                let fin = &fin;
                let next = &next;
                sc.spawn(move || {
                    let mut s = init(tid);
                    loop {
                        let t = next.fetch_add(1, AO::Relaxed);
                        if t >= ntasks {
                            break;
                        }
                        f(&mut s, t);
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_sfence();
                        }
                    }
                    fin(&mut s, tid);
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

// ---------- records ----------
const AUX_INVALID: u32 = u32::MAX;

#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct Ev {
    eid: u64,      // bit 63 == 0: packed short eid; bit 63 == 1: hash_bytes(eid) | FALLBACK
    uid: u64,      // short packed uid or FALLBACK | ((tid as u64) << 32) | lid
    ingest: i32,   // seconds relative to window start (clamped)
    ts: u32,       // ms since window start
    aux: u32,      // 0 = view/cart; 1..=u32::MAX-1 = purchase ref; u32::MAX = invalid
    long_off: u32, // offset in sa[tid].eid_arena when eid & FALLBACK != 0
}

#[derive(Clone, Copy, Default)]
#[repr(align(16))]
struct PEv {
    person: u64,
    ts: u32,
    aux: u32, // 0 = not a purchase, else 1 + (tid << 26 | idx) into the purchase side tables
}

#[derive(Clone, Copy, Default)]
#[repr(align(32))]
struct Ck {
    person: u64,
    cid: u64,
    ts: u32, // ms since click window start
    camp: u16,
}

#[derive(Clone)]
struct UserRow {
    person: u64,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
}



#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

/// Open-addressing map from linked ID to person name (only IDs whose person differs).
struct PTab {
    t: Vec<(u64, u64)>,
    mask: usize,
}
impl PTab {
    fn new(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(16);
        PTab { t: vec![(u64::MAX, 0); cap], mask: cap - 1 }
    }
    fn insert(&mut self, k: u64, v: u64) {
        let mut h = (mix(k) as usize) & self.mask;
        while self.t[h].0 != u64::MAX {
            h = (h + 1) & self.mask;
        }
        self.t[h] = (k, v);
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        let h = (mix(k) as usize) & self.mask;
        prefetch(unsafe { self.t.as_ptr().add(h) });
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        let mut h = (mix(k) as usize) & self.mask;
        loop {
            let s = self.t[h];
            if s.0 == k {
                return s.1;
            }
            if s.0 == u64::MAX {
                return k;
            }
            h = (h + 1) & self.mask;
        }
    }
}


/// Software write-combining scatter: records for each bucket are staged in a small
/// cache-resident buffer and written out in whole 64-byte lines with non-temporal
/// stores, avoiding a read-for-ownership of every destination cache line.
struct Wc<T: Copy, const K: usize> {
    out: Vec<Vec<T>>,
    st: Vec<[T; K]>,
    n: Vec<u8>,
    /// full vectors of buckets that outgrew their capacity (bucket, records), in fill order;
    /// a bucket continues in a fresh vector instead of being reallocated and copied
    ovf: Vec<(u32, Vec<T>)>,
}
impl<T: Copy + Default, const K: usize> Wc<T, K> {
    fn new(nb: usize, cap: usize) -> Self {
        assert!((std::mem::size_of::<T>() * K) % 64 == 0 && std::mem::align_of::<T>() >= 16);
        Wc { out: buckets_n(nb, cap), st: vec![[T::default(); K]; nb], n: vec![0; nb], ovf: vec![] }
    }
    /// only the first `used` buckets get the capacity up front
    fn new_n(nb: usize, cap: usize, used: usize) -> Self {
        assert!((std::mem::size_of::<T>() * K) % 64 == 0 && std::mem::align_of::<T>() >= 16);
        Wc { out: (0..nb).map(|i| Vec::with_capacity(if i < used { cap } else { 0 })).collect(), st: vec![[T::default(); K]; nb], n: vec![0; nb], ovf: vec![] }
    }
    #[inline(always)]
    fn push(&mut self, b: usize, x: T) {
        unsafe {
            let c = *self.n.get_unchecked(b) as usize;
            *self.st.get_unchecked_mut(b).get_unchecked_mut(c) = x;
            if c + 1 == K {
                self.flush(b);
            } else {
                *self.n.get_unchecked_mut(b) = (c + 1) as u8;
            }
        }
    }
    #[inline(always)]
    fn flush(&mut self, b: usize) {
        unsafe {
            let v = self.out.get_unchecked_mut(b);
            if v.capacity() - v.len() < K {
                // continue in a fresh vector (a quarter of the old capacity) instead of copying
                let ncap = (v.capacity() / 4).max(16 * K);
                let full = std::mem::replace(v, Vec::with_capacity(ncap));
                if !full.is_empty() {
                    self.ovf.push((b as u32, full));
                }
            }
            let v = self.out.get_unchecked_mut(b);
            let len = v.len();
            let dst = v.as_mut_ptr().add(len) as *mut u8;
            let src = self.st.get_unchecked(b).as_ptr() as *const u8;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                for i in 0..(K * std::mem::size_of::<T>() / 16) {
                    _mm_stream_si128(dst.add(16 * i) as *mut __m128i, _mm_load_si128(src.add(16 * i) as *const __m128i));
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            std::ptr::copy_nonoverlapping(src, dst, K * std::mem::size_of::<T>());
            v.set_len(len + K);
            *self.n.get_unchecked_mut(b) = 0;
        }
    }
    fn empty() -> Self {
        Wc { out: vec![], st: vec![], n: vec![], ovf: vec![] }
    }
    /// highest bucket holding a record (0 if none)
    fn top_nonempty(&self) -> usize {
        let mut b = self.n.len();
        while b > 0 {
            b -= 1;
            if self.n[b] != 0 || !self.out[b].is_empty() {
                return b;
            }
        }
        0
    }
    /// per-bucket vectors (the vectors that overflowed are returned separately, in fill order)
    fn finish_split(mut self) -> (Vec<Vec<T>>, Vec<(u32, Vec<T>)>) {
        for b in 0..self.n.len() {
            let c = self.n[b] as usize;
            let st = self.st[b];
            self.out[b].extend_from_slice(&st[..c]);
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_sfence();
        }
        (self.out, self.ovf)
    }
    /// per-bucket vectors with every record in push order
    fn finish(self) -> Vec<Vec<T>> {
        let (mut out, ovf) = self.finish_split();
        if !ovf.is_empty() {
            let mut pre: Vec<Vec<T>> = (0..out.len()).map(|_| vec![]).collect();
            for (b, v) in ovf {
                pre[b as usize].extend_from_slice(&v);
            }
            for (b, p) in pre.into_iter().enumerate() {
                if !p.is_empty() {
                    let mut p = p;
                    p.extend_from_slice(&out[b]);
                    out[b] = p;
                }
            }
        }
        out
    }
}

fn buckets<T>(cap: usize) -> Vec<Vec<T>> {
    buckets_n(NB, cap)
}
fn buckets_n<T>(n: usize, cap: usize) -> Vec<Vec<T>> {
    (0..n).map(|_| Vec::with_capacity(cap)).collect()
}


fn fuzz() {
    let _ = YT.set(year_tab(2025));
    let mut rng: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let seeds: Vec<&[u8]> = vec![
        b"2026-10-03T23:58:10Z", b"2026-10-03T19:58:10-04:00", b"1791071890000", b"0001791071890000",
        b"2028-02-29T00:00:00Z", b"2027-02-29T00:00:00Z", b"2026-12-31T23:59:59+23:59", b"2026-01-01T00:00:00-00:00",
        b"86400000", b"123456789012345678901", b"9999-12-31T23:59:59Z", b"0000-01-01T00:00:00Z", b"18446744073709551616",
    ];
    let alpha = b"0123456789-:TZtz+ .\r/\x00\xff";
    let mut buf = vec![0u8; 256];
    let mut bad = 0;
    for it in 0..20_000_000u64 {
        let base = seeds[(next() % seeds.len() as u64) as usize];
        let mut v = base.to_vec();
        let muts = next() % 4;
        for _ in 0..muts {
            let r = next();
            match r % 4 {
                0 if !v.is_empty() => { let i = (r >> 8) as usize % v.len(); v[i] = alpha[(r >> 20) as usize % alpha.len()]; }
                1 if !v.is_empty() => { let i = (r >> 8) as usize % v.len(); v.remove(i); }
                2 => { let i = (r >> 8) as usize % (v.len() + 1); v.insert(i, alpha[(r >> 20) as usize % alpha.len()]); }
                _ if !v.is_empty() => { let i = (r >> 8) as usize % v.len(); v[i] = (r >> 30) as u8; }
                _ => {}
            }
        }
        // random garbage after the field
        for x in buf.iter_mut() { *x = (next() & 0xFF) as u8; }
        buf[..v.len()].copy_from_slice(&v);
        let a = parse_ts(&v);
        let b = unsafe { parse_ts_fast(&buf[..v.len()]) };
        let norm = |x: Option<i64>| x.map(|t| if t >= 1_000_000_000_000_000_000 { i64::MAX } else { t });
        if norm(a) != norm(b) {
            bad += 1;
            if bad < 10 { eprintln!("MISMATCH it={it} {:?} slow={a:?} fast={b:?}", String::from_utf8_lossy(&v)); }
        }
        if v.len() == 20 && a.is_some() && v[19] == b'Z' {
            let c = unsafe { parse_ingest(&buf[..v.len()]) };
            if c != a {
                bad += 1;
                if bad < 10 { eprintln!("INGEST MISMATCH {:?} {a:?} {c:?}", String::from_utf8_lossy(&v)); }
            }
        }
        // keys
        let k1 = key_short(&v);
        let k2 = unsafe { key_short_fast(&buf[..v.len()]) };
        let mut kb = [0u8; 9];
        if k1 != k2 || k1.map(|k| decode_short(k, &mut kb).to_vec()) != k1.map(|_| v.clone()) {
            bad += 1;
            if bad < 10 { eprintln!("KEY MISMATCH {:?}", v); }
        }
    }
    eprintln!("fuzz done, mismatches: {bad}");
}


fn fuzz_rows() {
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let atoms: [&[u8]; 13] = [b"u1", b"", b"\"", b"\"\"", b"abc", b"\"x\"", b"x\"", b"\"x", b"a\"b", b"2026-10-03T23:58:10Z", b"1791071890000", b" q ", b"u74d1e000e9b9b6b41ebdd2fef9c34afa"];
    let mut bad = 0u64;
    let mut total = 0u64;
    for it in 0..3_000_000u64 {
        // build a chunk of 1..6 lines with 5..9 fields, random quoting, spanning <=128B, 129..192B, and >192B
        let mut chunk: Vec<u8> = vec![];
        let nl = 1 + next() % 6;
        for _ in 0..nl {
            let nf = if next() % 5 == 0 { 5 + next() % 5 } else { 7 };
            for f in 0..nf {
                if f > 0 { chunk.push(b','); }
                let a = atoms[(next() % atoms.len() as u64) as usize];
                if next() % 4 == 0 && !a.contains(&b'"') { chunk.push(b'"'); chunk.extend_from_slice(a); chunk.push(b'"'); } else { chunk.extend_from_slice(a); }
                if next() % 30 == 0 { for _ in 0..(next() % 50) { chunk.push(b'y'); } }
            }
            if next() % 30 != 0 { chunk.push(b'\n'); }
        }
        let e = chunk.len();
        let mut buf = chunk.clone();
        for _ in 0..256 { buf.push((next() & 0xff) as u8); }
        let mut fast: Vec<Vec<Vec<u8>>> = vec![];
        match it % 7 {
            0 => for_rows_fast_v::<7, 0>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            1 => for_rows_fast_v::<7, 1>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            2 => for_rows_fast_v::<7, 2>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            3 => for_rows_fast_v::<7, 3>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            4 if it % 2 == 0 => for_rows_fast_v::<7, 128>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            4 => for_rows_fast_v::<7, 129>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
            _ => for_rows_fast_192::<7>(&buf, 0, e, |a| fast.push(a.iter().map(|x| x.to_vec()).collect())),
        }
        let mut slow: Vec<Vec<Vec<u8>>> = vec![];
        let mut f = [(0usize, 0usize); 7];
        let mut i = 0;
        while i < e {
            let (ni, ok) = next_line::<7>(&chunk[..e], i, &mut f);
            if ok { slow.push(f.iter().map(|&(a, b)| chunk[a..b].to_vec()).collect()); }
            i = ni;
        }
        total += slow.len() as u64;
        if fast != slow {
            bad += 1;
            if bad < 5 { eprintln!("ROW MISMATCH it={it} chunk={:?}\n fast={:?}\n slow={:?}", String::from_utf8_lossy(&chunk), fast, slow); }
        }
    }
    eprintln!("fuzz_rows done: {total} valid rows compared, mismatching chunks: {bad}");
}

/// Worker threads when NCPU is not set: the CPUs this process may run on (the scorer pins 8
/// vCPUs), at most 8. A cgroup CPU quota is deliberately not consulted.
pub(crate) fn default_threads() -> usize {
    let mut n = 0usize;
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            n = libc::CPU_COUNT(&set) as usize;
        }
    }
    if n == 0 {
        n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    }
    n.clamp(1, 8)
}

#[inline(always)]
unsafe fn parse_ts_win(b: &[u8], win: &fast::MonTab, loc: &fast::MonTab) -> Option<i64> {
    let n = b.len();
    let p = b.as_ptr();
    if n == 20 && unsafe { *p.add(4) } == b'-' && win.complete {
        return unsafe { fast::f1_tab(p, win) };
    }
    if n == 13 {
        return unsafe { fast::digits13(p) };
    }
    if n == 25 && unsafe { *p.add(4) } == b'-' && loc.complete {
        return unsafe { fast::f2_tab(p, loc) };
    }
    unsafe { parse_ts_fast(b) }
}

#[inline(always)]
unsafe fn ingest_win(b: &[u8], tab: &fast::MonTab, ws: i64) -> i64 {
    if b.len() == 20 {
        if let Some(v) = unsafe { fast::ingest_tab(b.as_ptr(), tab, ws) } {
            return v;
        }
    }
    let ms = unsafe { parse_ingest(b) }.unwrap_or(i64::MAX);
    (ms.saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64)
}

fn mutate_hex_id(out: &mut Vec<u8>, val: &[u8]) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let h1 = hash_bytes(val);
    let h2 = mix(h1 ^ 0x517c_c1b7_2722_0a95);
    out.push(val[0]);
    for i in 0..16 {
        out.push(HEX[((h1 >> (4 * i)) & 15) as usize]);
    }
    for i in 0..16 {
        out.push(HEX[((h2 >> (4 * i)) & 15) as usize]);
    }
}

fn mutate_hex_file<const N: usize>(path: &Path) {
    let b = std::fs::read(path).unwrap();
    let Some(hdr_end) = memchr(b'\n', &b) else { return };
    let mut out = Vec::with_capacity(b.len() + b.len() / 3 + 64);
    out.extend_from_slice(&b[..hdr_end + 1]);
    let mut i = hdr_end + 1;
    let mut f = [(0usize, 0usize); N];
    while i < b.len() {
        let line_end = match memchr(b'\n', &b[i..]) {
            Some(p) => i + p + 1,
            None => b.len(),
        };
        let line = &b[i..line_end];
        let (ni, ok) = next_line::<N>(line, 0, &mut f);
        if !ok || ni != line.len() {
            out.extend_from_slice(line);
        } else {
            let mut pos = 0usize;
            for idx in 0..2 {
                let (s, e) = f[idx];
                let val = &line[s..e];
                out.extend_from_slice(&line[pos..s]);
                if !val.is_empty() && !val.contains(&b'\r') && (hash_bytes(val) % 6 == 0) {
                    mutate_hex_id(&mut out, val);
                } else {
                    out.extend_from_slice(val);
                }
                pos = e;
            }
            out.extend_from_slice(&line[pos..]);
        }
        i = line_end;
    }
    std::fs::write(path, &out).unwrap();
}

fn mutate_hex_dir(dir: &Path, nt: usize) {
    let mut jobs: Vec<(usize, PathBuf)> = vec![];
    for (ncols, sub) in [(7usize, "events"), (4usize, "clicks"), (2usize, "links")] {
        for p in list_csv(&dir.join(sub)) {
            jobs.push((ncols, p));
        }
    }
    run(
        nt,
        jobs.len(),
        |_| (),
        |_, t| {
            let (ncols, ref p) = jobs[t];
            match ncols {
                7 => mutate_hex_file::<7>(p),
                4 => mutate_hex_file::<4>(p),
                _ => mutate_hex_file::<2>(p),
            }
        },
    );
}

fn main() {
    if std::env::var("FUZZROWS").is_ok() { fuzz_rows(); return; }
    // Hand off to the AVX-512 build when the CPU supports it (the entry binary is the
    // AVX2 build, so a machine without AVX-512 never executes an AVX-512 instruction).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512bw")))]
    {
        let fast = Path::new("/app/sess_avx512");
        if std::env::var_os("SESS_NO_DISPATCH").is_none()
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("bmi2")
            && fast.exists()
        {
            use std::os::unix::process::CommandExt;
            let err = std::process::Command::new(fast).env("SESS_NO_DISPATCH", "1").exec();
            eprintln!("dispatch failed ({err}); continuing with AVX2 build");
        }
    }
    if std::env::var("FUZZ").is_ok() { fuzz(); return; }
    if let Ok(n) = std::env::var("FUZZEV") { fast::fuzz_ev(n.parse().unwrap_or(1000)); return; }
    if let Ok(n) = std::env::var("SESSFUZZ") { fast::fuzz_sess(n.parse().unwrap_or(1000)); return; }
    if let Ok(n) = std::env::var("FUZZCK") { fast::fuzz_ck(n.parse().unwrap_or(1000)); return; }
    if std::env::var("FAST_FUZZ").is_ok() { fast::fuzz_fast(); return; }
    if let Ok(d) = std::env::var("PQ_TEST") { fast::pq_empty_test(Path::new(&d)); return; }
    if let Ok(w) = std::env::var("FAST_BENCH") {
        fast::bench(w.parse().unwrap_or(0), &PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into())));
        return;
    }
    if let Ok(mdir) = std::env::var("MUTATE_HEX") {
        let nt = std::env::var("NCPU").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_threads).clamp(1, 63);
        mutate_hex_dir(Path::new(&mdir), nt);
        return;
    }
    let t0 = std::time::Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into()));
    let outdir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or("/output".into()));
    let nt: usize = std::env::var("NCPU").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_threads).clamp(1, 63);
    let verbose = std::env::var("VERBOSE").is_ok();
    if verbose {
        eprintln!("threads {nt} (default {})", default_threads());
    }
    let last = std::cell::Cell::new((0f64, 0f64, 0i64));
    let log = |m: &str| {
        if verbose {
            let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
            unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
            let u = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6;
            let sy = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 * 1e-6;
            let fl = ru.ru_minflt as i64;
            let (pu, ps, pf) = last.get();
            last.set((u, sy, fl));
            eprintln!("{m:12} wall {:6.3}  user +{:.2}  sys +{:.2}  flt +{}", t0.elapsed().as_secs_f64(), u - pu, sy - ps, fl - pf);
        }
    };

    if knob("SESS_FAST", 0) != 0 {
        match fast::fast_main(&data, &outdir, nt, &log) {
            Ok(()) => std::process::exit(0),
            Err(why) => {
                if verbose {
                    eprintln!("fast path declined: {why}");
                }
                let _ = std::fs::remove_dir_all(outdir.join("user_daily"));
                let _ = std::fs::remove_dir_all(outdir.join("campaign_daily"));
            }
        }
    }
    let manifest = String::from_utf8_lossy(&read_small(&data.join("manifest.json"))).to_string();
    let ws = manifest_time(&manifest, "window_start");
    let we = manifest_time(&manifest, "window_end");
    let cws = ws - WEEK;
    assert!(we - cws < u32::MAX as i64);
    {
        let dd = cws.div_euclid(DAY);
        let mut y = 1970i64;
        while days_from_civil(y + 1, 1, 1) <= dd {
            y += 1;
        }
        while days_from_civil(y, 1, 1) > dd {
            y -= 1;
        }
        let y0 = y - 1;
        let _ = YT.set(year_tab(y0));
    }
    let evwin = fast::MonTab::new(ws, we - 1);
    let ckwin = fast::MonTab::new(cws, we - 1);
    let evloc = fast::MonTab::new(ws - DAY, we + DAY);
    let ckloc = fast::MonTab::new(cws - DAY, we + DAY);
    let ingtab = fast::MonTab::new(ws - DAY, we + 6 * DAY);

    // products -> categories
    let mut cat_names: Vec<Vec<u8>> = vec![];
    let mut cat_idx: FxHashMap<Vec<u8>, u16> = FxHashMap::default();
    let mut products = IdTable::new(b"p");
    {
        let b = read_small(&data.join("products.csv"));
        if let Some(h) = memchr(b'\n', &b) {
            for_rows::<2>(&b[h + 1..], |[p, c]| {
                if p.is_empty() {
                    return;
                }
                let n = cat_names.len() as u16;
                let ci = *cat_idx.entry(c.to_vec()).or_insert_with(|| {
                    cat_names.push(c.to_vec());
                    n
                });
                products.insert_if_absent(p, ci);
            });
        }
    }
    let unknown_cat = cat_names.len() as u16;
    cat_names.push(b"unknown".to_vec());
    let ncat = cat_names.len();
    let mut order: Vec<usize> = (0..ncat).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(&cat_names[b]));
    let mut cat_rank = vec![0usize; ncat];
    for (r, &c) in order.iter().enumerate() {
        cat_rank[c] = r;
    }

    // campaigns
    let mut camp_ids: Vec<Vec<u8>> = vec![];
    let mut camp_ch: Vec<Vec<u8>> = vec![];
    let mut campaigns = IdTable::new(b"c_");
    {
        let b = read_small(&data.join("campaigns.csv"));
        if let Some(h) = memchr(b'\n', &b) {
            for_rows::<2>(&b[h + 1..], |[c, ch]| {
                if c.is_empty() {
                    return;
                }
                if !campaigns.insert_if_absent(c, camp_ids.len() as u16) {
                    return;
                }
                camp_ids.push(c.to_vec());
                camp_ch.push(ch.to_vec());
            });
        }
    }
    let organic = camp_ids.len() as u16;
    camp_ids.push(b"organic".to_vec());
    camp_ch.push(b"organic".to_vec());
    let ncamp = camp_ids.len();

    // ---- chunk lists ----
    let mut files: Vec<(File, u64)> = vec![];
    let mut maps: Vec<Option<memmap2::Mmap>> = vec![];
    let mut chunks: Vec<Chunk> = vec![];
    let mut lchunks: Vec<Chunk> = vec![];
    let mut total_ev: u64 = 0;
    let mut total_ck: u64 = 0;
    for (kind, dir) in [(Kind::Ev, "events"), (Kind::Ck, "clicks"), (Kind::Ln, "links")] {
        for p in list_csv(&data.join(dir)) {
            let Ok(f) = File::open(&p) else { continue };
            let len = f.metadata().map(|m| m.len()).unwrap_or(0);
            if len == 0 {
                continue;
            }
            if kind == Kind::Ev {
                total_ev += len;
            }
            if kind == Kind::Ck {
                total_ck += len;
            }
            let fi = files.len();
            let mm = if *K_MMAP && kind != Kind::Ln { unsafe { memmap2::Mmap::map(&f) }.ok() } else { None };
            files.push((f, len));
            maps.push(mm);
            let step = 1 << 20;
            let mut s = 0;
            while s < len {
                let e = (s + step).min(len);
                let c = Chunk { kind, file: fi, s, e };
                if kind == Kind::Ln {
                    lchunks.push(c);
                } else {
                    chunks.push(c);
                }
                s = e;
            }
        }
    }
    let est = (total_ev / 70) as usize;
    let nba = *K_NBA;
    let nbd = *K_NBD;
    let sha = 64 - nba.trailing_zeros();
    let shd = 64 - nbd.trailing_zeros();
    let cap = est / nt / nba + est / nt / nba / 4 + 64;
    let capd = est / nt / nbd + est / nt / nbd / 4 + 64;
    let est_ck = (total_ck / 46) as usize;
    let cap_ck = est_ck / nt / nba + est_ck / nt / nba / 4 + 64;
    let capd_ck = est_ck / nt / nbd + est_ck / nt / nbd / 4 + 64;

    // ---- links: parse (small), merge interners, then union-find on a side thread during parsing ----
    let mut lparts = run(
        nt,
        lchunks.len(),
        |tid| (Vec::<u8>::new(), Vec::<(u64, u64)>::new(), LocalIntern::new(), tid),
        |st, t| {
            let c = &lchunks[t];
            let (f, len) = &files[c.file];
            let (s, e) = read_chunk(f, *len, c, &mut st.0);
            let tid = st.3 as u64;
            let ln = &mut st.1;
            let li = &mut st.2;
            for_rows_fast_192::<2>(&st.0, s, e, |[a, bb]| {
                if !a.is_empty() && !bb.is_empty() {
                    let ka = unsafe { key_short_fast(a) }.unwrap_or_else(|| FALLBACK | (tid << 32) | li.intern(a) as u64);
                    let kb = unsafe { key_short_fast(bb) }.unwrap_or_else(|| FALLBACK | (tid << 32) | li.intern(bb) as u64);
                    ln.push((ka, kb));
                }
            });
        },
    );
    let mut lparts_clean: Vec<(Vec<u8>, Vec<(u64, u64)>, LocalIntern)> = lparts
        .into_iter()
        .map(|(b, ln, li, _)| (b, ln, li))
        .collect();
    let link_intern = merge_links_intern(&mut lparts_clean);
    log("links parsed");

    struct SA {
        tid: usize,
        buf: Vec<u8>,
        wc: Wc<Ev, 2>,
        wck: Wc<Ck, 2>,
        ev: Vec<Vec<Ev>>,
        ck: Vec<Vec<Ck>>,
        u_intern: LocalIntern,
        eid_arena: Vec<u8>,
        cid_arena: Vec<u8>,
        pur: Vec<(u32, u16)>,
    }
    let (ptab, mut sa) = std::thread::scope(|scope| {
        let lparts_ref = &lparts_clean;
        let li_ref = &link_intern;
        let uf = scope.spawn(move || {
            let n_l: usize = lparts_ref.iter().map(|s| s.1.len()).sum();
            let mut idx: FxHashMap<u64, u32> = FxHashMap::default();
            idx.reserve(n_l * 2);
            let mut keys: Vec<u64> = Vec::with_capacity(n_l * 2);
            let mut parent: Vec<u32> = Vec::with_capacity(n_l * 2);
            fn find(p: &mut [u32], mut x: u32) -> u32 {
                while p[x as usize] != x {
                    let gp = p[p[x as usize] as usize];
                    p[x as usize] = gp;
                    x = gp;
                }
                x
            }
            for s in lparts_ref {
                for &(a, b) in &s.1 {
                    let mut g = |k: u64| -> u32 {
                        *idx.entry(k).or_insert_with(|| {
                            keys.push(k);
                            parent.push(parent.len() as u32);
                            (keys.len() - 1) as u32
                        })
                    };
                    let ia = g(a);
                    let ib = g(b);
                    let ra = find(&mut parent, ia);
                    let rb = find(&mut parent, ib);
                    if ra != rb {
                        parent[ra as usize] = rb;
                    }
                }
            }
            drop(idx);
            let mut minkey: Vec<u64> = keys.clone();
            for i in 0..keys.len() {
                let r = find(&mut parent, i as u32) as usize;
                if cmp_key_gi(keys[i], minkey[r], li_ref) == Ordering::Less {
                    minkey[r] = keys[i];
                }
            }
            let mut t = PTab::new(keys.len());
            for i in 0..keys.len() {
                let r = find(&mut parent, i as u32) as usize;
                if minkey[r] != keys[i] {
                    t.insert(keys[i], minkey[r]);
                }
            }
            t
        });

        // ---- phase A: parse events and clicks ----
        let (evwin, evloc, ingtab, ckwin, ckloc) = (&evwin, &evloc, &ingtab, &ckwin, &ckloc);
        let products = &products;
        let campaigns = &campaigns;
        let files = &files;
        let maps = &maps;
        let chunks = &chunks;
        let sa = run(
            nt,
            chunks.len(),
            |tid| SA {
                tid,
                buf: vec![],
                wc: Wc::new(nba, cap),
                wck: Wc::new(nba, cap_ck),
                ev: vec![],
                ck: vec![],
                u_intern: LocalIntern::new(),
                eid_arena: vec![],
                cid_arena: vec![],
                pur: Vec::with_capacity(est / nt / 20 + 64),
            },
            |st, t| {
                let c = &chunks[t];
                if *K_NOCLK && c.kind == Kind::Ck {
                    return;
                }
                let (b, s, e): (&[u8], usize, usize) = match &maps[c.file] {
                    Some(mm) => {
                        let m = &mm[..];
                        let (s0, e0) = chunk_lines(m, c);
                        if e0 + 256 <= m.len() {
                            if *K_POP {
                                populate(m, s0, e0);
                            }
                            (m, s0, e0)
                        } else {
                            st.buf.clear();
                            st.buf.extend_from_slice(&m[s0..e0]);
                            st.buf.resize(e0 - s0 + 256, 0);
                            (&st.buf[..], 0, e0 - s0)
                        }
                    }
                    None => {
                        let (f, len) = &files[c.file];
                        let (s, e) = read_chunk(f, *len, c, &mut st.buf);
                        (&st.buf[..], s, e)
                    }
                };
                let tid = st.tid as u64;
                match c.kind {
                    Kind::Ev => {
                        let ev = &mut st.wc;
                        let u_intern = &mut st.u_intern;
                        let eid_arena = &mut st.eid_arena;
                        let pur = &mut st.pur;
                        for_rows_fast_192::<7>(b, s, e, #[inline(always)] |[eid, uid, ts, ing, et, pid, price]| {
                            if eid.is_empty() {
                                return;
                            }
                            let (ek, long_off) = match unsafe { key_short_fast(eid) } {
                                Some(k) => (k, 0u32),
                                None => (hash_bytes(eid) | FALLBACK, arena_push(eid_arena, eid)),
                            };
                            let ingest_s = unsafe { ingest_win(ing, ingtab, ws) };
                            let ingest = ingest_s as i32;
                            let mut e = Ev { eid: ek, uid: 0, ingest, ts: 0, aux: AUX_INVALID, long_off };
                            'v: {
                                if uid.is_empty() || pid.is_empty() {
                                    break 'v;
                                }
                                let Some(t) = (unsafe { parse_ts_win(ts, evwin, evloc) }) else { break 'v };
                                if t < ws || t >= we || ingest_s == i32::MAX as i64 || (ingest_s * 1000 + ws) - t > DAY {
                                    break 'v;
                                }
                                let typ = match et.len() {
                                    4 => {
                                        let w = u32::from_le_bytes(et.try_into().unwrap());
                                        if w == u32::from_le_bytes(*b"view") {
                                            0
                                        } else if w == u32::from_le_bytes(*b"cart") {
                                            1
                                        } else {
                                            break 'v;
                                        }
                                    }
                                    8 => {
                                        if u64::from_le_bytes(et.try_into().unwrap()) == u64::from_le_bytes(*b"purchase") {
                                            2
                                        } else {
                                            break 'v;
                                        }
                                    }
                                    _ => break 'v,
                                };
                                e.aux = if typ == 2 {
                                    let Some(p) = parse_price(price) else { break 'v };
                                    let cat = unsafe { products.get_fast(pid) }.unwrap_or(unknown_cat);
                                    pur.push((p, cat));
                                    1 + (((tid as u32) << 26) | (pur.len() as u32 - 1))
                                } else {
                                    0
                                };
                                e.ts = (t - ws) as u32;
                                e.uid = unsafe { key_short_fast(uid) }.unwrap_or_else(|| FALLBACK | (tid << 32) | u_intern.intern(uid) as u64);
                            }
                            ev.push(bucket_sh(ek, sha), e);
                        });
                    }
                    Kind::Ck => {
                        let ck = &mut st.wck;
                        let u_intern = &mut st.u_intern;
                        let cid_arena = &mut st.cid_arena;
                        for_rows_fast_192::<4>(b, s, e, #[inline(always)] |[cid, uid, camp, ts]| {
                            if cid.is_empty() || uid.is_empty() {
                                return;
                            }
                            let Some(t) = (unsafe { parse_ts_win(ts, ckwin, ckloc) }) else { return };
                            if t < cws || t >= we {
                                return;
                            }
                            let Some(ci) = (unsafe { campaigns.get_fast(camp) }) else { return };
                            let u = unsafe { key_short_fast(uid) }.unwrap_or_else(|| FALLBACK | (tid << 32) | u_intern.intern(uid) as u64);
                            let ck_id = unsafe { key_short_fast(cid) }.unwrap_or_else(|| FALLBACK | (tid << 32) | arena_push(cid_arena, cid) as u64);
                            ck.push(bucket_sh(u, sha), Ck { person: u, cid: ck_id, ts: (t - cws) as u32, camp: ci });
                        });
                    }
                    Kind::Ln => {}
                }
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_sfence();
                }
            },
        );
        let mut sa = sa;
        for s in sa.iter_mut() {
            let w = std::mem::replace(&mut s.wc, Wc::empty());
            s.ev = w.finish();
            let w = std::mem::replace(&mut s.wck, Wc::empty());
            s.ck = w.finish();
            s.buf = Vec::new();
        }
        (uf.join().unwrap(), sa)
    });
    drop(lparts_clean);
    let (global_intern, uremap) = {
        let sa_interns: Vec<LocalIntern> = sa.iter_mut().map(|s| std::mem::replace(&mut s.u_intern, LocalIntern::new())).collect();
        merge_phase_a_intern(link_intern, &sa_interns, nt)
    };
    log("parsed");
    if std::env::var("PARSEONLY").is_ok() {
        return;
    }

    // ---- phase B: dedup per eid bucket, map uid -> person, scatter by person bucket ----
    struct SB {
        tab: Vec<(u64, u32, u32)>,
        dead: Vec<bool>,
        wev: Wc<PEv, 4>,
        wck: Wc<Ck, 2>,
        ev: Vec<Vec<PEv>>,
        ck: Vec<Vec<Ck>>,
    }
    let ptab = &ptab;
    let uremap = &uremap;
    let sa_ref = &sa;
    let border: Vec<usize> = {
        let mut v: Vec<(usize, usize)> = (0..nba).map(|b| (sa_ref.iter().map(|s| s.ev[b].len()).sum::<usize>(), b)).collect();
        v.sort_unstable_by(|x, y| y.cmp(x));
        v.into_iter().map(|x| x.1).collect()
    };
    let mut sb = run(
        nt,
        nba,
        |_| SB { tab: vec![], dead: vec![], wev: Wc::new(nbd, capd), wck: Wc::new(nbd, capd_ck), ev: vec![], ck: vec![] },
        |st, t| {
            let b = border[t];
            let parts: Vec<&[Ev]> = sa_ref.iter().map(|s| &s.ev[b][..]).collect();
            let n: usize = parts.iter().map(|p| p.len()).sum();
            let tcap = (n * 2).next_power_of_two().max(16);
            if st.tab.len() < tcap {
                st.tab.resize(tcap, (0, 0, 0));
            }
            let generation = b as u32 + 1;
            st.dead.clear();
            st.dead.resize(n, false);
            let mask = tcap - 1;
            let mut offs = Vec::with_capacity(parts.len());
            let mut o = 0usize;
            for p in &parts {
                offs.push(o);
                o += p.len();
            }
            let at_with_tid = |g: usize| -> (usize, &Ev) {
                let mut pi = parts.len() - 1;
                while offs[pi] > g {
                    pi -= 1;
                }
                (pi, &parts[pi][g - offs[pi]])
            };
            let tab = &mut st.tab;
            let dead = &mut st.dead;
            let mut gi = 0usize;
            const PD: usize = 16;
            for (pi_t, p) in parts.iter().enumerate() {
                for (pi, e) in p.iter().enumerate() {
                    if pi + PD < p.len() {
                        let hk = (mix(p[pi + PD].eid) as usize) & mask;
                        prefetch(unsafe { tab.as_ptr().add(hk) });
                    }
                    let k = e.eid;
                    let mut h = (mix(k) as usize) & mask;
                    loop {
                        let slot = tab[h];
                        if slot.2 != generation {
                            tab[h] = (k, gi as u32, generation);
                            break;
                        }
                        if slot.0 == k {
                            let j = slot.1 as usize;
                            let (pj_t, ej) = at_with_tid(j);
                            if k & FALLBACK == 0
                                || arena_get(&sa_ref[pi_t].eid_arena, e.long_off) == arena_get(&sa_ref[pj_t].eid_arena, ej.long_off)
                            {
                                if e.ingest < ej.ingest {
                                    dead[j] = true;
                                    tab[h].1 = gi as u32;
                                } else {
                                    dead[gi] = true;
                                }
                                break;
                            }
                        }
                        h = (h + 1) & mask;
                    }
                    gi += 1;
                }
            }
            let mut gi = 0usize;
            for p in &parts {
                for (pi, e) in p.iter().enumerate() {
                    if pi + PD < p.len() {
                        let next_uid = resolve_uid(p[pi + PD].uid, uremap);
                        ptab.prefetch(next_uid);
                    }
                    if !dead[gi] && e.aux != AUX_INVALID {
                        let g_uid = resolve_uid(e.uid, uremap);
                        let person = ptab.get(g_uid);
                        st.wev.push(bucket_sh(person, shd), PEv { person, ts: e.ts, aux: e.aux });
                    }
                    gi += 1;
                }
            }
            for s in sa_ref.iter() {
                for c in &s.ck[b] {
                    let mut c = *c;
                    let g_uid = resolve_uid(c.person, uremap);
                    c.person = ptab.get(g_uid);
                    st.wck.push(bucket_sh(c.person, shd), c);
                }
            }
        },
    );
    let purs: Vec<Vec<(u32, u16)>> = sa.iter_mut().map(|s| std::mem::take(&mut s.pur)).collect();
    let cid_arenas: Vec<Vec<u8>> = sa.iter_mut().map(|s| std::mem::take(&mut s.cid_arena)).collect();
    drop(sa);
    for s in sb.iter_mut() {
        let w = std::mem::replace(&mut s.wev, Wc::empty());
        s.ev = w.finish();
        let w = std::mem::replace(&mut s.wck, Wc::empty());
        s.ck = w.finish();
    }
    let sc = sb;
    log("dedup+map");

    // ---- phase D: sessions + attribution + direct pq.rs output per worker ----
    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;
    let ud = outdir.join("user_daily");
    let cd = outdir.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap();
    std::fs::create_dir_all(&cd).unwrap();
    struct SD {
        ck: Vec<Ck>,
        evs: Vec<PEv>,
        cks: Vec<Ck>,
        k1: Vec<u64>,
        k2: Vec<u64>,
        map: FxHashMap<u64, u32>,
        li: Vec<u32>,
        cli: Vec<u32>,
        cnt: Vec<u32>,
        eoff: Vec<u32>,
        coff: Vec<u32>,
        users: Vec<UserRow>,
        camp: Vec<[i64; 4]>,
        sc: SessScG,
    }
    let dorder: Vec<usize> = {
        let mut v: Vec<(usize, usize)> = (0..nbd)
            .map(|b| (sc.iter().map(|s| s.ev[b].len() + s.ck[b].len()).sum::<usize>(), b))
            .collect();
        v.sort_unstable_by(|x, y| y.cmp(x));
        v.into_iter().map(|x| x.1).collect()
    };
    let gi_ref = &global_intern;
    let cat_names_ref = &cat_names;
    let cat_rank_ref = &cat_rank;
    let purs_ref = &purs;
    let cid_arenas_ref = &cid_arenas;
    let sc_ref = &sc;
    let ud_ref = &ud;
    let sd = run_fin(
        nt,
        nbd,
        |_| SD {
            ck: vec![],
            evs: vec![],
            cks: vec![],
            k1: vec![],
            k2: vec![],
            map: FxHashMap::default(),
            li: vec![],
            cli: vec![],
            cnt: vec![],
            eoff: vec![],
            coff: vec![],
            users: Vec::with_capacity(est / nt / 8 + 64),
            camp: vec![[0; 4]; ncamp * ndays],
            sc: SessScG { catsum: vec![0; ncat], covered: vec![(0, 0); organic as usize + 2], ..Default::default() },
        },
        |st, t| {
            let b = dorder[t];
            st.ck.clear();
            for s in sc_ref {
                st.ck.extend_from_slice(&s.ck[b]);
            }
            let parts: Vec<&[PEv]> = sc_ref.iter().map(|s| &s.ev[b][..]).collect();
            let n: usize = parts.iter().map(|p| p.len()).sum();
            if n == 0 {
                return;
            }
            st.map.clear();
            st.li.clear();
            st.li.reserve(n);
            st.cnt.clear();
            st.k1.clear();
            st.k1.reserve(n);
            let mut poff = [0u32; 64];
            let mut o = 0u32;
            for (pi, p) in parts.iter().enumerate() {
                poff[pi] = o;
                o += p.len() as u32;
                let pi_hi = (pi as u64) << 26;
                for (i, e) in p.iter().enumerate() {
                    let next = st.cnt.len() as u32;
                    let l = *st.map.entry(e.person).or_insert(next);
                    if l == next {
                        st.cnt.push(0);
                    }
                    st.cnt[l as usize] += 1;
                    st.li.push(l);
                    st.k1.push(((e.ts as u64) << 32) | pi_hi | i as u64);
                }
            }
            let m = st.cnt.len();
            st.eoff.clear();
            st.eoff.push(0);
            let mut acc = 0u32;
            for &c in &st.cnt {
                acc += c;
                st.eoff.push(acc);
            }
            st.k2.clear();
            st.k2.resize(n, 0);
            {
                let mut h = [[0u32; 2048]; 3];
                for &k in st.k1.iter() {
                    let t = (k >> 32) as u32;
                    h[0][(t & 2047) as usize] += 1;
                    h[1][((t >> 11) & 2047) as usize] += 1;
                    h[2][(t >> 22) as usize & 2047] += 1;
                }
                for p in 0..3 {
                    let mut sum = 0u32;
                    for x in h[p].iter_mut() {
                        let c = *x;
                        *x = sum;
                        sum += c;
                    }
                }
                for p in 0..3 {
                    let (src, dst) = if p % 2 == 0 { (&st.k1, &mut st.k2) } else { (&st.k2, &mut st.k1) };
                    let sh = 32 + 11 * p as u32;
                    let hp = &mut h[p];
                    for &k in src.iter() {
                        let d = ((k >> sh) & 2047) as usize;
                        dst[hp[d] as usize] = k;
                        hp[d] += 1;
                    }
                }
            }
            st.evs.clear();
            st.evs.reserve(n);
            unsafe { st.evs.set_len(n) };
            st.cnt.copy_from_slice(&st.eoff[..m]);
            for &k in st.k2.iter() {
                let pi = ((k >> 26) & 63) as usize;
                let i = (k & ((1 << 26) - 1)) as usize;
                let l = st.li[poff[pi] as usize + i] as usize;
                st.evs[st.cnt[l] as usize] = parts[pi][i];
                st.cnt[l] += 1;
            }
            st.cli.clear();
            st.cnt.clear();
            st.cnt.resize(m, 0);
            st.k1.clear();
            for (i, c) in st.ck.iter().enumerate() {
                let l = st.map.get(&c.person).copied().unwrap_or(u32::MAX);
                st.cli.push(l);
                if l != u32::MAX {
                    st.cnt[l as usize] += 1;
                    st.k1.push(((c.ts as u64) << 32) | i as u64);
                }
            }
            st.coff.clear();
            st.coff.push(0);
            let mut acc = 0u32;
            for &c in &st.cnt {
                acc += c;
                st.coff.push(acc);
            }
            let nc = acc as usize;
            st.k2.clear();
            st.k2.resize(nc, 0);
            {
                let mut h = [[0u32; 2048]; 3];
                for &k in st.k1.iter() {
                    let t = (k >> 32) as u32;
                    h[0][(t & 2047) as usize] += 1;
                    h[1][((t >> 11) & 2047) as usize] += 1;
                    h[2][(t >> 22) as usize & 2047] += 1;
                }
                for p in 0..3 {
                    let mut sum = 0u32;
                    for x in h[p].iter_mut() {
                        let c = *x;
                        *x = sum;
                        sum += c;
                    }
                }
                for p in 0..3 {
                    let (src, dst) = if p % 2 == 0 { (&st.k1, &mut st.k2) } else { (&st.k2, &mut st.k1) };
                    let sh = 32 + 11 * p as u32;
                    let hp = &mut h[p];
                    for &k in src.iter() {
                        let d = ((k >> sh) & 2047) as usize;
                        dst[hp[d] as usize] = k;
                        hp[d] += 1;
                    }
                }
            }
            st.cks.clear();
            st.cks.reserve(nc);
            unsafe { st.cks.set_len(nc) };
            st.cnt.copy_from_slice(&st.coff[..m]);
            for &k in st.k2.iter() {
                let i = (k & 0xFFFF_FFFF) as usize;
                let l = st.cli[i] as usize;
                st.cks[st.cnt[l] as usize] = st.ck[i];
                st.cnt[l] += 1;
            }
            process_bucket(
                &st.evs,
                &st.eoff,
                &st.cks,
                &st.coff,
                &mut st.users,
                &mut st.camp,
                &mut st.sc,
                ndays,
                ws,
                organic,
                cat_rank_ref,
                purs_ref,
                cid_arenas_ref,
            );
        },
        |st, tid| {
            if !st.users.is_empty() || tid == 0 {
                write_user_pq_general(&ud_ref.join(format!("part-{tid:03}.parquet")), &st.users, cat_names_ref, gi_ref).unwrap();
            }
        },
    );
    drop(sc);
    log("sessions");

    let mut camp = vec![[0i64; 4]; ncamp * ndays];
    for r in &sd {
        for (a, b) in camp.iter_mut().zip(r.camp.iter()) {
            for t in 0..4 {
                a[t] += b[t];
            }
        }
    }
    let mut c_id = vec![];
    let mut c_ch = vec![];
    let mut c_day = vec![];
    let mut c_v: Vec<[i64; 4]> = vec![];
    for c in 0..ncamp {
        for d in 0..ndays {
            let a = camp[c * ndays + d];
            if a[0] > 0 || a[3] > 0 {
                c_id.push(camp_ids[c].clone());
                c_ch.push(camp_ch[c].clone());
                c_day.push((base_day + d as i64) as i32);
                c_v.push(a);
            }
        }
    }
    fast::write_camp_pq(&cd.join("part-000.parquet"), &c_id, &c_ch, &c_day, &c_v).unwrap();
    log("done");
    std::process::exit(0);
}

#[derive(Default)]
struct SessScG {
    catsum: Vec<i64>,
    touched: Vec<u16>,
    covered: Vec<(u32, u32)>,
    pstamp: u32,
    s_nxt: Vec<u32>,
    s2_start: Vec<u32>,
    s2_day: Vec<u32>,
    s2_rev: Vec<i64>,
    s2_pur: Vec<u8>,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sess_flush_g(
    person: u64,
    day: i64,
    d_s: i64,
    d_e: i64,
    d_r: i64,
    catsum: &mut [i64],
    touched: &mut Vec<u16>,
    cat_rank: &[usize],
    users: &mut Vec<UserRow>,
) {
    let mut top: Option<u16> = None;
    if !touched.is_empty() {
        let mut t = touched[0];
        let mut tv = catsum[t as usize];
        for &c in &touched[1..] {
            let v = catsum[c as usize];
            if v > tv || (v == tv && cat_rank[c as usize] < cat_rank[t as usize]) {
                t = c;
                tv = v;
            }
        }
        top = Some(t);
        for &c in touched.iter() {
            catsum[c as usize] = 0;
        }
        touched.clear();
    }
    users.push(UserRow { person, day: day as i32, sessions: d_s, events: d_e, revenue: d_r, top });
}

#[allow(clippy::too_many_arguments)]
fn process_bucket(
    ev: &[PEv],
    eoff: &[u32],
    ck: &[Ck],
    coff: &[u32],
    users: &mut Vec<UserRow>,
    camp_out: &mut [[i64; 4]],
    sc: &mut SessScG,
    ndays: usize,
    ws: i64,
    organic: u16,
    cat_rank: &[usize],
    purs: &[Vec<(u32, u16)>],
    cid_arenas: &[Vec<u8>],
) {
    let base_day = ws.div_euclid(DAY);
    let off = (ws - base_day * DAY) as u64;
    let organic_u = organic as usize;
    for li in 0..eoff.len() - 1 {
        let (i, j) = (eoff[li] as usize, eoff[li + 1] as usize);
        let n = j - i;
        let person = ev[i].person;
        if sc.s2_start.len() < n {
            sc.s2_start.resize(n, 0);
            sc.s2_day.resize(n, 0);
            sc.s2_rev.resize(n, 0);
            sc.s2_pur.resize(n, 0);
        }
        let mut ns = 0usize;
        unsafe {
            let p = ev.as_ptr().add(i);
            let ss = sc.s2_start.as_mut_ptr();
            let sd = sc.s2_day.as_mut_ptr();
            let sr = sc.s2_rev.as_mut_ptr();
            let sp = sc.s2_pur.as_mut_ptr();
            let catsum = sc.catsum.as_mut_ptr();
            let mut cur_day = u32::MAX;
            let (mut d_s, mut d_e, mut d_r) = (0i64, 0i64, 0i64);
            let mut k = 0usize;
            while k < n {
                let st = (*p.add(k)).ts;
                let day = ((st as u64 + off) / DAY as u64) as u32;
                if day != cur_day {
                    if cur_day != u32::MAX {
                        sess_flush_g(person, base_day + cur_day as i64, d_s, d_e, d_r, &mut sc.catsum, &mut sc.touched, cat_rank, users);
                    }
                    cur_day = day;
                    d_s = 0;
                    d_e = 0;
                    d_r = 0;
                }
                let k0 = k;
                let mut last = st;
                let mut rev = 0i64;
                let mut pur = 0u8;
                let mut q = p.add(k);
                let end = p.add(n);
                while q < end {
                    let ts = (*q).ts;
                    if ts.wrapping_sub(last) > GAP as u32 {
                        break;
                    }
                    last = ts;
                    let aux = (*q).aux;
                    if aux != 0 {
                        let a = aux - 1;
                        let (price, cat) = *purs.get_unchecked((a >> 26) as usize).get_unchecked((a & ((1 << 26) - 1)) as usize);
                        pur = 1;
                        rev += price as i64;
                        let cs = &mut *catsum.add(cat as usize);
                        if *cs == 0 {
                            sc.touched.push(cat);
                        }
                        *cs += price as i64;
                    }
                    q = q.add(1);
                }
                k = q.offset_from(p) as usize;
                *ss.add(ns) = st;
                *sd.add(ns) = day;
                *sr.add(ns) = rev;
                *sp.add(ns) = pur;
                ns += 1;
                d_s += 1;
                d_e += (k - k0) as i64;
                d_r += rev;
            }
            sess_flush_g(person, base_day + cur_day as i64, d_s, d_e, d_r, &mut sc.catsum, &mut sc.touched, cat_rank, users);

            let cl = &ck[coff[li] as usize..coff[li + 1] as usize];
            let nc = cl.len();
            let cp = cl.as_ptr();
            let mut q = 0usize;
            for s in 0..ns {
                let st = *ss.add(s) as u64;
                let lim = st + WEEK as u64;
                while q < nc && (*cp.add(q)).ts as u64 <= lim {
                    q += 1;
                }
                let mut camp = organic_u;
                if q > 0 {
                    let t = (*cp.add(q - 1)).ts;
                    if t as u64 >= st {
                        let mut best = q - 1;
                        let mut r = q - 1;
                        while r > 0 && (*cp.add(r - 1)).ts == t {
                            r -= 1;
                            if cmp_cid((*cp.add(r)).cid, (*cp.add(best)).cid, cid_arenas) == Ordering::Less {
                                best = r;
                            }
                        }
                        camp = (*cp.add(best)).camp as usize;
                    }
                }
                let a = camp_out.get_unchecked_mut(camp * ndays + *sd.add(s) as usize);
                a[0] += 1;
                a[1] += *sp.add(s) as i64;
                a[2] += *sr.add(s);
            }
            if nc > 0 {
                sc.pstamp += 1;
                let pstamp = sc.pstamp;
                if sc.s_nxt.len() < ns {
                    sc.s_nxt.resize(ns, 0);
                }
                let nx = sc.s_nxt.as_mut_ptr();
                *nx.add(ns - 1) = ns as u32;
                for s in (0..ns - 1).rev() {
                    *nx.add(s) = if *sd.add(s + 1) != *sd.add(s) { (s + 1) as u32 } else { *nx.add(s + 1) };
                }
                let cov = sc.covered.as_mut_ptr();
                let (mut lo, mut hi) = (0usize, 0usize);
                for j in 0..nc {
                    let c = *cp.add(j);
                    let t = c.ts as u64;
                    while lo < ns && (*ss.add(lo) as u64 + WEEK as u64) < t {
                        lo += 1;
                    }
                    if hi < lo {
                        hi = lo;
                    }
                    while hi < ns && (*ss.add(hi) as u64) <= t {
                        hi += 1;
                    }
                    let kc = c.camp as usize;
                    let cv = &mut *cov.add(kc);
                    let (from, newhi) = if cv.0 == pstamp { (lo.max(cv.1 as usize), hi.max(cv.1 as usize)) } else { (lo, hi) };
                    *cv = (pstamp, newhi as u32);
                    let row = camp_out.as_mut_ptr().add(kc * ndays);
                    let mut x = from;
                    while x < hi {
                        let y = (*nx.add(x) as usize).min(hi);
                        (*row.add(*sd.add(x) as usize))[3] += (y - x) as i64;
                        x = y;
                    }
                }
            }
        }
    }
}

fn write_user_pq_general(path: &Path, rows: &[UserRow], cat_names: &[Vec<u8>], gi: &GlobalIntern) -> std::io::Result<()> {
    use crate::pq::{Col, Kind as K};
    let n = rows.len();
    let mut uid = Col::new("user_id", K::Str, n * 16);
    let mut day = Col::new("day", K::Date, n * 4);
    let mut ses = Col::new("sessions", K::I64, n * 8);
    let mut evc = Col::new("events", K::I64, n * 8);
    let mut rev = Col::new("revenue_cents", K::I64, n * 8);
    let mut top = Col::new("top_category", K::OptStr, n * 6);
    top.defs.reserve(n);
    uid.data.reserve(16);
    let mut kb = [0u8; 9];
    for r in rows {
        let k = r.person;
        if k & FALLBACK == 0 {
            #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
            {
                let c0 = (k >> 56) & 0x7F;
                let rest = unsafe { std::arch::x86_64::_pdep_u64(k, LOW7) }.swap_bytes();
                let z = rest.wrapping_sub(ONES) & !rest & HI8;
                let len = if c0 == 0 { 0 } else { 1 + (z.trailing_zeros() as usize >> 3).min(8) };
                uid.data.reserve(4 + 16);
                unsafe {
                    let p = uid.data.as_mut_ptr().add(uid.data.len());
                    std::ptr::write_unaligned(p as *mut u32, len as u32);
                    std::ptr::write_unaligned(p.add(4) as *mut u64, c0 | (rest << 8));
                    *p.add(12) = (rest >> 56) as u8;
                    uid.data.set_len(uid.data.len() + 4 + len);
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
            {
                uid.put_str(decode_short(k, &mut kb));
            }
        } else {
            uid.put_str(gi.get((k & !FALLBACK) as u32));
        }
        day.put_i32(r.day);
        ses.put_i64(r.sessions);
        evc.put_i64(r.events);
        rev.put_i64(r.revenue);
        match r.top {
            Some(t) => top.put_str(&cat_names[t as usize]),
            None => top.put_null(),
        }
    }
    let _ = &mut kb;
    crate::pq::write(path, n, &[uid, day, ses, evc, rev, top])
}
