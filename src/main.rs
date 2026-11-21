// Shopping Sessions pipeline (v3). Implements rules.md exactly.
use memchr::memchr;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::{LazyLock, Mutex};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 256;

// ---------- keys ----------
// Strings of at most 9 bytes, every byte in 1..=127, are packed 7 bits per byte
// (big-endian, 0 = padding) into 63 bits, which preserves byte-wise order.
// Anything else is interned and gets the top bit set; comparisons involving
// those go through bytes.
const FALLBACK: u64 = 1 << 63;
const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const HI8: u64 = 0x8080_8080_8080_8080;
const ONES: u64 = 0x0101_0101_0101_0101;

struct Interner {
    map: FxHashMap<Vec<u8>, u64>,
    strs: Vec<Vec<u8>>,
}
static INTERN: LazyLock<Mutex<Interner>> =
    LazyLock::new(|| Mutex::new(Interner { map: FxHashMap::default(), strs: vec![] }));

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
fn pack(x: u64, c9: u8, len: usize) -> Option<u64> {
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
fn key(s: &[u8]) -> u64 {
    let n = s.len();
    if n <= 9 {
        let mut b = [0u8; 9];
        b[..n].copy_from_slice(s);
        let x = u64::from_le_bytes(b[..8].try_into().unwrap());
        if let Some(k) = pack(x, b[8], n) {
            return k;
        }
    }
    intern(s)
}

/// Same as key, but may read up to 9 bytes past the start of s (caller guarantees padding).
#[inline(always)]
unsafe fn key_fast(s: &[u8]) -> u64 {
    let n = s.len();
    if n <= 9 {
        let p = s.as_ptr();
        let raw = unsafe { std::ptr::read_unaligned(p as *const u64) };
        let lm = if n >= 8 { u64::MAX } else { (1u64 << (8 * n)) - 1 };
        let c9 = if n == 9 { unsafe { *p.add(8) } } else { 0 };
        if let Some(k) = pack(raw & lm, c9, n) {
            return k;
        }
    }
    intern(s)
}

#[cold]
#[inline(never)]
fn intern(s: &[u8]) -> u64 {
    let mut g = INTERN.lock().unwrap();
    if let Some(&k) = g.map.get(s) {
        return k;
    }
    let k = FALLBACK | g.strs.len() as u64;
    g.strs.push(s.to_vec());
    g.map.insert(s.to_vec(), k);
    k
}

fn key_bytes(k: u64) -> Vec<u8> {
    if k & FALLBACK != 0 {
        INTERN.lock().unwrap().strs[(k & !FALLBACK) as usize].clone()
    } else {
        let mut out = Vec::with_capacity(9);
        for i in 0..9 {
            let c = ((k >> (7 * (8 - i))) & 0x7F) as u8;
            if c == 0 {
                break;
            }
            out.push(c);
        }
        out
    }
}

#[inline]
fn cmp_key(a: u64, b: u64) -> Ordering {
    if (a | b) & FALLBACK == 0 {
        a.cmp(&b)
    } else {
        key_bytes(a).cmp(&key_bytes(b))
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
    map: FxHashMap<u64, u16>,
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
            let k = key(s);
            if self.map.contains_key(&k) {
                return false;
            }
            self.map.insert(k, v);
            true
        }
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
            self.map.get(&key(s)).copied()
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
fn low_mask(k: usize) -> u64 {
    if k >= 64 { u64::MAX } else { (1u64 << k) - 1 }
}

/// Iterates rows of buf[s..e]. buf must have at least 128 readable bytes past e.
#[inline(always)]
fn for_rows_fast<const N: usize>(buf: &[u8], s: usize, e: usize, mut g: impl FnMut([&[u8]; N])) {
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
                g(a);
            }
            ls = ni;
            continue;
        };
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
            let lp = unsafe { base.add(ls) };
            let mut a: [&[u8]; N] = [&[]; N];
            let nq = mq[0].count_ones() + mq[1].count_ones();
            // branch-free unquoting: a field is quoted iff it starts and ends with '"' and has length >= 2
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
        if buf.len() < n + 128 {
            buf.resize(n + 128, 0);
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
                    }
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

// ---------- records ----------
#[derive(Clone, Copy)]
struct Ev {
    eid: u64,
    uid: u64,
    ingest: i32, // seconds relative to window start (clamped)
    ts: u32,     // ms since window start
    price: u32,
    cat: u16,
    typ: u8, // 0 view, 1 cart, 2 purchase, 255 invalid
    dead: bool,
}

#[derive(Clone, Copy)]
struct PEv {
    person: u64,
    ts: u32,
    aux: u32, // 0 = not a purchase, else 1 + (tid << 26 | idx) into the purchase side tables
}

#[derive(Clone, Copy)]
struct Ck {
    person: u64,
    cid: u64,
    ts: u32, // ms since click window start
    camp: u16,
}

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

fn buckets<T>(cap: usize) -> Vec<Vec<T>> {
    (0..NB).map(|_| Vec::with_capacity(cap)).collect()
}


fn bench(path: &str) {
    let mut b = std::fs::read(path).unwrap();
    let h = memchr(b'\n', &b).unwrap();
    let e0 = b.len();
    b.extend_from_slice(&[0u8; 128]);
    let full = &b[..];
    let b = &full[h + 1..e0];
    let (s0, e0) = (h + 1, e0);
    let mut seps = vec![];
    let mut quotes = vec![];
    let best = |name: &str, f: &mut dyn FnMut() -> u64| {
        let mut m = f64::MAX;
        let mut r = 0;
        for _ in 0..7 {
            let t = std::time::Instant::now();
            r = f();
            m = m.min(t.elapsed().as_secs_f64());
        }
        eprintln!("{name:12} {:.1} ms  ({:.0} MB/s) r={r}", m * 1e3, b.len() as f64 / m / 1e6);
    };
    best("scan", &mut || { scan(b, &mut seps, &mut quotes); seps.len() as u64 });
    best("rows_old", &mut || { let mut c = 0u64; for_rows::<7>(b, |a| c += a[0].len() as u64); c });
    best("rows_fast", &mut || { let mut c = 0u64; for_rows_fast::<7>(full, s0, e0, |a| c += a[0].len() as u64); c });
    best("+keys", &mut || { let mut c = 0u64; for_rows_fast::<7>(full, s0, e0, |a| c = c.wrapping_add(key(a[0]) ^ key(a[1]) ^ key(a[5]))); c });
    let mut dc = DC::new();
    let mut dc2 = DC::new();
    best("+ts_c", &mut || { let mut c = 0u64; for_rows_fast::<7>(full, s0, e0, |a| c = c.wrapping_add(parse_ts_c(a[2], &mut dc).unwrap_or(0) as u64 ^ parse_ts_c(a[3], &mut dc2).unwrap_or(0) as u64)); c });
    let mut bk: Vec<Vec<(u64, u64, u64)>> = buckets(1 << 14);
    best("+push", &mut || { for v in bk.iter_mut() { v.clear(); } for_rows_fast::<7>(full, s0, e0, |a| { let k = key(a[0]); bk[bucket(k)].push((k, key(a[1]), 0)); }); bk.iter().map(|v| v.len() as u64).sum() });
    best("+ts", &mut || { let mut c = 0u64; for_rows_fast::<7>(full, s0, e0, |a| c = c.wrapping_add(parse_ts(a[2]).unwrap_or(0) as u64 ^ parse_ts(a[3]).unwrap_or(0) as u64)); c });
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
        let k1 = key(&v);
        let k2 = unsafe { key_fast(&buf[..v.len()]) };
        if k1 != k2 || key_bytes(k1) != v {
            bad += 1;
            if bad < 10 { eprintln!("KEY MISMATCH {:?}", v); }
        }
    }
    eprintln!("fuzz done, mismatches: {bad}");
}

fn main() {
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
    if let Ok(p) = std::env::var("BENCH") { bench(&p); return; }
    let t0 = std::time::Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into()));
    let outdir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or("/output".into()));
    let nt: usize = std::env::var("NCPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))
        .max(1);
    let verbose = std::env::var("VERBOSE").is_ok();
    let last = std::cell::Cell::new((0f64, 0f64));
    let log = |m: &str| {
        if verbose {
            let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
            unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
            let u = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6;
            let sy = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 * 1e-6;
            let (pu, ps) = last.get();
            last.set((u, sy));
            eprintln!("{m:12} wall {:6.3}  user +{:.2}  sys +{:.2}", t0.elapsed().as_secs_f64(), u - pu, sy - ps);
        }
    };

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
        let mut start = [0i64; 8];
        for i in 0..8 {
            start[i] = days_from_civil(y0 + i as i64, 1, 1);
        }
        let _ = YT.set(year_tab(y0));
    }

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
            let mm = unsafe { memmap2::Mmap::map(&f) }.ok();
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
    let cap = est / nt / NB + est / nt / NB / 4 + 64;
    let est_ck = (total_ck / 46) as usize;
    let cap_ck = est_ck / nt / NB + est_ck / nt / NB / 4 + 64;

    // ---- links: parse (small), then union-find on a side thread during parsing ----
    let lparts = run(
        nt,
        lchunks.len(),
        |_| (Vec::<u8>::new(), Vec::<(u64, u64)>::new()),
        |st, t| {
            let c = &lchunks[t];
            let (f, len) = &files[c.file];
            let (s, e) = read_chunk(f, *len, c, &mut st.0);
            let ln = &mut st.1;
            for_rows_fast::<2>(&st.0, s, e, |[a, bb]| {
                if !a.is_empty() && !bb.is_empty() {
                    ln.push(unsafe { (key_fast(a), key_fast(bb)) });
                }
            });
        },
    );
    log("links parsed");

    struct SA {
        buf: Vec<u8>,
        ev: Vec<Vec<Ev>>,
        ck: Vec<Vec<Ck>>,
    }
    let (ptab, sa) = std::thread::scope(|scope| {
        let lparts = &lparts;
        let uf = scope.spawn(move || {
            let n_l: usize = lparts.iter().map(|s| s.1.len()).sum();
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
            for s in lparts {
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
                if cmp_key(keys[i], minkey[r]) == Ordering::Less {
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
        let sa = run(
            nt,
            chunks.len(),
            |_| SA { buf: vec![], ev: buckets(cap), ck: buckets(cap_ck) },
            |st, t| {
                let c = &chunks[t];
                // zero-copy from the page cache (/data is tmpfs on the scorer); the tail
                // chunk of a file is copied into a padded buffer for the 128-byte look-ahead
                let (b, s, e): (&[u8], usize, usize) = match &maps[c.file] {
                    Some(mm) => {
                        let m = &mm[..];
                        let (s0, e0) = chunk_lines(m, c);
                        if e0 + 128 <= m.len() {
                            populate(m, s0, e0);
                            (m, s0, e0)
                        } else {
                            st.buf.clear();
                            st.buf.extend_from_slice(&m[s0..e0]);
                            st.buf.resize(e0 - s0 + 128, 0);
                            (&st.buf[..], 0, e0 - s0)
                        }
                    }
                    None => {
                        let (f, len) = &files[c.file];
                        let (s, e) = read_chunk(f, *len, c, &mut st.buf);
                        (&st.buf[..], s, e)
                    }
                };
                match c.kind {
                    Kind::Ev => {
                        let ev = &mut st.ev;
                        let mut dc1 = DC::new();
                        let mut dc2 = DC::new();
                        for_rows_fast::<7>(b, s, e, #[inline(always)] |[eid, uid, ts, ing, et, pid, price]| {
                            if eid.is_empty() {
                                return;
                            }
                            let ek = unsafe { key_fast(eid) };
                            let ingest_ms = unsafe { parse_ingest(ing) }.unwrap_or(i64::MAX);
                            let ingest = (ingest_ms.saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                            let mut e = Ev { eid: ek, uid: 0, ingest, ts: 0, price: 0, cat: 0, typ: 255, dead: false };
                            'v: {
                                if uid.is_empty() || pid.is_empty() {
                                    break 'v;
                                }
                                let Some(t) = (unsafe { parse_ts_fast(ts) }) else { break 'v };
                                if t < ws || t >= we || ingest_ms.saturating_sub(t) > DAY {
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
                                if typ == 2 {
                                    match parse_price(price) {
                                        Some(p) => e.price = p,
                                        None => break 'v,
                                    }
                                }
                                e.ts = (t - ws) as u32;
                                e.typ = typ;
                                if typ == 2 {
                                    e.cat = products.get(pid).unwrap_or(unknown_cat);
                                }
                                e.uid = unsafe { key_fast(uid) };
                            }
                            ev[bucket(ek)].push(e);
                        });
                    }
                    Kind::Ck => {
                        let ck = &mut st.ck;
                        let mut dc1 = DC::new();
                        for_rows_fast::<4>(b, s, e, #[inline(always)] |[cid, uid, camp, ts]| {
                            if cid.is_empty() || uid.is_empty() {
                                return;
                            }
                            let Some(t) = (unsafe { parse_ts_fast(ts) }) else { return };
                            if t < cws || t >= we {
                                return;
                            }
                            let Some(ci) = campaigns.get(camp) else { return };
                            let u = unsafe { key_fast(uid) };
                            ck[bucket(u)].push(Ck { person: u, cid: unsafe { key_fast(cid) }, ts: (t - cws) as u32, camp: ci });
                        });
                    }
                    Kind::Ln => {}
                }
            },
        );
        (uf.join().unwrap(), sa)
    });
    drop(lparts);
    log("parsed");
    if std::env::var("PARSEONLY").is_ok() {
        return;
    }

    // ---- phase B: dedup per eid bucket, map uid -> person, scatter by person bucket ----
    struct SB {
        tid: usize,
        pur: Vec<(u32, u16)>,
        tab: Vec<(u64, u32, u32)>,
        dead: Vec<bool>,
        ev: Vec<Vec<PEv>>,
        ck: Vec<Vec<Ck>>,
    }
    let ptab = &ptab;
    let border: Vec<usize> = {
        let mut v: Vec<(usize, usize)> = (0..NB).map(|b| (sa.iter().map(|s| s.ev[b].len()).sum::<usize>(), b)).collect();
        v.sort_unstable_by(|x, y| y.cmp(x));
        v.into_iter().map(|x| x.1).collect()
    };
    let mut sb = run(
        nt,
        NB,
        |tid| SB { tid, pur: Vec::with_capacity(est / nt / 20 + 64), tab: vec![], dead: vec![], ev: buckets(cap), ck: buckets(cap_ck) },
        |st, t| {
            let b = border[t];
            // virtual concatenation of every thread's bucket b
            let parts: Vec<&[Ev]> = sa.iter().map(|s| &s.ev[b][..]).collect();
            let n: usize = parts.iter().map(|p| p.len()).sum();
            // generation-tagged table: slots from other buckets count as empty, so no clearing
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
            let at = |g: usize| -> &Ev {
                let mut pi = parts.len() - 1;
                while offs[pi] > g {
                    pi -= 1;
                }
                &parts[pi][g - offs[pi]]
            };
            let tab = &mut st.tab;
            let dead = &mut st.dead;
            let mut gi = 0usize;
            const PD: usize = 16;
            for p in &parts {
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
                            if e.ingest < at(j).ingest {
                                dead[j] = true;
                                tab[h].1 = gi as u32;
                            } else {
                                dead[gi] = true;
                            }
                            break;
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
                        ptab.prefetch(p[pi + PD].uid);
                    }
                    if !dead[gi] && e.typ != 255 {
                        let person = ptab.get(e.uid);
                        let aux = if e.typ == 2 {
                            st.pur.push((e.price, e.cat));
                            1 + (((st.tid as u32) << 26) | (st.pur.len() as u32 - 1))
                        } else {
                            0
                        };
                        st.ev[bucket(person)].push(PEv { person, ts: e.ts, aux });
                    }
                    gi += 1;
                }
            }
            for s in sa.iter() {
                for c in &s.ck[b] {
                    let mut c = *c;
                    c.person = ptab.get(c.person);
                    st.ck[bucket(c.person)].push(c);
                }
            }
        },
    );
    drop(sa);
    let purs: Vec<Vec<(u32, u16)>> = sb.iter_mut().map(|s| std::mem::take(&mut s.pur)).collect();
    let sc = sb;
    log("dedup+map");

    // ---- phase D: sessions + attribution per person bucket ----
    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;
    struct SD {
        ev: Vec<PEv>,
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
    }
    let dorder: Vec<usize> = {
        let mut v: Vec<(usize, usize)> = (0..NB)
            .map(|b| (sc.iter().map(|s| s.ev[b].len() + s.ck[b].len()).sum::<usize>(), b))
            .collect();
        v.sort_unstable_by(|x, y| y.cmp(x));
        v.into_iter().map(|x| x.1).collect()
    };
    let sd = run(
        nt,
        NB,
        |_| SD { ev: vec![], ck: vec![], evs: vec![], cks: vec![], k1: vec![], k2: vec![], map: FxHashMap::default(), li: vec![], cli: vec![], cnt: vec![], eoff: vec![], coff: vec![], users: Vec::with_capacity(est / nt / 8 + 64), camp: vec![[0; 4]; ncamp * ndays] },
        |st, t| {
            let b = dorder[t];
            st.ck.clear();
            for s in &sc {
                st.ck.extend_from_slice(&s.ck[b]);
            }
            let parts: Vec<&[PEv]> = sc.iter().map(|s| &s.ev[b][..]).collect();
            // local person index (order of first appearance)
            let n: usize = parts.iter().map(|p| p.len()).sum();
            st.map.clear();
            st.li.clear();
            st.cnt.clear();
            for e in parts.iter().flat_map(|p| p.iter()) {
                let next = st.cnt.len() as u32;
                let l = *st.map.entry(e.person).or_insert(next);
                if l == next {
                    st.cnt.push(0);
                }
                st.cnt[l as usize] += 1;
                st.li.push(l);
            }
            let m = st.cnt.len();
            st.eoff.clear();
            st.eoff.push(0);
            let mut acc = 0u32;
            for &c in &st.cnt {
                acc += c;
                st.eoff.push(acc);
            }
            // LSD radix sort of (ts, index) by ts: 3 passes of 11 bits
            st.k1.clear();
            for (pi, p) in parts.iter().enumerate() {
                st.k1.extend(p.iter().enumerate().map(|(i, e)| ((e.ts as u64) << 32) | ((pi as u64) << 26) | i as u64));
            }
            let mut poff = [0u32; 64];
            {
                let mut o = 0u32;
                for (pi, p) in parts.iter().enumerate() {
                    poff[pi] = o;
                    o += p.len() as u32;
                }
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
            // after 3 passes the sorted keys are in k2
            // stable counting placement by person: groups come out ts-sorted
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
            // clicks: same person order; clicks of persons without events are dropped.
            // Radix-sort by ts, then stable placement by person: groups come out ts-sorted.
            // (Ties in ts are resolved by click_id at credit time.)
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
            process_bucket(&st.evs, &st.eoff, &st.cks, &st.coff, &mut st.users, &mut st.camp, ncat, ndays, ws, cws, organic, &cat_rank, &purs);
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

    let ud = outdir.join("user_daily");
    let cd = outdir.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap();
    std::fs::create_dir_all(&cd).unwrap();
    std::thread::scope(|scope| {
        for (g, s) in sd.iter().enumerate() {
            let ud = &ud;
            let cat_names = &cat_names;
            if !s.users.is_empty() || g == 0 {
                scope.spawn(move || write_user(&ud.join(format!("part-{g:03}.parquet")), &s.users, cat_names));
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
        write_camp(&cd.join("part-000.parquet"), &c_id, &c_ch, &c_day, &c_v);
    });
    log("done");
    std::process::exit(0);
}

#[allow(clippy::too_many_arguments)]
fn process_bucket(
    ev: &[PEv],
    eoff: &[u32],
    ck: &[Ck],
    coff: &[u32],
    users: &mut Vec<UserRow>,
    camp_out: &mut [[i64; 4]],
    ncat: usize,
    ndays: usize,
    ws: i64,
    cws: i64,
    organic: u16,
    cat_rank: &[usize],
    purs: &[Vec<(u32, u16)>],
) {
    let base_day = ws.div_euclid(DAY);
    let mut catsum = vec![0i64; ncat];
    let mut touched: Vec<u16> = vec![];
    let mut s_start: Vec<i64> = vec![];
    let mut s_day: Vec<usize> = vec![];
    let mut s_rev: Vec<i64> = vec![];
    let mut s_pur: Vec<bool> = vec![];
    let mut covered: Vec<(u32, u32)> = vec![(0, 0); organic as usize + 2];
    let mut pstamp: u32 = 0;
    let mut cts: Vec<i64> = vec![];
    for li in 0..eoff.len() - 1 {
        let (i, j) = (eoff[li] as usize, eoff[li + 1] as usize);
        let p = ev[i].person;
        s_start.clear();
        s_day.clear();
        s_rev.clear();
        s_pur.clear();
        let mut cur_day: i64 = i64::MIN;
        let (mut d_s, mut d_e, mut d_r) = (0i64, 0i64, 0i64);
        let flush = |day: i64, d_s: i64, d_e: i64, d_r: i64, catsum: &mut Vec<i64>, touched: &mut Vec<u16>, users: &mut Vec<UserRow>| {
            let mut top: Option<u16> = None;
            for &c in touched.iter() {
                top = match top {
                    None => Some(c),
                    Some(t) => {
                        let (v, tv) = (catsum[c as usize], catsum[t as usize]);
                        if v > tv || (v == tv && cat_rank[c as usize] < cat_rank[t as usize]) {
                            Some(c)
                        } else {
                            Some(t)
                        }
                    }
                };
            }
            for &c in touched.iter() {
                catsum[c as usize] = 0;
            }
            touched.clear();
            users.push(UserRow { person: p, day: day as i32, sessions: d_s, events: d_e, revenue: d_r, top });
        };
        let mut k = i;
        while k < j {
            let start = ev[k].ts as i64 + ws;
            let mut last = ev[k].ts as i64;
            let mut n = 0i64;
            let mut rev = 0i64;
            let mut pur = false;
            let day = start.div_euclid(DAY);
            if day != cur_day {
                if cur_day != i64::MIN {
                    flush(cur_day, d_s, d_e, d_r, &mut catsum, &mut touched, users);
                }
                cur_day = day;
                d_s = 0;
                d_e = 0;
                d_r = 0;
            }
            while k < j && ev[k].ts as i64 - last <= GAP {
                let e = ev[k];
                last = e.ts as i64;
                n += 1;
                if e.aux != 0 {
                    let a = e.aux - 1;
                    let (price, cat) = purs[(a >> 26) as usize][(a & ((1 << 26) - 1)) as usize];
                    pur = true;
                    rev += price as i64;
                    if catsum[cat as usize] == 0 {
                        touched.push(cat);
                    }
                    catsum[cat as usize] += price as i64;
                }
                k += 1;
            }
            d_s += 1;
            d_e += n;
            d_r += rev;
            s_start.push(start);
            s_day.push((day - base_day) as usize);
            s_rev.push(rev);
            s_pur.push(pur);
        }
        flush(cur_day, d_s, d_e, d_r, &mut catsum, &mut touched, users);

        let cl = &ck[coff[li] as usize..coff[li + 1] as usize];
        cts.clear();
        cts.extend(cl.iter().map(|c| c.ts as i64 + cws));
        let mut q = 0usize;
        for s in 0..s_start.len() {
            let st = s_start[s];
            while q < cl.len() && cts[q] <= st {
                q += 1;
            }
            let mut camp = organic;
            if q > 0 && st - cts[q - 1] <= WEEK {
                let t = cts[q - 1];
                let mut best = q - 1;
                let mut r = q - 1;
                while r > 0 && cts[r - 1] == t {
                    r -= 1;
                    if cmp_key(cl[r].cid, cl[best].cid) == Ordering::Less {
                        best = r;
                    }
                }
                camp = cl[best].camp;
            }
            let a = &mut camp_out[camp as usize * ndays + s_day[s]];
            a[0] += 1;
            a[1] += s_pur[s] as i64;
            a[2] += s_rev[s];
        }
        if !cl.is_empty() && !s_start.is_empty() {
            // assists: clicks are ts-sorted, so the session window [t, t + 7d] only moves right.
            // covered[k] = first session index not yet counted for campaign k (per person).
            pstamp += 1;
            let ns = s_start.len();
            let (mut lo, mut hi) = (0usize, 0usize);
            for (c, &t) in cl.iter().zip(cts.iter()) {
                while lo < ns && s_start[lo] < t {
                    lo += 1;
                }
                if hi < lo {
                    hi = lo;
                }
                while hi < ns && s_start[hi] <= t + WEEK {
                    hi += 1;
                }
                let k = c.camp as usize;
                let cv = &mut covered[k];
                let from = if cv.0 == pstamp { lo.max(cv.1 as usize) } else { lo };
                *cv = (pstamp, hi.max(if cv.0 == pstamp { cv.1 as usize } else { 0 }) as u32);
                let mut x = from;
                while x < hi {
                    let d = s_day[x];
                    let mut y = x + 1;
                    while y < hi && s_day[y] == d {
                        y += 1;
                    }
                    camp_out[k * ndays + d][3] += (y - x) as i64;
                    x = y;
                }
            }
        }
    }
}

// ---------- parquet ----------
use parquet::basic::Compression;
use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::sync::Arc;

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build(),
    )
}

fn write_user(path: &Path, rows: &[UserRow], cat_names: &[Vec<u8>]) {
    let schema = Arc::new(
        parse_message_type(
            "message user_daily { required binary user_id (UTF8); required int32 day (DATE); required int64 sessions; required int64 events; required int64 revenue_cents; optional binary top_category (UTF8); }",
        )
        .unwrap(),
    );
    let f = std::io::BufWriter::with_capacity(1 << 20, File::create(path).unwrap());
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
    let cats: Vec<ByteArray> = cat_names.iter().map(|c| ByteArray::from(c.clone())).collect();
    while let Some(mut c) = rg.next_column().unwrap() {
        match col {
            0 => {
                let v: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(key_bytes(r.person))).collect();
                c.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
            }
            1 => {
                let v: Vec<i32> = rows.iter().map(|r| r.day).collect();
                c.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
            }
            2 | 3 | 4 => {
                let v: Vec<i64> = rows
                    .iter()
                    .map(|r| match col {
                        2 => r.sessions,
                        3 => r.events,
                        _ => r.revenue,
                    })
                    .collect();
                c.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
            }
            _ => {
                let def: Vec<i16> = rows.iter().map(|r| r.top.is_some() as i16).collect();
                let v: Vec<ByteArray> = rows.iter().filter_map(|r| r.top.map(|t| cats[t as usize].clone())).collect();
                c.typed::<ByteArrayType>().write_batch(&v, Some(&def), None).unwrap();
            }
        }
        c.close().unwrap();
        col += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_camp(path: &Path, ids: &[Vec<u8>], ch: &[Vec<u8>], day: &[i32], v: &[[i64; 4]]) {
    let schema = Arc::new(
        parse_message_type(
            "message campaign_daily { required binary campaign_id (UTF8); required binary channel (UTF8); required int32 day (DATE); required int64 sessions; required int64 purchasing_sessions; required int64 revenue_cents; required int64 assisted_sessions; }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
    while let Some(mut c) = rg.next_column().unwrap() {
        match col {
            0 | 1 => {
                let src = if col == 0 { ids } else { ch };
                let b: Vec<ByteArray> = src.iter().map(|s| ByteArray::from(s.clone())).collect();
                c.typed::<ByteArrayType>().write_batch(&b, None, None).unwrap();
            }
            2 => {
                c.typed::<Int32Type>().write_batch(day, None, None).unwrap();
            }
            _ => {
                let k = col - 3;
                let b: Vec<i64> = v.iter().map(|a| a[k]).collect();
                c.typed::<Int64Type>().write_batch(&b, None, None).unwrap();
            }
        }
        c.close().unwrap();
        col += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}
