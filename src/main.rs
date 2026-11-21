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
// Strings of at most 10 chars from [0-9_a-z] are packed 6 bits per char
// (0 = padding), which preserves byte-wise order. Anything else is interned
// and gets the top bit set; comparisons involving those go through bytes.
static ALPHA: LazyLock<[u8; 256]> = LazyLock::new(|| {
    let mut t = [0u8; 256];
    let mut v = 1u8;
    for c in b'0'..=b'9' {
        t[c as usize] = v;
        v += 1;
    }
    t[b'_' as usize] = v;
    v += 1;
    for c in b'a'..=b'z' {
        t[c as usize] = v;
        v += 1;
    }
    t
});
const DEC: &[u8; 38] = b"\x000123456789_abcdefghijklmnopqrstuvwxyz";
const FALLBACK: u64 = 1 << 63;

struct Interner {
    map: FxHashMap<Vec<u8>, u64>,
    strs: Vec<Vec<u8>>,
}
static INTERN: LazyLock<Mutex<Interner>> =
    LazyLock::new(|| Mutex::new(Interner { map: FxHashMap::default(), strs: vec![] }));

#[inline(always)]
fn key(s: &[u8]) -> u64 {
    if s.len() <= 10 {
        let t = &*ALPHA;
        let mut v: u64 = 0;
        let mut bad = 0u8;
        for &c in s {
            let d = t[c as usize];
            bad |= (d == 0) as u8;
            v = (v << 6) | d as u64;
        }
        if bad != 0 {
            return intern(s);
        }
        v << (6 * (10 - s.len()))
    } else {
        intern(s)
    }
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
        let mut out = Vec::with_capacity(10);
        for i in (0..10).rev() {
            let d = ((k >> (6 * i)) & 63) as usize;
            if d == 0 {
                break;
            }
            out.push(DEC[d]);
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
fn masks128(p: &[u8]) -> (u128, u128, u128) {
    debug_assert!(p.len() >= 128);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let vc = _mm256_set1_epi8(b',' as i8);
        let vn = _mm256_set1_epi8(b'\n' as i8);
        let vq = _mm256_set1_epi8(b'"' as i8);
        let mut c = 0u128;
        let mut n = 0u128;
        let mut q = 0u128;
        for k in 0..4 {
            let v = _mm256_loadu_si256(p.as_ptr().add(32 * k) as *const __m256i);
            c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vc)) as u32 as u128) << (32 * k);
            n |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vn)) as u32 as u128) << (32 * k);
            q |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vq)) as u32 as u128) << (32 * k);
        }
        return (c, n, q);
    }
    #[allow(unreachable_code)]
    {
        let (mut c, mut n, mut q) = (0u128, 0u128, 0u128);
        for i in 0..128 {
            match p[i] {
                b',' => c |= 1 << i,
                b'\n' => n |= 1 << i,
                b'"' => q |= 1 << i,
                _ => {}
            }
        }
        (c, n, q)
    }
}

/// Iterates rows of buf[s..e]. buf must have at least 128 readable bytes past e.
#[inline(always)]
fn for_rows_fast<const N: usize>(buf: &[u8], s: usize, e: usize, mut g: impl FnMut([&[u8]; N])) {
    assert!(buf.len() >= e + 128);
    let mut ls = s;
    let mut f = [(0usize, 0usize); N];
    while ls < e {
        let lim = e - ls;
        let (mc, mn, mq) = masks128(&buf[ls..ls + 128]);
        let valid = if lim < 128 { (1u128 << lim) - 1 } else { u128::MAX };
        let mn = mn & valid;
        let le_off;
        if mn != 0 {
            le_off = mn.trailing_zeros() as usize;
        } else if lim <= 128 {
            le_off = lim;
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
        }
        let lm = if le_off >= 128 { u128::MAX } else { (1u128 << le_off) - 1 };
        let mut commas = mc & lm;
        if commas.count_ones() as usize == N - 1 {
            let quotes = mq & lm;
            let mut a: [&[u8]; N] = [&[]; N];
            let mut fs = 0usize;
            if quotes == 0 {
                for k in 0..N {
                    let fe = if k < N - 1 {
                        let t = commas.trailing_zeros() as usize;
                        commas &= commas - 1;
                        t
                    } else {
                        le_off
                    };
                    a[k] = &buf[ls + fs..ls + fe];
                    fs = fe + 1;
                }
                g(a);
            } else {
                let mut cand = 0u32;
                for k in 0..N {
                    let fe = if k < N - 1 {
                        let t = commas.trailing_zeros() as usize;
                        commas &= commas - 1;
                        t
                    } else {
                        le_off
                    };
                    if fe >= fs + 2 && (quotes >> fs) & 1 == 1 && (quotes >> (fe - 1)) & 1 == 1 {
                        a[k] = &buf[ls + fs + 1..ls + fe - 1];
                        cand += 1;
                    } else {
                        a[k] = &buf[ls + fs..ls + fe];
                    }
                    fs = fe + 1;
                }
                if quotes.count_ones() == 2 * cand {
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

/// Runs `ntasks` tasks over `nt` threads; each thread owns a state.
fn run<S: Send>(nt: usize, ntasks: usize, init: impl Fn() -> S + Sync, f: impl Fn(&mut S, usize) + Sync) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|_| {
                sc.spawn(|| {
                    let mut s = init();
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
    price: u32,
    cat: u16,
    typ: u8,
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
    best("+ts", &mut || { let mut c = 0u64; for_rows_fast::<7>(full, s0, e0, |a| c = c.wrapping_add(parse_ts(a[2]).unwrap_or(0) as u64 ^ parse_ts(a[3]).unwrap_or(0) as u64)); c });
}

fn main() {
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

    // ---- phase A: parse everything ----
    let mut files: Vec<(File, u64)> = vec![];
    let mut chunks: Vec<Chunk> = vec![];
    let mut total_ev: u64 = 0;
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
            let fi = files.len();
            files.push((f, len));
            let mut s = 0;
            while s < len {
                let e = (s + (4 << 20)).min(len);
                chunks.push(Chunk { kind, file: fi, s, e });
                s = e;
            }
        }
    }
    let est = (total_ev / 72) as usize;
    let cap = est / nt / NB + est / nt / NB / 8 + 16;

    struct SA {
        seps: Vec<u32>,
        quotes: Vec<u32>,
        buf: Vec<u8>,
        ev: Vec<Vec<Ev>>,
        ck: Vec<Vec<Ck>>,
        ln: Vec<(u64, u64)>,
    }
    let sa = run(
        nt,
        chunks.len(),
        || SA { seps: vec![], quotes: vec![], buf: vec![], ev: buckets(cap), ck: buckets(cap / 4), ln: vec![] },
        |st, t| {
            let c = &chunks[t];
            let (f, len) = &files[c.file];
            let (s, e) = read_chunk(f, *len, c, &mut st.buf);
            let b = &st.buf[..];
            match c.kind {
                Kind::Ev => {
                    let ev = &mut st.ev;
                    let mut dc1 = DC::new();
                    let mut dc2 = DC::new();
                    for_rows_fast::<7>(b, s, e, |[eid, uid, ts, ing, et, pid, price]| {
                        if eid.is_empty() {
                            return;
                        }
                        let ek = key(eid);
                        let ingest_ms = parse_ts_c(ing, &mut dc2).unwrap_or(i64::MAX);
                        let ingest = (ingest_ms.saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                        let mut e = Ev { eid: ek, uid: 0, ingest, ts: 0, price: 0, cat: 0, typ: 255, dead: false };
                        'v: {
                            if uid.is_empty() || pid.is_empty() {
                                break 'v;
                            }
                            let Some(t) = parse_ts_c(ts, &mut dc1) else { break 'v };
                            if t < ws || t >= we || ingest_ms.saturating_sub(t) > DAY {
                                break 'v;
                            }
                            let typ = match et {
                                b"view" => 0,
                                b"cart" => 1,
                                b"purchase" => 2,
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
                            e.cat = products.get(pid).unwrap_or(unknown_cat);
                            e.uid = key(uid);
                        }
                        ev[bucket(ek)].push(e);
                    });
                }
                Kind::Ck => {
                    let ck = &mut st.ck;
                    let mut dc1 = DC::new();
                    for_rows_fast::<4>(b, s, e, |[cid, uid, camp, ts]| {
                        if cid.is_empty() || uid.is_empty() {
                            return;
                        }
                        let Some(t) = parse_ts_c(ts, &mut dc1) else { return };
                        if t < cws || t >= we {
                            return;
                        }
                        let Some(ci) = campaigns.get(camp) else { return };
                        let u = key(uid);
                        ck[bucket(u)].push(Ck { person: u, cid: key(cid), ts: (t - cws) as u32, camp: ci });
                    });
                }
                Kind::Ln => {
                    let ln = &mut st.ln;
                    for_rows_fast::<2>(b, s, e, |[a, bb]| {
                        if !a.is_empty() && !bb.is_empty() {
                            ln.push((key(a), key(bb)));
                        }
                    });
                }
            }
        },
    );
    log("parsed");
    if std::env::var("PARSEONLY").is_ok() { let n: usize = sa.iter().map(|s| s.ev.iter().map(|v| v.len()).sum::<usize>()).sum(); eprintln!("events {n}"); return; }
    if std::env::var("PARSEONLY").is_ok() { let n: usize = sa.iter().map(|s| s.ev.iter().map(|v| v.len()).sum::<usize>()).sum(); eprintln!("events {n}"); return; }

    // ---- links: union-find, alongside dedup ----
    let (person_maps, sb) = std::thread::scope(|scope| {
        let sa = &sa;
        let links = scope.spawn(move || {
            let n_l: usize = sa.iter().map(|s| s.ln.len()).sum();
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
            for s in sa {
                for &(a, b) in &s.ln {
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
            let mut maps: Vec<FxHashMap<u64, u64>> = (0..NB).map(|_| FxHashMap::default()).collect();
            for i in 0..keys.len() {
                let r = find(&mut parent, i as u32) as usize;
                if minkey[r] != keys[i] {
                    maps[bucket(keys[i])].insert(keys[i], minkey[r]);
                }
            }
            maps
        });

        // ---- phase B: dedup per eid bucket -> scatter by uid bucket ----
        struct SB {
            v: Vec<Ev>,
            tab: Vec<(u64, u32)>,
            out: Vec<Vec<PEv>>,
        }
        let sb = run(
            nt,
            NB,
            || SB { v: vec![], tab: vec![], out: buckets(cap) },
            |st, b| {
                st.v.clear();
                for s in sa {
                    st.v.extend_from_slice(&s.ev[b]);
                }
                let n = st.v.len();
                let tcap = (n * 2).next_power_of_two().max(16);
                st.tab.clear();
                st.tab.resize(tcap, (u64::MAX, 0));
                let mask = tcap - 1;
                let v = &mut st.v;
                let tab = &mut st.tab;
                for i in 0..n {
                    let k = v[i].eid;
                    let mut h = (mix(k) as usize) & mask;
                    loop {
                        let slot = tab[h];
                        if slot.0 == u64::MAX {
                            tab[h] = (k, i as u32);
                            break;
                        }
                        if slot.0 == k {
                            let j = slot.1 as usize;
                            if v[i].ingest < v[j].ingest {
                                v[j].dead = true;
                                tab[h].1 = i as u32;
                            } else {
                                v[i].dead = true;
                            }
                            break;
                        }
                        h = (h + 1) & mask;
                    }
                }
                for e in v.iter() {
                    if !e.dead && e.typ != 255 {
                        st.out[bucket(e.uid)].push(PEv { person: e.uid, ts: e.ts, price: e.price, cat: e.cat, typ: e.typ });
                    }
                }
            },
        );
        (links.join().unwrap(), sb)
    });
    let sb: Vec<Vec<Vec<PEv>>> = sb.into_iter().map(|s| s.out).collect();
    let ck_a: Vec<Vec<Vec<Ck>>> = sa.into_iter().map(|s| s.ck).collect();
    log("dedup+links");

    // ---- phase C: uid -> person, scatter by person bucket ----
    struct SC {
        ev: Vec<Vec<PEv>>,
        ck: Vec<Vec<Ck>>,
    }
    let sc = run(
        nt,
        NB,
        || SC { ev: buckets(cap), ck: buckets(cap / 4) },
        |st, b| {
            let m = &person_maps[b];
            for s in &sb {
                for e in &s[b] {
                    let mut e = *e;
                    if let Some(&q) = m.get(&e.person) {
                        e.person = q;
                    }
                    st.ev[bucket(e.person)].push(e);
                }
            }
            for s in &ck_a {
                for c in &s[b] {
                    let mut c = *c;
                    if let Some(&q) = m.get(&c.person) {
                        c.person = q;
                    }
                    st.ck[bucket(c.person)].push(c);
                }
            }
        },
    );
    drop(sb);
    drop(ck_a);
    log("mapped");

    // ---- phase D: sessions + attribution per person bucket ----
    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;
    struct SD {
        ev: Vec<PEv>,
        ck: Vec<Ck>,
        users: Vec<UserRow>,
        camp: Vec<[i64; 4]>,
    }
    let sd = run(
        nt,
        NB,
        || SD { ev: vec![], ck: vec![], users: vec![], camp: vec![[0; 4]; ncamp * ndays] },
        |st, b| {
            st.ev.clear();
            st.ck.clear();
            for s in &sc {
                st.ev.extend_from_slice(&s.ev[b]);
                st.ck.extend_from_slice(&s.ck[b]);
            }
            st.ev.sort_unstable_by_key(|e| ((e.person as u128) << 32) | e.ts as u128);
            st.ck.sort_unstable_by(|x, y| {
                x.person.cmp(&y.person).then(x.ts.cmp(&y.ts)).then_with(|| cmp_key(x.cid, y.cid))
            });
            process_bucket(&st.ev, &st.ck, &mut st.users, &mut st.camp, ncat, ndays, ws, cws, organic, &cat_rank);
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
}

#[allow(clippy::too_many_arguments)]
fn process_bucket(
    ev: &[PEv],
    ck: &[Ck],
    users: &mut Vec<UserRow>,
    camp_out: &mut [[i64; 4]],
    ncat: usize,
    ndays: usize,
    ws: i64,
    cws: i64,
    organic: u16,
    cat_rank: &[usize],
) {
    let base_day = ws.div_euclid(DAY);
    let mut catsum = vec![0i64; ncat];
    let mut touched: Vec<u16> = vec![];
    let mut s_start: Vec<i64> = vec![];
    let mut s_day: Vec<usize> = vec![];
    let mut s_rev: Vec<i64> = vec![];
    let mut s_pur: Vec<bool> = vec![];
    let mut cbyc: Vec<(u16, i64)> = vec![];
    let mut cts: Vec<i64> = vec![];
    let mut ci = 0usize;
    let mut i = 0usize;
    while i < ev.len() {
        let p = ev[i].person;
        let mut j = i;
        while j < ev.len() && ev[j].person == p {
            j += 1;
        }
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
                if e.typ == 2 {
                    pur = true;
                    rev += e.price as i64;
                    if catsum[e.cat as usize] == 0 {
                        touched.push(e.cat);
                    }
                    catsum[e.cat as usize] += e.price as i64;
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

        while ci < ck.len() && ck[ci].person < p {
            ci += 1;
        }
        let c0 = ci;
        while ci < ck.len() && ck[ci].person == p {
            ci += 1;
        }
        let cl = &ck[c0..ci];
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
                let mut r = q - 1;
                while r > 0 && cts[r - 1] == t {
                    r -= 1;
                }
                camp = cl[r].camp;
            }
            let a = &mut camp_out[camp as usize * ndays + s_day[s]];
            a[0] += 1;
            a[1] += s_pur[s] as i64;
            a[2] += s_rev[s];
        }
        if !cl.is_empty() && !s_start.is_empty() {
            cbyc.clear();
            cbyc.extend(cl.iter().zip(cts.iter()).map(|(c, &t)| (c.camp, t)));
            cbyc.sort_unstable();
            let mut x = 0;
            while x < cbyc.len() {
                let camp = cbyc[x].0;
                let mut lo = cbyc[x].1;
                let mut hi = lo + WEEK;
                x += 1;
                loop {
                    let more = x < cbyc.len() && cbyc[x].0 == camp;
                    if more && cbyc[x].1 <= hi {
                        hi = cbyc[x].1 + WEEK;
                        x += 1;
                        continue;
                    }
                    let a = s_start.partition_point(|&v| v < lo);
                    let z = s_start.partition_point(|&v| v <= hi);
                    let mut s = a;
                    while s < z {
                        let d = s_day[s];
                        let e = s + s_day[s..z].partition_point(|&dd| dd == d);
                        camp_out[camp as usize * ndays + d][3] += (e - s) as i64;
                        s = e;
                    }
                    if more {
                        lo = cbyc[x].1;
                        hi = lo + WEEK;
                        x += 1;
                        continue;
                    }
                    break;
                }
            }
        }
        i = j;
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
