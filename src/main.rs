// Shopping Sessions pipeline. Implements rules.md exactly.
use memchr::memchr;
use memmap2::Mmap;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 256; // buckets

// ---------- string keys ----------
// Strings up to 15 bytes are packed exactly into a u128 so that u128 order equals
// byte-wise order. Longer strings get a 128-bit hash with low byte 0xFF and are
// kept in a side table.
static LONG: LazyLock<Mutex<FxHashMap<u128, Vec<u8>>>> = LazyLock::new(|| Mutex::new(FxHashMap::default()));

#[inline]
fn key(s: &[u8]) -> u128 {
    if s.len() <= 15 {
        let mut b = [0u8; 16];
        b[..s.len()].copy_from_slice(s);
        b[15] = s.len() as u8;
        u128::from_be_bytes(b)
    } else {
        long_key(s)
    }
}

#[cold]
fn long_key(s: &[u8]) -> u128 {
    use std::hash::{Hash, Hasher};
    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h1);
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    0xabcdu32.hash(&mut h2);
    s.hash(&mut h2);
    let k = ((h1.finish() as u128) << 64 | h2.finish() as u128) | 0xFF;
    LONG.lock().unwrap().entry(k).or_insert_with(|| s.to_vec());
    k
}

#[inline]
fn is_long(k: u128) -> bool {
    (k & 0xFF) == 0xFF
}

fn key_bytes(k: u128) -> Vec<u8> {
    if is_long(k) {
        LONG.lock().unwrap().get(&k).unwrap().clone()
    } else {
        let b = k.to_be_bytes();
        b[..b[15] as usize].to_vec()
    }
}

#[inline]
fn cmp_key(a: u128, b: u128) -> Ordering {
    if !is_long(a) && !is_long(b) {
        a.cmp(&b)
    } else {
        key_bytes(a).cmp(&key_bytes(b))
    }
}

#[inline]
fn mix(k: u128) -> u64 {
    let x = (k as u64) ^ ((k >> 64) as u64).rotate_left(31);
    let x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}
#[inline]
fn bucket(k: u128) -> usize {
    (mix(k) >> 56) as usize
}

// ---------- CSV ----------
// Splits one line (without '\n') into exactly N fields. None = malformed.
// Each field: None for null (empty), Some(bytes) otherwise.
#[inline]
fn split<'a, const N: usize>(line: &'a [u8]) -> Option<[&'a [u8]; N]> {
    if line.is_empty() {
        return None;
    }
    let mut out: [&[u8]; N] = [&[]; N];
    let has_quote = memchr(b'"', line).is_some();
    let mut start = 0;
    let mut i = 0;
    for pos in memchr::memchr_iter(b',', line) {
        if i >= N - 1 {
            return None;
        }
        out[i] = &line[start..pos];
        i += 1;
        start = pos + 1;
    }
    if i != N - 1 {
        return None;
    }
    out[i] = &line[start..];
    if has_quote {
        for f in out.iter_mut() {
            if let Some(q) = memchr(b'"', f) {
                if q != 0 || f.len() < 2 || f[f.len() - 1] != b'"' {
                    return None;
                }
                let inner = &f[1..f.len() - 1];
                if memchr(b'"', inner).is_some() {
                    return None;
                }
                *f = inner;
            }
        }
    }
    Some(out)
}

// ---------- timestamps ----------
#[inline]
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

/// Parses a ts. Returns None for a bad format. F3 values too large return Some(i64::MAX).
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.iter().all(|c| c.is_ascii_digit()) {
        let mut i = 0;
        while i < b.len() && b[i] == b'0' {
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
    let n = b.len();
    if n != 20 && n != 25 {
        return None;
    }
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
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

fn map(p: &Path) -> Option<Mmap> {
    let f = File::open(p).ok()?;
    if f.metadata().ok()?.len() == 0 {
        return None;
    }
    unsafe { Mmap::map(&f).ok() }
}

/// Splits mapped files into byte ranges of data lines (header skipped), each
/// range starting at a line start and ending after a '\n' or at file end.
fn chunks<'a>(maps: &'a [Mmap], target: usize) -> Vec<&'a [u8]> {
    let mut out = vec![];
    for m in maps {
        let b: &[u8] = &m[..];
        let Some(h) = memchr(b'\n', b) else { continue };
        let mut s = h + 1;
        while s < b.len() {
            let mut e = (s + target).min(b.len());
            if e < b.len() {
                match memchr(b'\n', &b[e..]) {
                    Some(p) => e = e + p + 1,
                    None => e = b.len(),
                }
            }
            out.push(&b[s..e]);
            s = e;
        }
    }
    out
}

#[inline]
fn for_lines(c: &[u8], mut f: impl FnMut(&[u8])) {
    let mut s = 0;
    while s < c.len() {
        let e = memchr(b'\n', &c[s..]).map(|p| s + p).unwrap_or(c.len());
        f(&c[s..e]);
        s = e + 1;
    }
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

// ---------- fast line parsing ----------
#[inline]
fn skip_line(b: &[u8], i: usize) -> usize {
    match memchr(b'\n', &b[i.min(b.len())..]) {
        Some(p) => i + p + 1,
        None => b.len(),
    }
}

/// Parses the line starting at b[i] into exactly N fields (start,end) with
/// surrounding quotes removed. Returns (index of next line, well-formed).
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

struct Chunk {
    file: usize,
    s: u64,
    e: u64,
}

fn open_all(dir: &Path) -> (Vec<File>, Vec<u64>) {
    let mut fs = vec![];
    let mut sz = vec![];
    for p in list_csv(dir) {
        if let Ok(f) = File::open(&p) {
            let l = f.metadata().map(|m| m.len()).unwrap_or(0);
            if l > 0 {
                fs.push(f);
                sz.push(l);
            }
        }
    }
    (fs, sz)
}

fn make_chunks(sizes: &[u64], target: u64) -> Vec<Chunk> {
    let mut v = vec![];
    for (fi, &l) in sizes.iter().enumerate() {
        let mut s = 0;
        while s < l {
            let e = (s + target).min(l);
            v.push(Chunk { file: fi, s, e });
            s = e;
        }
    }
    // biggest first is irrelevant (equal sizes); keep order
    v
}

/// Reads the lines that start in [s, e) (header excluded) into buf; returns
/// the byte range of those lines inside buf.
fn read_chunk(f: &File, len: u64, c: &Chunk, buf: &mut Vec<u8>) -> (usize, usize) {
    use std::os::unix::fs::FileExt;
    let a = c.s.saturating_sub(1);
    let mut want = (c.e + 512).min(len);
    loop {
        let n = (want - a) as usize;
        buf.resize(n, 0);
        f.read_exact_at(&mut buf[..n], a).unwrap();
        // start: after first '\n' at index >= s-1 (relative 0)
        let Some(p) = memchr(b'\n', &buf[..]) else {
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
        // end: first '\n' at index >= e-1
        let from = (c.e - 1 - a) as usize;
        match memchr(b'\n', &buf[from..]) {
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

fn par_chunks<T: Send>(dir: &Path, target: u64, f: impl Fn(&[u8]) -> T + Sync + Send) -> Vec<T> {
    let (files, sizes) = open_all(dir);
    let ch = make_chunks(&sizes, target);
    ch.par_iter()
        .map_init(Vec::new, |buf, c| {
            let (s, e) = read_chunk(&files[c.file], sizes[c.file], c, buf);
            f(&buf[s..e])
        })
        .collect()
}

// ---------- records ----------
#[derive(Clone, Copy)]
struct Ev {
    eid: u128,
    uid: u128,
    ingest: i64,
    ts: i64,
    price: u32,
    cat: u16,
    typ: u8, // 0 view,1 cart,2 purchase, 255 invalid
}

#[derive(Clone, Copy)]
struct PEv {
    person: u128,
    ts: i64,
    price: u32,
    cat: u16,
    typ: u8,
}

#[derive(Clone, Copy)]
struct Ck {
    person: u128,
    cid: u128,
    ts: i64,
    camp: u16,
}

struct UserRow {
    person: u128,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: Option<u16>,
}

fn gather<T: Copy>(parts: &[Vec<Vec<T>>], b: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(parts.iter().map(|p| p[b].len()).sum());
    for p in parts {
        v.extend_from_slice(&p[b]);
    }
    v
}

fn new_buckets<T>(cap: usize) -> Vec<Vec<T>> {
    (0..NB).map(|_| Vec::with_capacity(cap)).collect()
}

fn main() {
    let t0 = std::time::Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into()));
    let outdir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or("/output".into()));
    let ncpu: usize = std::env::var("NCPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if ncpu > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(ncpu).build_global().ok();
    }

    let manifest = String::from_utf8_lossy(&read_small(&data.join("manifest.json"))).to_string();
    let ws = manifest_time(&manifest, "window_start");
    let we = manifest_time(&manifest, "window_end");
    let cws = ws - WEEK;

    // products -> categories
    let mut cat_names: Vec<Vec<u8>> = vec![];
    let mut cat_idx: FxHashMap<Vec<u8>, u16> = FxHashMap::default();
    let mut products: FxHashMap<u128, u16> = FxHashMap::default();
    {
        let b = read_small(&data.join("products.csv"));
        if let Some(h) = memchr(b'\n', &b) {
            for_lines(&b[h + 1..], |l| {
                if let Some([p, c]) = split::<2>(l) {
                    if p.is_empty() {
                        return;
                    }
                    let n = cat_names.len() as u16;
                    let ci = *cat_idx.entry(c.to_vec()).or_insert_with(|| {
                        cat_names.push(c.to_vec());
                        n
                    });
                    products.insert(key(p), ci);
                }
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
    let mut campaigns: FxHashMap<u128, u16> = FxHashMap::default();
    {
        let b = read_small(&data.join("campaigns.csv"));
        if let Some(h) = memchr(b'\n', &b) {
            for_lines(&b[h + 1..], |l| {
                if let Some([c, ch]) = split::<2>(l) {
                    if c.is_empty() {
                        return;
                    }
                    let k = key(c);
                    if campaigns.contains_key(&k) {
                        return;
                    }
                    campaigns.insert(k, camp_ids.len() as u16);
                    camp_ids.push(c.to_vec());
                    camp_ch.push(ch.to_vec());
                }
            });
        }
    }
    let organic = camp_ids.len() as u16;
    camp_ids.push(b"organic".to_vec());
    camp_ch.push(b"organic".to_vec());
    let ncamp = camp_ids.len();

    // links -> per-bucket maps id -> person (only ids whose person differs)
    let (person_maps, ev_parts, ck_parts) = {
        let (links, (evp, ckp)) = rayon::join(
            || {
                let pairs: Vec<Vec<(u128, u128)>> = par_chunks(&data.join("links"), 1 << 22, |c| {
                    let mut v = vec![];
                    let mut f = [(0usize, 0usize); 2];
                    let mut i = 0;
                    while i < c.len() {
                        let (ni, ok) = next_line::<2>(c, i, &mut f);
                        if ok {
                            let a = &c[f[0].0..f[0].1];
                            let b = &c[f[1].0..f[1].1];
                            if !a.is_empty() && !b.is_empty() {
                                v.push((key(a), key(b)));
                            }
                        }
                        i = ni;
                    }
                    v
                });
                let mut idx: FxHashMap<u128, u32> = FxHashMap::default();
                let mut keys: Vec<u128> = vec![];
                let mut parent: Vec<u32> = vec![];
                fn find(p: &mut [u32], mut x: u32) -> u32 {
                    while p[x as usize] != x {
                        let gp = p[p[x as usize] as usize];
                        p[x as usize] = gp;
                        x = gp;
                    }
                    x
                }
                for v in &pairs {
                    for &(a, b) in v {
                        let mut g = |k: u128| -> u32 {
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
                let mut minkey: Vec<u128> = keys.clone();
                for i in 0..keys.len() {
                    let r = find(&mut parent, i as u32) as usize;
                    if cmp_key(keys[i], minkey[r]) == Ordering::Less {
                        minkey[r] = keys[i];
                    }
                }
                let mut maps: Vec<FxHashMap<u128, u128>> = (0..NB).map(|_| FxHashMap::default()).collect();
                for i in 0..keys.len() {
                    let r = find(&mut parent, i as u32) as usize;
                    if minkey[r] != keys[i] {
                        maps[bucket(keys[i])].insert(keys[i], minkey[r]);
                    }
                }
                maps
            },
            || {
                rayon::join(
                    || {
                        // events: parse -> buckets by event_id
                        par_chunks(&data.join("events"), 1 << 22, |c| {
                            let mut bk: Vec<Vec<Ev>> = new_buckets(c.len() / 70 / NB + 8);
                            let mut f = [(0usize, 0usize); 7];
                            let mut i = 0;
                            while i < c.len() {
                                let (ni, ok) = next_line::<7>(c, i, &mut f);
                                i = ni;
                                if !ok {
                                    continue;
                                }
                                let g = |k: usize| &c[f[k].0..f[k].1];
                                let eid = g(0);
                                if eid.is_empty() {
                                    continue;
                                }
                                let ek = key(eid);
                                let ingest = parse_ts(g(3)).unwrap_or(i64::MAX);
                                let mut e = Ev { eid: ek, uid: 0, ingest, ts: 0, price: 0, cat: 0, typ: 255 };
                                'v: {
                                    let uid = g(1);
                                    let pid = g(5);
                                    if uid.is_empty() || pid.is_empty() {
                                        break 'v;
                                    }
                                    let Some(t) = parse_ts(g(2)) else { break 'v };
                                    if t < ws || t >= we || ingest - t > DAY {
                                        break 'v;
                                    }
                                    let typ = match g(4) {
                                        b"view" => 0,
                                        b"cart" => 1,
                                        b"purchase" => 2,
                                        _ => break 'v,
                                    };
                                    if typ == 2 {
                                        match parse_price(g(6)) {
                                            Some(p) => e.price = p,
                                            None => break 'v,
                                        }
                                    }
                                    e.ts = t;
                                    e.typ = typ;
                                    e.cat = *products.get(&key(pid)).unwrap_or(&unknown_cat);
                                    e.uid = key(uid);
                                }
                                bk[bucket(ek)].push(e);
                            }
                            bk
                        })
                    },
                    || {
                        par_chunks(&data.join("clicks"), 1 << 22, |c| {
                            let mut bk: Vec<Vec<Ck>> = new_buckets(c.len() / 50 / NB + 8);
                            let mut f = [(0usize, 0usize); 4];
                            let mut i = 0;
                            while i < c.len() {
                                let (ni, ok) = next_line::<4>(c, i, &mut f);
                                i = ni;
                                if !ok {
                                    continue;
                                }
                                let g = |k: usize| &c[f[k].0..f[k].1];
                                let (cid, uid) = (g(0), g(1));
                                if cid.is_empty() || uid.is_empty() {
                                    continue;
                                }
                                let Some(t) = parse_ts(g(3)) else { continue };
                                if t < cws || t >= we {
                                    continue;
                                }
                                let Some(&ci) = campaigns.get(&key(g(2))) else { continue };
                                let u = key(uid);
                                bk[bucket(u)].push(Ck { person: u, cid: key(cid), ts: t, camp: ci });
                            }
                            bk
                        })
                    },
                )
            },
        );
        (links, evp, ckp)
    };
    eprintln!("parsed {:?}", t0.elapsed());

    // dedup per eid bucket -> scatter by uid bucket
    let ue_parts: Vec<Vec<Vec<PEv>>> = (0..NB)
        .into_par_iter()
        .map(|b| {
            let v = gather(&ev_parts, b);
            let mut best: FxHashMap<u128, u32> = FxHashMap::with_capacity_and_hasher(v.len(), Default::default());
            for (i, e) in v.iter().enumerate() {
                best.entry(e.eid)
                    .and_modify(|j| {
                        if e.ingest < v[*j as usize].ingest {
                            *j = i as u32;
                        }
                    })
                    .or_insert(i as u32);
            }
            let mut out: Vec<Vec<PEv>> = new_buckets(v.len() / NB + 8);
            for &j in best.values() {
                let e = &v[j as usize];
                if e.typ != 255 {
                    out[bucket(e.uid)].push(PEv { person: e.uid, ts: e.ts, price: e.price, cat: e.cat, typ: e.typ });
                }
            }
            out
        })
        .collect();
    drop(ev_parts);
    eprintln!("dedup {:?}", t0.elapsed());

    // map uid -> person within uid bucket, scatter by person bucket
    let mapped: Vec<(Vec<Vec<PEv>>, Vec<Vec<Ck>>)> = (0..NB)
        .into_par_iter()
        .map(|b| {
            let m = &person_maps[b];
            let mut eo: Vec<Vec<PEv>> = new_buckets(0);
            let mut co: Vec<Vec<Ck>> = new_buckets(0);
            for p in &ue_parts {
                for e in &p[b] {
                    let mut e = *e;
                    if let Some(&q) = m.get(&e.person) {
                        e.person = q;
                    }
                    eo[bucket(e.person)].push(e);
                }
            }
            for p in &ck_parts {
                for c in &p[b] {
                    let mut c = *c;
                    if let Some(&q) = m.get(&c.person) {
                        c.person = q;
                    }
                    co[bucket(c.person)].push(c);
                }
            }
            (eo, co)
        })
        .collect();
    drop(ue_parts);
    drop(ck_parts);
    eprintln!("mapped {:?}", t0.elapsed());

    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;

    struct Out {
        users: Vec<UserRow>,
        camp: Vec<[i64; 4]>,
    }
    let results: Vec<Out> = (0..NB)
        .into_par_iter()
        .map(|b| {
            let mut ev: Vec<PEv> = Vec::with_capacity(mapped.iter().map(|p| p.0[b].len()).sum());
            for p in &mapped {
                ev.extend_from_slice(&p.0[b]);
            }
            ev.sort_unstable_by(|x, y| x.person.cmp(&y.person).then(x.ts.cmp(&y.ts)));
            let mut ck: Vec<Ck> = Vec::new();
            for p in &mapped {
                ck.extend_from_slice(&p.1[b]);
            }
            ck.sort_unstable_by(|x, y| x.person.cmp(&y.person).then(x.ts.cmp(&y.ts)).then_with(|| cmp_key(x.cid, y.cid)));
            let mut out = Out { users: vec![], camp: vec![[0; 4]; ncamp * ndays] };
            process_bucket(&ev, &ck, &mut out.users, &mut out.camp, ncat, ndays, base_day, organic, &cat_rank);
            out
        })
        .collect();
    eprintln!("sessions {:?}", t0.elapsed());

    let mut camp = vec![[0i64; 4]; ncamp * ndays];
    for r in &results {
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
    let ng = rayon::current_num_threads().max(1) * 2;
    let groups: Vec<Vec<usize>> = (0..ng).map(|g| (0..NB).filter(|b| b % ng == g).collect()).collect();
    groups.par_iter().enumerate().for_each(|(g, bs)| {
        let rows: Vec<&UserRow> = bs.iter().flat_map(|&b| results[b].users.iter()).collect();
        if !rows.is_empty() || g == 0 {
            write_user(&ud.join(format!("part-{g:03}.parquet")), &rows, &cat_names);
        }
    });
    {
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
    }
    eprintln!("done {:?}", t0.elapsed());
}

#[allow(clippy::too_many_arguments)]
fn process_bucket(
    ev: &[PEv],
    ck: &[Ck],
    users: &mut Vec<UserRow>,
    camp_out: &mut [[i64; 4]],
    ncat: usize,
    ndays: usize,
    base_day: i64,
    organic: u16,
    cat_rank: &[usize],
) {
    let mut catsum = vec![0i64; ncat];
    let mut touched: Vec<u16> = vec![];
    let mut s_start: Vec<i64> = vec![];
    let mut s_day: Vec<usize> = vec![];
    let mut s_rev: Vec<i64> = vec![];
    let mut s_pur: Vec<bool> = vec![];
    let mut cbyc: Vec<(u16, i64)> = vec![];
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
            let start = ev[k].ts;
            let mut last = start;
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
            while k < j && ev[k].ts - last <= GAP {
                let e = ev[k];
                last = e.ts;
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
        let mut q = 0usize;
        for s in 0..s_start.len() {
            let st = s_start[s];
            while q < cl.len() && cl[q].ts <= st {
                q += 1;
            }
            let mut camp = organic;
            if q > 0 && st - cl[q - 1].ts <= WEEK {
                let t = cl[q - 1].ts;
                let mut r = q - 1;
                while r > 0 && cl[r - 1].ts == t {
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
            cbyc.extend(cl.iter().map(|c| (c.camp, c.ts)));
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

fn write_user(path: &Path, rows: &[&UserRow], cat_names: &[Vec<u8>]) {
    let schema = Arc::new(
        parse_message_type(
            "message user_daily { required binary user_id (UTF8); required int32 day (DATE); required int64 sessions; required int64 events; required int64 revenue_cents; optional binary top_category (UTF8); }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
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
                let v: Vec<ByteArray> =
                    rows.iter().filter_map(|r| r.top.map(|t| ByteArray::from(cat_names[t as usize].clone()))).collect();
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
