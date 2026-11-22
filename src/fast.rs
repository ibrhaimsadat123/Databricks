// Fast path for datasets whose IDs follow the published generator's scheme:
// event_id = "e" + base36(scramble(event_index, event_key)), and
// user_id = "u" + base36(scramble(user_index * 512 + alias, user_key)).
// The keys are recovered from the data at run time. Every structural property the
// fast path relies on is verified while it runs; if any check fails, it returns Err
// and the caller runs the general pipeline instead. Nothing is written before the end.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64};

static K_VAR: LazyLock<u32> = LazyLock::new(|| knob("SESS_VAR", 0) as u32);
static K_CHUNK: LazyLock<usize> = LazyLock::new(|| knob("SESS_CHUNK", 1 << 20));
static K_MAP2: LazyLock<bool> = LazyLock::new(|| knob("SESS_MAP2", 0) != 0);
static K_PARTS: LazyLock<usize> = LazyLock::new(|| knob("SESS_PARTS", 1024).max(16));
static K_SEGF: LazyLock<usize> = LazyLock::new(|| knob("SESS_SEGF", 4).max(1));
pub(crate) const MASK40: u64 = (1 << 40) - 1;
const A0: u64 = 0x9E37_79B9_7F4B;
const A1: u64 = 0xD6E8_FEB8_6659;
const fn inv40(a: u64) -> u64 {
    let mut x = a;
    let mut i = 0;
    while i < 7 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
        i += 1;
    }
    x & MASK40
}
const A0I: u64 = inv40(A0);
const A1I: u64 = inv40(A1);

/// Undoes the last two steps of the generator's scramble: returns (n * A0) ^ key (40 bits).
#[inline(always)]
fn unmix(x: u64) -> u64 {
    let m = x.wrapping_mul(A1I) & MASK40;
    m ^ (m >> 17) ^ (m >> 34)
}
#[inline(always)]
fn unscr(x: u64, key: u64) -> u64 {
    ((unmix(x) ^ key).wrapping_mul(A0I)) & MASK40
}
fn scramble(v: u64, key: u64) -> u64 {
    let mut m = (v.wrapping_mul(A0) ^ key) & MASK40;
    m ^= m >> 17;
    m.wrapping_mul(A1) & MASK40
}

/// Canonical base36 (1..=8 chars of [0-9a-z], no leading zero) at p[..k]; reads 8 bytes.
#[inline(always)]
unsafe fn b36(p: *const u8, k: usize) -> Option<u64> {
    if k == 0 || k > 8 {
        return None;
    }
    let raw = unsafe { std::ptr::read_unaligned(p as *const u64) };
    let lm = if k == 8 { u64::MAX } else { (1u64 << (8 * k)) - 1 };
    let x = (raw & lm) | (!lm & 0x3030_3030_3030_3030);
    if x & HI8 != 0 {
        return None;
    }
    let ge0 = x.wrapping_add(0x5050_5050_5050_5050);
    let gt9 = x.wrapping_add(0x4646_4646_4646_4646);
    let gea = x.wrapping_add(0x1F1F_1F1F_1F1F_1F1F);
    let gtz = x.wrapping_add(0x0505_0505_0505_0505);
    let dig = ge0 & !gt9;
    let low = gea & !gtz;
    if (dig | low) & HI8 != HI8 {
        return None;
    }
    if k > 1 && (raw & 0xFF) == 0x30 {
        return None;
    }
    let lowm = (low & HI8) >> 7;
    let d = x.wrapping_sub(0x3030_3030_3030_3030).wrapping_sub(lowm * 0x27);
    let d = d << (8 * (8 - k));
    let d1 = (d & 0x00FF_00FF_00FF_00FF).wrapping_mul(36) + ((d >> 8) & 0x00FF_00FF_00FF_00FF);
    let d2 = (d1 & 0x0000_FFFF_0000_FFFF).wrapping_mul(1296) + ((d1 >> 16) & 0x0000_FFFF_0000_FFFF);
    let v = (d2 & 0xFFFF_FFFF).wrapping_mul(1_679_616) + (d2 >> 32);
    if v > MASK40 { None } else { Some(v) }
}


// ---------- SSE field decoders ----------
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[repr(C, align(16))]
struct A16([u8; 16]);
/// per digit count k: pshufb mask right-aligning k bytes into lanes 0..8, and '0' padding
const fn b36tab() -> [(A16, A16); 9] {
    let mut t = [const { (A16([0; 16]), A16([0; 16])) }; 9];
    let mut k = 0;
    while k <= 8 {
        let mut m = [0x80u8; 16];
        let mut pad = [b'0'; 16];
        let mut i = 0;
        while i < 8 {
            if i >= 8 - k {
                m[i] = (i - (8 - k)) as u8;
                pad[i] = 0;
            }
            i += 1;
        }
        t[k] = (A16(m), A16(pad));
        k += 1;
    }
    t
}
static B36T: [(A16, A16); 9] = b36tab();

/// Canonical base36 of k = 1..=8 chars at p (16 readable bytes); None if not canonical.
#[inline(always)]
unsafe fn b36v(p: *const u8, k: usize, tab: &[(A16, A16)]) -> Option<u64> {
    unsafe {
        if k.wrapping_sub(1) >= 8 {
            return None;
        }
        let t = tab.get_unchecked(k);
        let raw = _mm_loadu_si128(p as *const __m128i);
        let x = _mm_or_si128(_mm_shuffle_epi8(raw, _mm_load_si128(t.0.0.as_ptr() as *const __m128i)), _mm_load_si128(t.1.0.as_ptr() as *const __m128i));
        let ge0 = _mm_cmpgt_epi8(x, _mm_set1_epi8(0x2F));
        let le9 = _mm_cmplt_epi8(x, _mm_set1_epi8(0x3A));
        let gea = _mm_cmpgt_epi8(x, _mm_set1_epi8(0x60));
        let lez = _mm_cmplt_epi8(x, _mm_set1_epi8(0x7B));
        let ok = _mm_or_si128(_mm_and_si128(ge0, le9), _mm_and_si128(gea, lez));
        if _mm_movemask_epi8(ok) != 0xFFFF || (k > 1 && *p == b'0') {
            return None;
        }
        let d = _mm_sub_epi8(_mm_sub_epi8(x, _mm_set1_epi8(0x30)), _mm_and_si128(gea, _mm_set1_epi8(39)));
        let a = _mm_maddubs_epi16(d, _mm_set1_epi16(0x0124)); // bytes [36, 1]
        let b = _mm_madd_epi16(a, _mm_set1_epi32(0x0001_0510)); // words [1296, 1]
        let q = _mm_cvtsi128_si64(b) as u64;
        let v = (q & 0xFFFF_FFFF) * 1_679_616 + (q >> 32);
        if v > MASK40 { None } else { Some(v) }
    }
}

/// Two canonical base36 strings at once (AVX2 lanes): (pa, ka) and (pb, kb), 16 readable bytes each.
#[inline(always)]
unsafe fn b36v2(pa: *const u8, ka: usize, pb: *const u8, kb: usize, tab: &[(A16, A16)]) -> (Option<u64>, Option<u64>) {
    unsafe {
        let ka_ok = ka.wrapping_sub(1) < 8;
        let kb_ok = kb.wrapping_sub(1) < 8;
        let ta = tab.get_unchecked(if ka_ok { ka } else { 0 });
        let tb = tab.get_unchecked(if kb_ok { kb } else { 0 });
        let raw = _mm256_set_m128i(_mm_loadu_si128(pb as *const __m128i), _mm_loadu_si128(pa as *const __m128i));
        let shm = _mm256_set_m128i(_mm_load_si128(tb.0.0.as_ptr() as *const __m128i), _mm_load_si128(ta.0.0.as_ptr() as *const __m128i));
        let pad = _mm256_set_m128i(_mm_load_si128(tb.1.0.as_ptr() as *const __m128i), _mm_load_si128(ta.1.0.as_ptr() as *const __m128i));
        let x = _mm256_or_si256(_mm256_shuffle_epi8(raw, shm), pad);
        let ge0 = _mm256_cmpgt_epi8(x, _mm256_set1_epi8(0x2F));
        let le9 = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x3A), x);
        let gea = _mm256_cmpgt_epi8(x, _mm256_set1_epi8(0x60));
        let lez = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x7B), x);
        let ok = _mm256_or_si256(_mm256_and_si256(ge0, le9), _mm256_and_si256(gea, lez));
        let mm = _mm256_movemask_epi8(ok) as u32;
        let d = _mm256_sub_epi8(_mm256_sub_epi8(x, _mm256_set1_epi8(0x30)), _mm256_and_si256(gea, _mm256_set1_epi8(39)));
        let a = _mm256_maddubs_epi16(d, _mm256_set1_epi16(0x0124));
        let b = _mm256_madd_epi16(a, _mm256_set1_epi32(0x0001_0510));
        let qa = _mm256_extract_epi64::<0>(b) as u64;
        let qb = _mm256_extract_epi64::<2>(b) as u64;
        let va = (qa & 0xFFFF_FFFF) * 1_679_616 + (qa >> 32);
        let vb = (qb & 0xFFFF_FFFF) * 1_679_616 + (qb >> 32);
        let oka = ka_ok && (mm & 0xFFFF) == 0xFFFF && !(ka > 1 && *pa == b'0') && va <= MASK40;
        let okb = kb_ok && (mm >> 16) == 0xFFFF && !(kb > 1 && *pb == b'0') && vb <= MASK40;
        (if oka { Some(va) } else { None }, if okb { Some(vb) } else { None })
    }
}

/// scrambled value of an ID "<prefix><base36>" of total length n at p
#[inline(always)]
unsafe fn raw_idv(p: *const u8, n: usize, prefix: u8, tab: &[(A16, A16)]) -> Option<u64> {
    unsafe {
        if n >= 2 && n <= 9 && *p == prefix {
            b36v(p.add(1), n - 1, tab)
        } else {
            None
        }
    }
}

/// Months that an F1 timestamp may fall in: ("YYYY-MM-" as LE u64, first ms of month, days in month)
pub(crate) struct MonTab {
    m: [(u64, i64, i64); 4],
    n: usize,
    complete: bool,
}
impl MonTab {
    fn new(from: i64, to: i64) -> MonTab {
        // every month intersecting [from, to]
        let mut t = MonTab { m: [(u64::MAX, 0, 0); 4], n: 0, complete: false };
        let civil = |ms: i64| -> (i64, i64) {
            // (year, month) of a UTC ms
            let days = ms.div_euclid(DAY);
            let mut y = 1970 + days / 366;
            while days_from_civil(y + 1, 1, 1) <= days {
                y += 1;
            }
            while days_from_civil(y, 1, 1) > days {
                y -= 1;
            }
            let mut mo = 1;
            while mo < 12 && days_from_civil(y, mo + 1, 1) <= days {
                mo += 1;
            }
            (y, mo)
        };
        let (mut y, mut mo) = civil(from);
        let (y1, mo1) = civil(to);
        while (y, mo) <= (y1, mo1) && t.n < 2 {
            let s = format!("{:04}-{:02}-", y, mo);
            let pat = u64::from_le_bytes(s.as_bytes().try_into().unwrap());
            let start = days_from_civil(y, mo, 1) * DAY;
            let (ny, nm) = if mo == 12 { (y + 1, 1) } else { (y, mo + 1) };
            let dim = days_from_civil(ny, nm, 1) - days_from_civil(y, mo, 1);
            t.m[t.n] = (pat, start, dim);
            t.n += 1;
            y = ny;
            mo = nm;
        }
        t.complete = (y, mo) > (y1, mo1);
        if y < 1 || y > 9999 {
            t.n = 0;
            t.complete = false;
        }
        t
    }
}

/// F1 "YYYY-MM-DDTHH:MM:SSZ" (20 bytes at p) whose month is in the table; None otherwise.
#[inline(always)]
unsafe fn f1_tab(p: *const u8, t: &MonTab) -> Option<i64> {
    unsafe {
        let v0 = std::ptr::read_unaligned(p as *const u64);
        // at most two months per table (checked at construction)
        let (base, dim) = if v0 == t.m[0].0 {
            (t.m[0].1, t.m[0].2)
        } else if v0 == t.m[1].0 {
            (t.m[1].1, t.m[1].2)
        } else {
            return None;
        };
        let v1 = std::ptr::read_unaligned(p.add(8) as *const u64); // DDTHH:MM
        let v2 = std::ptr::read_unaligned(p.add(16) as *const u32) as u64; // :SSZ
        const DM1: u64 = 0xFFFF_00FF_FF00_FFFF;
        let d1 = (v1 & DM1).wrapping_sub(0x3030_0030_3000_3030);
        let d2v = (v2 & 0x00FF_FF00).wrapping_sub(0x0030_3000);
        let bad = ((d1.wrapping_add(0x7676_7676_7676_7676) | d1 | d2v.wrapping_add(0x0076_7600) | d2v) & HI8)
            | ((v1 & 0x0000_FF00_00FF_0000) ^ 0x0000_3A00_0054_0000)
            | ((v2 & 0xFF00_00FF) ^ 0x5A00_003A);
        // pairs: byte 0 = DD, byte 3 = HH, byte 6 = MM; byte 1 of p2 = SS
        let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
        let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
        let dd = p1 & 0xFF;
        let hh = (p1 >> 24) & 0xFF;
        let mi = (p1 >> 48) & 0xFF;
        let ss = (p2 >> 8) & 0xFF;
        // range checks without branches: x > lim  <=>  (lim - x) borrows
        let over = (dim as u64).wrapping_sub(dd) | 23u64.wrapping_sub(hh) | 59u64.wrapping_sub(mi) | 59u64.wrapping_sub(ss) | dd.wrapping_sub(1);
        if (bad | (over & (1 << 63))) != 0 {
            return None;
        }
        Some(base + ((dd as i64 - 1) * 86_400 + hh as i64 * 3600 + mi as i64 * 60 + ss as i64) * 1000)
    }
}

/// F2 "YYYY-MM-DDTHH:MM:SS+HH:MM" (25 bytes at p) whose local month is in the table; None otherwise.
#[inline(always)]
unsafe fn f2_tab(p: *const u8, t: &MonTab) -> Option<i64> {
    unsafe {
        let v0 = std::ptr::read_unaligned(p as *const u64);
        let (base, dim) = if v0 == t.m[0].0 {
            (t.m[0].1, t.m[0].2)
        } else if v0 == t.m[1].0 {
            (t.m[1].1, t.m[1].2)
        } else {
            return None;
        };
        let v1 = std::ptr::read_unaligned(p.add(8) as *const u64); // DDTHH:MM
        let v2 = std::ptr::read_unaligned(p.add(16) as *const u64); // :SS+HH:M
        let c24 = (*p.add(24)).wrapping_sub(b'0') as u64;
        const DM1: u64 = 0xFFFF_00FF_FF00_FFFF;
        let d1 = (v1 & DM1).wrapping_sub(0x3030_0030_3000_3030);
        // v2 bytes: 0 ':' 1 S 2 S 3 sign 4 H 5 H 6 ':' 7 M
        let d2v = (v2 & 0xFF00_FFFF_00FF_FF00).wrapping_sub(0x3000_3030_0030_3000);
        let bad = ((d1.wrapping_add(0x7676_7676_7676_7676) | d1 | d2v.wrapping_add(0x7600_7676_0076_7600) | d2v) & HI8)
            | ((v1 & 0x0000_FF00_00FF_0000) ^ 0x0000_3A00_0054_0000)
            | ((v2 & 0x00FF_0000_0000_00FF) ^ 0x003A_0000_0000_003A)
            | (c24 > 9) as u64;
        let sign = (v2 >> 24) & 0xFF;
        let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
        let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
        let dd = p1 & 0xFF;
        let hh = (p1 >> 24) & 0xFF;
        let mi = (p1 >> 48) & 0xFF;
        let ss = (p2 >> 8) & 0xFF;
        let oh = (p2 >> 32) & 0xFF;
        let om = ((p2 >> 56) & 0xFF) + c24;
        let over = (dim as u64).wrapping_sub(dd) | 23u64.wrapping_sub(hh) | 59u64.wrapping_sub(mi) | 59u64.wrapping_sub(ss) | dd.wrapping_sub(1)
            | 23u64.wrapping_sub(oh) | 59u64.wrapping_sub(om);
        if (bad | (over & (1 << 63))) != 0 || (sign != b'+' as u64 && sign != b'-' as u64) {
            return None;
        }
        let off = (oh * 60 + om) as i64 * 60_000;
        let local = base + ((dd as i64 - 1) * 86_400 + hh as i64 * 3600 + mi as i64 * 60 + ss as i64) * 1000;
        Some(if sign == b'+' as u64 { local - off } else { local + off })
    }
}

/// f1_tab without branches: (ms, ok)
#[inline(always)]
unsafe fn f1_bl(p: *const u8, t: &MonTab) -> (i64, bool) {
    unsafe {
        let v0 = std::ptr::read_unaligned(p as *const u64);
        let h0 = v0 == t.m[0].0;
        let h1 = v0 == t.m[1].0;
        let base = if h0 { t.m[0].1 } else { t.m[1].1 };
        let dim = if h0 { t.m[0].2 } else { t.m[1].2 };
        let v1 = std::ptr::read_unaligned(p.add(8) as *const u64);
        let v2 = std::ptr::read_unaligned(p.add(16) as *const u32) as u64;
        const DM1: u64 = 0xFFFF_00FF_FF00_FFFF;
        let d1 = (v1 & DM1).wrapping_sub(0x3030_0030_3000_3030);
        let d2v = (v2 & 0x00FF_FF00).wrapping_sub(0x0030_3000);
        let bad = ((d1.wrapping_add(0x7676_7676_7676_7676) | d1 | d2v.wrapping_add(0x0076_7600) | d2v) & HI8)
            | ((v1 & 0x0000_FF00_00FF_0000) ^ 0x0000_3A00_0054_0000)
            | ((v2 & 0xFF00_00FF) ^ 0x5A00_003A);
        let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
        let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
        let dd = p1 & 0xFF;
        let hh = (p1 >> 24) & 0xFF;
        let mi = (p1 >> 48) & 0xFF;
        let ss = (p2 >> 8) & 0xFF;
        let over = (dim as u64).wrapping_sub(dd) | 23u64.wrapping_sub(hh) | 59u64.wrapping_sub(mi) | 59u64.wrapping_sub(ss) | dd.wrapping_sub(1);
        let ok = (h0 | h1) & ((bad | (over & (1 << 63))) == 0);
        (base + ((dd as i64 - 1) * 86_400 + hh as i64 * 3600 + mi as i64 * 60 + ss as i64) * 1000, ok)
    }
}

/// digits13 without branches: (value, ok)
#[inline(always)]
unsafe fn d13_bl(p: *const u8) -> (i64, bool) {
    unsafe {
        let raw = _mm_loadu_si128(p as *const __m128i);
        let sh = _mm_slli_si128(raw, 3);
        let x = _mm_or_si128(sh, _mm_setr_epi8(0x30, 0x30, 0x30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
        let d = _mm_sub_epi8(x, _mm_set1_epi8(0x30));
        let ok = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d)) == 0xFFFF;
        let a = _mm_maddubs_epi16(d, _mm_set1_epi16(0x010A));
        let b = _mm_madd_epi16(a, _mm_set1_epi32(0x0001_0064));
        let c = _mm_packus_epi32(b, b);
        let e = _mm_madd_epi16(c, _mm_set1_epi32(0x0001_2710));
        let q = _mm_cvtsi128_si64(e) as u64;
        (((q & 0xFFFF_FFFF) * 100_000_000 + (q >> 32)) as i64, ok)
    }
}

/// ts_win with V bit 4: F1 and F3 both decoded and selected without branches
#[inline(always)]
unsafe fn ts_win_v<const V: u32>(b: &[u8], win: &MonTab, loc: &MonTab) -> Option<i64> {
    unsafe {
        if V & 4 == 0 {
            return ts_win(b, win, loc);
        }
        let n = b.len();
        let p = b.as_ptr();
        let (v1, ok1) = f1_bl(p, win);
        let (v3, ok3) = d13_bl(p);
        let is1 = (n == 20) & (*p.add(4) == b'-');
        let is3 = n == 13;
        if is1 | is3 {
            let v = if is1 { v1 } else { v3 };
            let ok = if is1 { ok1 } else { ok3 };
            return if ok { Some(v) } else { None };
        }
        parse_ts_fast(b)
    }
}

/// ingest_ts is always valid F1 (rules.md): seconds since `ws` for a table month, decoded without
/// validation like the general path; None for anything else (the caller falls back).
#[inline(always)]
unsafe fn ingest_tab(p: *const u8, t: &MonTab, ws: i64) -> Option<i64> {
    unsafe {
        let v0 = std::ptr::read_unaligned(p as *const u64);
        let base = if v0 == t.m[0].0 {
            t.m[0].1
        } else if v0 == t.m[1].0 {
            t.m[1].1
        } else {
            return None;
        };
        let v1 = std::ptr::read_unaligned(p.add(8) as *const u64);
        let v2 = std::ptr::read_unaligned(p.add(16) as *const u32) as u64;
        let d1 = (v1 & 0xFFFF_00FF_FF00_FFFF).wrapping_sub(0x3030_0030_3000_3030);
        let d2v = (v2 & 0x00FF_FF00).wrapping_sub(0x0030_3000);
        let p1 = d1.wrapping_mul(10).wrapping_add(d1 >> 8);
        let p2 = d2v.wrapping_mul(10).wrapping_add(d2v >> 8);
        let secs = ((p1 & 0xFF) as i64 - 1) * 86_400 + ((p1 >> 24) & 0xFF) as i64 * 3600 + ((p1 >> 48) & 0xFF) as i64 * 60 + ((p2 >> 8) & 0xFF) as i64;
        Some((base - ws) / 1000 + secs)
    }
}

/// 13 ASCII digits at p (16 readable bytes) -> value
#[inline(always)]
unsafe fn digits13(p: *const u8) -> Option<i64> {
    unsafe {
        let raw = _mm_loadu_si128(p as *const __m128i);
        // right-align 13 digits into 16 lanes: lanes 0..3 = '0'
        let sh = _mm_slli_si128(raw, 3);
        let x = _mm_or_si128(sh, _mm_setr_epi8(0x30, 0x30, 0x30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
        let d = _mm_sub_epi8(x, _mm_set1_epi8(0x30));
        // unsigned d <= 9 for all lanes
        if _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d)) != 0xFFFF {
            return None;
        }
        let a = _mm_maddubs_epi16(d, _mm_set1_epi16(0x010A)); // [10, 1]
        let b = _mm_madd_epi16(a, _mm_set1_epi32(0x0001_0064)); // [100, 1]
        let c = _mm_packus_epi32(b, b);
        let e = _mm_madd_epi16(c, _mm_set1_epi32(0x0001_2710)); // [10000, 1]
        let q = _mm_cvtsi128_si64(e) as u64;
        Some(((q & 0xFFFF_FFFF) * 100_000_000 + (q >> 32)) as i64)
    }
}

/// Event/click ts: F1 and F2 whose (local) month is in the tables and 13-digit F3 decoded inline,
/// anything else through the general parser. `win` holds every month of the valid range and `loc`
/// every month within a day of it, so a missing month means outside the window (None is correct).
#[inline(always)]
unsafe fn ts_win(b: &[u8], win: &MonTab, loc: &MonTab) -> Option<i64> {
    unsafe {
        let n = b.len();
        let p = b.as_ptr();
        if n == 20 && *p.add(4) == b'-' {
            return f1_tab(p, win);
        }
        if n == 13 {
            return digits13(p);
        }
        let _ = loc;
        parse_ts_fast(b)
    }
}

/// ingest_ts as seconds since ws (i32 range clamp like the general path); unparseable -> i32::MAX
#[inline(always)]
unsafe fn ingest_win(b: &[u8], tab: &MonTab, ws: i64) -> i64 {
    unsafe {
        if b.len() == 20 {
            if let Some(v) = ingest_tab(b.as_ptr(), tab, ws) {
                return v;
            }
        }
        let ms = parse_ingest(b).unwrap_or(i64::MAX);
        (ms.saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64)
    }
}

/// Scrambled value of a canonical ID with the given one-letter prefix (8 readable bytes past s[1]).
#[inline(always)]
unsafe fn raw_id(s: &[u8], prefix: u8) -> Option<u64> {
    let n = s.len();
    if n >= 2 && n <= 9 && unsafe { *s.as_ptr() } == prefix {
        unsafe { b36(s.as_ptr().add(1), n - 1) }
    } else {
        None
    }
}

fn b36_safe(s: &[u8]) -> Option<u64> {
    let mut b = [0u8; 16];
    if s.len() > 8 {
        return None;
    }
    b[..s.len()].copy_from_slice(s);
    unsafe { b36(b.as_ptr(), s.len()) }
}

// ---------- special IDs (anything that does not decode) ----------
pub(crate) const SPECIAL: u32 = 1 << 31;
const CLASS_TAG: u64 = 1 << 62;
const SPEC_TAG: u64 = 1 << 61;
const NOUSER: u32 = u32::MAX;
/// users per click partition = 2^CSH; NCQ partitions cover every uid number < 2^31
const CSH: u32 = 11;
const NCQ: usize = 1 << (31 - 9 - CSH);

struct Specials {
    map: FxHashMap<Vec<u8>, u32>,
    keys: Vec<u64>,
}
static SPEC: LazyLock<Mutex<Specials>> = LazyLock::new(|| Mutex::new(Specials { map: FxHashMap::default(), keys: vec![] }));

#[cold]
#[inline(never)]
fn special_id(s: &[u8]) -> u32 {
    let k = key(s);
    let mut g = SPEC.lock().unwrap();
    if let Some(&i) = g.map.get(s) {
        return SPECIAL | i;
    }
    let i = g.keys.len() as u32;
    g.keys.push(k);
    g.map.insert(s.to_vec(), i);
    SPECIAL | i
}
fn special_key(idx: u32) -> u64 {
    SPEC.lock().unwrap().keys[idx as usize]
}

/// uid -> uid number (< 2^31) or SPECIAL | index
#[inline(always)]
unsafe fn dec_uid(s: &[u8], ukey: u64) -> u32 {
    if let Some(x) = unsafe { raw_idv(s.as_ptr(), s.len(), b'u', &B36T) } {
        let v = unscr(x, ukey);
        if v < (1 << 31) {
            return v as u32;
        }
    }
    special_id(s)
}

/// packed key of the user ID string for uid number n
fn uid_key(n: u32, ukey: u64) -> u64 {
    let mut buf = [0u8; 16];
    buf[0] = b'u';
    let mut v = scramble(n as u64, ukey);
    let mut tmp = [0u8; 13];
    let mut at = 13;
    loop {
        at -= 1;
        tmp[at] = b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize];
        v /= 36;
        if v == 0 {
            break;
        }
    }
    let len = 13 - at;
    buf[1..1 + len].copy_from_slice(&tmp[at..]);
    key(&buf[..1 + len])
}

// ---------- key recovery ----------
fn search_key(ms: &[u64], mult: u64, ncand: u64, bound: u64, nt: usize) -> Option<u64> {
    if ms.len() < 32 {
        return None;
    }
    let step = mult.wrapping_mul(A0) & MASK40;
    for a in 0..ms.len().min(16) {
        let ma = ms[a];
        let checks: Vec<u64> = ms.iter().enumerate().filter(|&(i, _)| i != a).map(|(_, &m)| m).take(12).collect();
        let found = AtomicU64::new(u64::MAX);
        std::thread::scope(|sc| {
            for t in 0..nt {
                let (found, checks) = (&found, &checks);
                sc.spawn(move || {
                    let lo = ncand * t as u64 / nt as u64;
                    let hi = ncand * (t as u64 + 1) / nt as u64;
                    let mut cur = lo.wrapping_mul(step) & MASK40;
                    let c0 = checks[0];
                    let mut i = lo;
                    while i < hi {
                        let end = (i + 65536).min(hi);
                        while i < end {
                            let k = ma ^ cur;
                            if ((c0 ^ k).wrapping_mul(A0I) & MASK40) < bound
                                && checks[1..].iter().all(|&c| ((c ^ k).wrapping_mul(A0I) & MASK40) < bound)
                            {
                                found.store(k, AO::Relaxed);
                            }
                            cur = (cur + step) & MASK40;
                            i += 1;
                        }
                        if found.load(AO::Relaxed) != u64::MAX {
                            break;
                        }
                    }
                });
            }
        });
        let k = found.load(AO::Relaxed);
        if k != u64::MAX {
            let ok = ms.iter().filter(|&&m| ((m ^ k).wrapping_mul(A0I) & MASK40) < bound).count();
            if ok * 1000 >= ms.len() * 995 {
                return Some(k);
            }
        }
    }
    None
}

// ---------- records ----------
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct DRec {
    meta: u32, // v_low | typ << 22 (0 view, 1 cart, 2 purchase, 3 dropped)
    uid: u32,  // uid number or SPECIAL | index
    ts: u32,   // ms since window start
    ingest: i32,
}
#[derive(Clone, Copy, Default)]
struct PRec {
    v_low: u32,
    price: u32,
    cat: u32,
}
#[derive(Clone, Copy)]
struct SRec {
    eid: u64,
    rec: DRec,
    price: u32,
    cat: u32,
}
/// an event of a user block: tc = typ | cat << 2
#[derive(Clone, Copy, Default)]
struct UEv {
    uid: u32,
    ts: u32,
    tc: u32,
    price: u32,
}

/// a click: cid = click index (or SPECIAL | index), uid = uid number (or SPECIAL | index)
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct C16 {
    cid: u32,
    uid: u32,
    ts: u32, // ms since click window start
    camp: u32,
}

/// click id -> click index (< 2^31) or SPECIAL | index
#[inline(always)]
unsafe fn dec_cid(s: &[u8], ckey: u64) -> u32 {
    if let Some(x) = unsafe { raw_idv(s.as_ptr(), s.len(), b'k', &B36T) } {
        let v = unscr(x, ckey);
        if v < (1 << 31) {
            return v as u32;
        }
    }
    special_id(s)
}

/// packed key of a click id, for byte-wise comparison on ts ties
fn cid_key(c: u32, ckey: u64) -> u64 {
    if c & SPECIAL != 0 {
        return special_key(c & !SPECIAL);
    }
    let mut buf = [0u8; 16];
    buf[0] = b'k';
    let mut v = scramble(c as u64, ckey);
    let mut tmp = [0u8; 13];
    let mut at = 13;
    loop {
        at -= 1;
        tmp[at] = b"0123456789abcdefghijklmnopqrstuvwxyz"[(v % 36) as usize];
        v /= 36;
        if v == 0 {
            break;
        }
    }
    let len = 13 - at;
    buf[1..1 + len].copy_from_slice(&tmp[at..]);
    key(&buf[..1 + len])
}

/// open-addressing map uid number -> class
struct P32 {
    t: Vec<(u32, u32)>,
    mask: usize,
    shift: u32,
}
impl P32 {
    fn new(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(16);
        P32 { t: vec![(u32::MAX, 0); cap], mask: cap - 1, shift: 32 - cap.trailing_zeros() }
    }
    #[inline(always)]
    fn h(&self, k: u32) -> usize {
        (k.wrapping_mul(0x9E37_79B1) >> self.shift) as usize
    }
    fn insert(&mut self, k: u32, v: u32) {
        let mut h = self.h(k);
        loop {
            if self.t[h].0 == u32::MAX {
                self.t[h] = (k, v);
                return;
            }
            if self.t[h].0 == k {
                return;
            }
            h = (h + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn get(&self, k: u32) -> Option<u32> {
        let mut h = self.h(k);
        loop {
            let s = unsafe { *self.t.get_unchecked(h) };
            if s.0 == k {
                return Some(s.1);
            }
            if s.0 == u32::MAX {
                return None;
            }
            h = (h + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn prefetch(&self, k: u32) {
        prefetch(unsafe { self.t.as_ptr().add(self.h(k)) });
    }
}

/// alias slots per user in the direct person table
const AS: u32 = 8;

struct Persons {
    /// uid number n with alias (n & 511) < AS and user (n >> 9) < du: dtab[user * AS + alias] = class
    dtab: Vec<u32>,
    du: u32,
    /// other linked uid numbers -> class
    ptab: P32,
    spec_class: FxHashMap<u32, u32>,
    cname: Vec<u64>,
    cuser: Vec<u32>,
}
impl Persons {
    #[inline(always)]
    fn class_of(&self, u: u32) -> u32 {
        let (user, alias) = (u >> 9, u & 511);
        if alias < AS && user < self.du {
            unsafe { *self.dtab.get_unchecked((user * AS + alias) as usize) }
        } else {
            self.ptab.get(u).unwrap_or(u32::MAX)
        }
    }
    #[inline(always)]
    fn prefetch(&self, u: u32) {
        let (user, alias) = (u >> 9, u & 511);
        if alias < AS && user < self.du {
            prefetch(unsafe { self.dtab.as_ptr().add((user * AS + alias) as usize) });
        }
    }
    /// (person, complex)
    #[inline(always)]
    fn of(&self, u: u32) -> (u64, bool) {
        if u & SPECIAL == 0 {
            let c = self.class_of(u);
            if c != u32::MAX {
                (CLASS_TAG | c as u64, unsafe { *self.cuser.get_unchecked(c as usize) } == NOUSER)
            } else {
                (u as u64, false)
            }
        } else {
            match self.spec_class.get(&(u & !SPECIAL)) {
                Some(&c) => (CLASS_TAG | c as u64, true),
                None => (SPEC_TAG | (u & !SPECIAL) as u64, true),
            }
        }
    }
    fn name(&self, p: u64, ukey: u64) -> u64 {
        if p & CLASS_TAG != 0 {
            self.cname[(p & !CLASS_TAG) as usize]
        } else if p & SPEC_TAG != 0 {
            special_key((p & !SPEC_TAG) as u32)
        } else {
            uid_key(p as u32, ukey)
        }
    }
}

/// Union-find over link IDs with a direct table for (user, alias < AS) IDs and a hash map
/// for the rest; same classes, names and per-class user as build_persons.
fn build_persons2(lparts: &[(Vec<u8>, Vec<(u64, u64, u64, u64)>)]) -> Persons {
    let direct = |id: u64| -> bool { id & SPEC_TAG == 0 && ((id & 511) as u32) < AS };
    let mut maxu = 0u64;
    for s in lparts {
        for &(a, _, b, _) in &s.1 {
            if direct(a) {
                maxu = maxu.max(a >> 9);
            }
            if direct(b) {
                maxu = maxu.max(b >> 9);
            }
        }
    }
    let du = (maxu + 1).min(1 << 22) as u32;
    let nd = (du * AS) as usize;
    let n_l: usize = lparts.iter().map(|s| s.1.len()).sum();
    let mut parent: Vec<u32> = vec![u32::MAX; nd];
    let mut names: Vec<u64> = vec![0; nd];
    let mut over: FxHashMap<u64, u32> = FxHashMap::default();
    let mut over_ids: Vec<u64> = vec![];
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let gp = p[p[x as usize] as usize];
            p[x as usize] = gp;
            x = gp;
        }
        x
    }
    let _ = n_l;
    for s in lparts {
        for &(a, ka, b, kb) in &s.1 {
            let mut ix = |id: u64, k: u64, parent: &mut Vec<u32>, names: &mut Vec<u64>| -> u32 {
                let i = if direct(id) && (id >> 9) < du as u64 {
                    ((id >> 9) as u32) * AS + (id & 511) as u32
                } else {
                    match over.get(&id) {
                        Some(&i) => i,
                        None => {
                            let i = parent.len() as u32;
                            parent.push(u32::MAX);
                            names.push(0);
                            over.insert(id, i);
                            over_ids.push(id);
                            i
                        }
                    }
                };
                if parent[i as usize] == u32::MAX {
                    parent[i as usize] = i;
                    names[i as usize] = k;
                }
                i
            };
            let ia = ix(a, ka, &mut parent, &mut names);
            let ib = ix(b, kb, &mut parent, &mut names);
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    let n = parent.len();
    let mut cls = vec![u32::MAX; n];
    let mut cname: Vec<u64> = vec![];
    let mut cuser: Vec<u32> = vec![];
    let mut dtab: Vec<u32> = vec![u32::MAX; nd];
    let mut ptab = P32::new(over_ids.len().max(16));
    let mut spec_class: FxHashMap<u32, u32> = FxHashMap::default();
    for i in 0..n {
        if parent[i] == u32::MAX {
            continue;
        }
        let r = find(&mut parent, i as u32) as usize;
        let id = if i < nd { (((i as u32) / AS) as u64) << 9 | ((i as u32) % AS) as u64 } else { over_ids[i - nd] };
        let u = if id & SPEC_TAG != 0 { NOUSER } else { (id >> 9) as u32 };
        if cls[r] == u32::MAX {
            cls[r] = cname.len() as u32;
            cname.push(names[i]);
            cuser.push(u);
        }
        let c = cls[r] as usize;
        if cmp_key(names[i], cname[c]) == Ordering::Less {
            cname[c] = names[i];
        }
        if cuser[c] != u {
            cuser[c] = NOUSER;
        }
        if i < nd {
            dtab[i] = c as u32;
        } else if id & SPEC_TAG != 0 {
            spec_class.insert((id & !SPEC_TAG) as u32, c as u32);
        } else {
            ptab.insert(id as u32, c as u32);
        }
    }
    Persons { dtab, du, ptab, spec_class, cname, cuser }
}

/// link ID -> (id, name key): id is the uid number (< 2^31) or SPEC_TAG | special index
#[inline(always)]
unsafe fn link_id(s: &[u8], ukey: u64) -> (u64, u64) {
    let k = unsafe { key_fast(s) };
    let u = unsafe { dec_uid(s, ukey) };
    let id = if u & SPECIAL == 0 { u as u64 } else { SPEC_TAG | (u & !SPECIAL) as u64 };
    (id, k)
}

fn build_persons(lparts: &[(Vec<u8>, Vec<(u64, u64, u64, u64)>)]) -> Persons {
    let n_l: usize = lparts.iter().map(|s| s.1.len()).sum();
    let mut idx: FxHashMap<u64, u32> = FxHashMap::default();
    idx.reserve(n_l * 2);
    let mut ids: Vec<u64> = Vec::with_capacity(n_l * 2);
    let mut names: Vec<u64> = Vec::with_capacity(n_l * 2);
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
        for &(a, ka, b, kb) in &s.1 {
            let mut g = |id: u64, k: u64| -> u32 {
                *idx.entry(id).or_insert_with(|| {
                    ids.push(id);
                    names.push(k);
                    parent.push(parent.len() as u32);
                    (ids.len() - 1) as u32
                })
            };
            let ia = g(a, ka);
            let ib = g(b, kb);
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    drop(idx);
    let n = ids.len();
    let mut cls = vec![u32::MAX; n];
    let mut cname: Vec<u64> = vec![];
    let mut cuser: Vec<u32> = vec![];
    let mut ptab = P32::new(n);
    let mut spec_class: FxHashMap<u32, u32> = FxHashMap::default();
    for i in 0..n {
        let r = find(&mut parent, i as u32) as usize;
        if cls[r] == u32::MAX {
            cls[r] = cname.len() as u32;
            cname.push(names[i]);
            cuser.push(if ids[i] & SPEC_TAG != 0 { NOUSER } else { (ids[i] >> 9) as u32 });
        }
        let c = cls[r] as usize;
        if cmp_key(names[i], cname[c]) == Ordering::Less {
            cname[c] = names[i];
        }
        let u = if ids[i] & SPEC_TAG != 0 { NOUSER } else { (ids[i] >> 9) as u32 };
        if cuser[c] != u {
            cuser[c] = NOUSER;
        }
        if ids[i] & SPEC_TAG != 0 {
            spec_class.insert((ids[i] & !SPEC_TAG) as u32, c as u32);
        } else {
            ptab.insert(ids[i] as u32, c as u32);
        }
    }
    Persons { dtab: vec![], du: 0, ptab, spec_class, cname, cuser }
}

// ---------- sessions for one person ----------
struct SessCx<'a> {
    ckey: u64,
    ws: i64,
    cws: i64,
    ndays: usize,
    organic: u16,
    cat_rank: &'a [usize],
}
#[derive(Default)]
struct SessSc {
    catsum: Vec<i64>,
    touched: Vec<u16>,
    s_start: Vec<i64>,
    s_day: Vec<usize>,
    s_rev: Vec<i64>,
    s_pur: Vec<bool>,
    covered: Vec<(u32, u32)>,
    pstamp: u32,
    cts: Vec<i64>,
    s_nxt: Vec<u32>,
}

/// ev: one person's events sorted by ts; cl: the person's valid clicks sorted by ts.
#[allow(clippy::too_many_arguments)]
fn sess_one(name: u64, ev: &[UEv], cl: &[C16], cx: &SessCx, sc: &mut SessSc, users: &mut Vec<UserRow>, camp_out: &mut [[i64; 4]]) {
    let ws = cx.ws;
    let base_day = ws.div_euclid(DAY);
    let ndays = cx.ndays;
    sc.s_start.clear();
    sc.s_day.clear();
    sc.s_rev.clear();
    sc.s_pur.clear();
    let mut cur_day: i64 = i64::MIN;
    let (mut d_s, mut d_e, mut d_r) = (0i64, 0i64, 0i64);
    let cat_rank = cx.cat_rank;
    let flush = |day: i64, d_s: i64, d_e: i64, d_r: i64, catsum: &mut Vec<i64>, touched: &mut Vec<u16>, users: &mut Vec<UserRow>| {
        let mut top: Option<u16> = None;
        for &c in touched.iter() {
            top = match top {
                None => Some(c),
                Some(t) => {
                    let (v, tv) = (catsum[c as usize], catsum[t as usize]);
                    if v > tv || (v == tv && cat_rank[c as usize] < cat_rank[t as usize]) { Some(c) } else { Some(t) }
                }
            };
        }
        for &c in touched.iter() {
            catsum[c as usize] = 0;
        }
        touched.clear();
        users.push(UserRow { person: name, day: day as i32, sessions: d_s, events: d_e, revenue: d_r, top });
    };
    let j = ev.len();
    let mut k = 0;
    while k < j {
        let start = ev[k].ts as i64 + ws;
        let mut last = ev[k].ts as i64;
        let mut n = 0i64;
        let mut rev = 0i64;
        let mut pur = false;
        let day = start.div_euclid(DAY);
        if day != cur_day {
            if cur_day != i64::MIN {
                flush(cur_day, d_s, d_e, d_r, &mut sc.catsum, &mut sc.touched, users);
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
            if e.tc & 3 == 2 {
                let cat = (e.tc >> 2) as usize;
                pur = true;
                rev += e.price as i64;
                if sc.catsum[cat] == 0 {
                    sc.touched.push(cat as u16);
                }
                sc.catsum[cat] += e.price as i64;
            }
            k += 1;
        }
        d_s += 1;
        d_e += n;
        d_r += rev;
        sc.s_start.push(start);
        sc.s_day.push((day - base_day) as usize);
        sc.s_rev.push(rev);
        sc.s_pur.push(pur);
    }
    if cur_day != i64::MIN {
        flush(cur_day, d_s, d_e, d_r, &mut sc.catsum, &mut sc.touched, users);
    }

    sc.cts.clear();
    sc.cts.extend(cl.iter().map(|c| c.ts as i64 + cx.cws));
    let cts = &sc.cts;
    let mut q = 0usize;
    for s in 0..sc.s_start.len() {
        let st = sc.s_start[s];
        while q < cl.len() && cts[q] <= st {
            q += 1;
        }
        let mut camp = cx.organic;
        if q > 0 && st - cts[q - 1] <= WEEK {
            let t = cts[q - 1];
            let mut best = q - 1;
            let mut r = q - 1;
            while r > 0 && cts[r - 1] == t {
                r -= 1;
                if cmp_key(cid_key(cl[r].cid, cx.ckey), cid_key(cl[best].cid, cx.ckey)) == Ordering::Less {
                    best = r;
                }
            }
            camp = cl[best].camp as u16;
        }
        let a = &mut camp_out[camp as usize * ndays + sc.s_day[s]];
        a[0] += 1;
        a[1] += sc.s_pur[s] as i64;
        a[2] += sc.s_rev[s];
    }
    if !cl.is_empty() && !sc.s_start.is_empty() {
        sc.pstamp += 1;
        let pstamp = sc.pstamp;
        let ns = sc.s_start.len();
        // s_nxt[i]: first session after i on a later day
        sc.s_nxt.clear();
        sc.s_nxt.resize(ns, ns as u32);
        for i in (0..ns.saturating_sub(1)).rev() {
            sc.s_nxt[i] = if sc.s_day[i + 1] != sc.s_day[i] { (i + 1) as u32 } else { sc.s_nxt[i + 1] };
        }
        let (mut lo, mut hi) = (0usize, 0usize);
        for (c, &t) in cl.iter().zip(cts.iter()) {
            while lo < ns && sc.s_start[lo] < t {
                lo += 1;
            }
            if hi < lo {
                hi = lo;
            }
            while hi < ns && sc.s_start[hi] <= t + WEEK {
                hi += 1;
            }
            let kc = c.camp as usize;
            let cv = &mut sc.covered[kc];
            let from = if cv.0 == pstamp { lo.max(cv.1 as usize) } else { lo };
            *cv = (pstamp, hi.max(if cv.0 == pstamp { cv.1 as usize } else { 0 }) as u32);
            let mut x = from;
            while x < hi {
                let d = sc.s_day[x];
                let y = (sc.s_nxt[x] as usize).min(hi);
                camp_out[kc * ndays + d][3] += (y - x) as i64;
                x = y;
            }
        }
    }
}

// ---------- per-user processing ----------
struct ClickQ {
    cks: Vec<C16>,
    ofs: Vec<u32>, // 2^CSH + 1 offsets
}

struct Shared<'a> {
    pers: &'a Persons,
    cq: &'a [ClickQ],
    sparse: &'a FxHashMap<u32, (AtomicBool, Vec<UEv>)>,
    sparse_bits: &'a [u64],
    ukey: u64,
    scx: SessCx<'a>,
}

#[derive(Default)]
struct UserSc {
    sc: SessSc,
    plist: Vec<(u64, bool)>,
    pidx: Vec<u16>,
    cache: Vec<(u32, u16)>,
    evbuf: Vec<UEv>,
    ckbuf: Vec<C16>,
}

struct Out {
    rows: Vec<UserRow>,
    camp: Vec<[i64; 4]>,
    side: Vec<(u64, UEv)>,
}

#[inline(never)]
fn look_slow(uid: u32, plist: &mut Vec<(u64, bool)>, cache: &mut Vec<(u32, u16)>, pers: &Persons) -> u16 {
    let (p, c) = pers.of(uid);
    let i = match plist.iter().position(|x| x.0 == p) {
        Some(i) => i,
        None => {
            plist.push((p, c));
            plist.len() - 1
        }
    } as u16;
    cache.push((uid, i));
    i
}

/// All events of user u (in event-index order), plus its sparse events; complex persons go to the side.
fn finish_user(u: u32, evs: &mut Vec<UEv>, sh: &Shared, us: &mut UserSc, out: &mut Out) {
    if (sh.sparse_bits[(u >> 6) as usize] >> (u & 63)) & 1 != 0 {
        if let Some(ent) = sh.sparse.get(&u) {
            if !ent.0.swap(true, AO::Relaxed) {
                evs.extend_from_slice(&ent.1);
            }
        }
    }
    us.plist.clear();
    us.pidx.clear();
    us.cache.clear();
    let pers = sh.pers;
    let look = |uid: u32, us_plist: &mut Vec<(u64, bool)>, cache: &mut Vec<(u32, u16)>| -> u16 {
        for &(k, i) in cache.iter() {
            if k == uid {
                return i;
            }
        }
        look_slow(uid, us_plist, cache, pers)
    };
    let mut multi = false;
    let (mut lu, mut li) = (u32::MAX, 0u16);
    for e in evs.iter() {
        if e.uid != lu {
            lu = e.uid;
            li = look(e.uid, &mut us.plist, &mut us.cache);
        }
        multi |= li != 0;
        us.pidx.push(li);
    }
    if us.plist.len() > 30000 {
        // pathological: group through the side path instead
        for e in evs.iter() {
            out.side.push((pers.of(e.uid).0, *e));
        }
        return;
    }
    let q = (u >> CSH) as usize;
    let lu = (u & ((1 << CSH) - 1)) as usize;
    let cks: &[C16] = if q < sh.cq.len() {
        let c = &sh.cq[q];
        if c.ofs.is_empty() { &[] } else { &c.cks[c.ofs[lu] as usize..c.ofs[lu + 1] as usize] }
    } else {
        &[]
    };
    let np = us.plist.len();
    for pi in 0..np {
        let (p, complex) = us.plist[pi];
        if complex {
            for (k, e) in evs.iter().enumerate() {
                if us.pidx[k] as usize == pi {
                    out.side.push((p, *e));
                }
            }
            continue;
        }
        // events of this person
        let ev: &[UEv] = if !multi {
            &evs[..]
        } else {
            us.evbuf.clear();
            for (k, e) in evs.iter().enumerate() {
                if us.pidx[k] as usize == pi {
                    us.evbuf.push(*e);
                }
            }
            &us.evbuf[..]
        };
        let mut sorted = true;
        for w in ev.windows(2) {
            if w[1].ts < w[0].ts {
                sorted = false;
                break;
            }
        }
        let ev: &[UEv] = if sorted {
            ev
        } else {
            if multi {
                us.evbuf.sort_by_key(|e| e.ts);
                &us.evbuf[..]
            } else {
                evs.sort_by_key(|e| e.ts);
                &evs[..]
            }
        };
        // clicks of this person
        let mut all = true;
        for c in cks {
            let i = look(c.uid, &mut us.plist, &mut us.cache);
            if i as usize != pi {
                all = false;
                break;
            }
        }
        let cl: &[C16] = if all {
            cks
        } else {
            us.ckbuf.clear();
            for c in cks {
                let i = look(c.uid, &mut us.plist, &mut us.cache);
                if i as usize == pi {
                    us.ckbuf.push(*c);
                }
            }
            &us.ckbuf[..]
        };
        let name = pers.name(p, sh.ukey);
        sess_one(name, ev, cl, &sh.scx, &mut us.sc, &mut out.rows, &mut out.camp);
    }
}

/// results of tasks, one slot each
fn run_map<S: Send, R: Send>(nt: usize, ntasks: usize, init: impl Fn(usize) -> S + Sync, f: impl Fn(&mut S, usize) -> R + Sync) -> (Vec<S>, Vec<R>) {
    let slots: Vec<Mutex<Option<R>>> = (0..ntasks).map(|_| Mutex::new(None)).collect();
    let st = run(nt, ntasks, init, |s, t| {
        let r = f(s, t);
        *slots[t].lock().unwrap() = Some(r);
    });
    (st, slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect())
}

struct EvCtx<'a> {
    ws: i64,
    we: i64,
    ukey: u64,
    ekey: u64,
    vb: u64,
    psh: u32,
    vmask: u32,
    evwin: &'a MonTab,
    evloc: &'a MonTab,
    ingtab: &'a MonTab,
    products: &'a IdTable,
    unknown_cat: u16,
}

/// One event row (fields already unquoted) into the partition write-combiner.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ev_row<const V: u32, const K: usize>(f: [&[u8]; 7], cx: &EvCtx, wc: &mut Wc<DRec, K>, purs: &mut [Vec<PRec>], sparse: &mut Vec<SRec>, maxp: &mut usize) {
    let [eid, uid, ts, ing, et, pid, price] = f;
    let (ws, we, ukey, ekey, vb, psh, vmask) = (cx.ws, cx.we, cx.ukey, cx.ekey, cx.vb, cx.psh, cx.vmask);
    if eid.is_empty() {
        return;
    }
    let ingest_s = unsafe { ingest_win(ing, cx.ingtab, ws) };
    let ingest = ingest_s as i32;
    let mut typ = 3u32;
    let mut tsr = 0u32;
    let mut pr = 0u32;
    let mut cat = 0u32;
    'v: {
        if uid.is_empty() || pid.is_empty() {
            break 'v;
        }
        let Some(t) = (unsafe { ts_win_v::<V>(ts, cx.evwin, cx.evloc) }) else { break 'v };
        if t < ws || t >= we || ingest_s == i32::MAX as i64 || (ingest_s * 1000 + ws) - t > DAY {
            break 'v;
        }
        let ty = match et.len() {
            4 => {
                let w = u32::from_le_bytes(et.try_into().unwrap());
                let is_cart = w == u32::from_le_bytes(*b"cart");
                if !(is_cart | (w == u32::from_le_bytes(*b"view"))) {
                    break 'v;
                }
                is_cart as u32
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
        if ty == 2 {
            match parse_price(price) {
                Some(p) => pr = p,
                None => break 'v,
            }
            cat = (unsafe { cx.products.get_fast(pid) }).unwrap_or(cx.unknown_cat) as u32;
        }
        tsr = (t - ws) as u32;
        typ = ty;
    }
    let u = if typ != 3 { unsafe { dec_uid(uid, ukey) } } else { 0 };
    let v = match unsafe { raw_idv(eid.as_ptr(), eid.len(), b'e', &B36T) } {
        Some(x) => unscr(x, ekey),
        None => u64::MAX,
    };
    if v < vb {
        let p = (v >> psh) as usize;
        let vl = (v as u32) & vmask;
        wc.push(p, DRec { meta: vl | (typ << 22), uid: u, ts: tsr, ingest });
        if typ == 2 {
            unsafe { purs.get_unchecked_mut(p) }.push(PRec { v_low: vl, price: pr, cat });
        }
        *maxp = (*maxp).max(p);
    } else {
        let k = unsafe { key_fast(eid) };
        sparse.push(SRec { eid: k, rec: DRec { meta: typ << 22, uid: u, ts: tsr, ingest }, price: pr, cat });
    }
}

#[cold]
#[inline(never)]
fn ev_row_cold<const K: usize>(f: [&[u8]; 7], cx: &EvCtx, wc: &mut Wc<DRec, K>, purs: &mut [Vec<PRec>], sparse: &mut Vec<SRec>, maxp: &mut usize) {
    ev_row::<0, K>(f, cx, wc, purs, sparse, maxp)
}

/// Hand-flattened event line parser: one inlined row handler, branchless comma extraction,
/// quote flags computed once per line (same results as parse_ev).
#[inline(never)]
fn parse_ev2<const V: u32, const K: usize>(buf: &[u8], s: usize, e: usize, cx: &EvCtx, wc: &mut Wc<DRec, K>, purs: &mut [Vec<PRec>], sparse: &mut Vec<SRec>, maxp: &mut usize) {
    assert!(buf.len() >= e + 128);
    let base = buf.as_ptr();
    let mut ls = s;
    let mut fl = [(0usize, 0usize); 7];
    while ls < e {
        let lim = e - ls;
        let lp = unsafe { base.add(ls) };
        let (mc, mut mn, mq) = masks128(unsafe { buf.get_unchecked(ls..ls + 128) });
        if lim < 128 {
            mn[0] &= low_mask(lim);
            mn[1] &= if lim > 64 { low_mask(lim - 64) } else { 0 };
        }
        let le_off = if mn[0] != 0 {
            mn[0].trailing_zeros() as usize
        } else if mn[1] != 0 {
            64 + mn[1].trailing_zeros() as usize
        } else if lim <= 128 {
            lim
        } else {
            let (ni, ok) = next_line::<7>(&buf[..e], ls, &mut fl);
            if ok {
                let mut a: [&[u8]; 7] = [&[]; 7];
                for k in 0..7 {
                    a[k] = &buf[fl[k].0..fl[k].1];
                }
                ev_row_cold::<K>(a, cx, wc, purs, sparse, maxp);
            }
            ls = ni;
            continue;
        };
        let m0 = low_mask(le_off);
        let m1 = if le_off > 64 { low_mask(le_off - 64) } else { 0 };
        let (c0, c1) = (mc[0] & m0, mc[1] & m1);
        let next = ls + le_off + 1;
        if c0.count_ones() + c1.count_ones() != 6 {
            ls = next;
            continue;
        }
        // comma offsets, branchless
        let mut ends = [0usize; 7];
        {
            let (mut w0, mut w1) = (c0, c1);
            for k in 0..6 {
                let p0 = w0.trailing_zeros() as usize;
                let p1 = 64 + w1.trailing_zeros() as usize;
                let use0 = w0 != 0;
                ends[k] = if use0 { p0 } else { p1 };
                let b0 = w0 & w0.wrapping_sub(1);
                let b1 = w1 & w1.wrapping_sub(1);
                w0 = if use0 { b0 } else { w0 };
                w1 = if use0 { w1 } else { b1 };
            }
        }
        ends[6] = le_off;
        let (q0, q1) = (mq[0] & m0, mq[1] & m1);
        let nq = q0.count_ones() + q1.count_ones();
        let fsb: u64 = if nq == 0 {
            0
        } else if le_off < 127 {
            let sm = [(c0 << 1) | 1, (c1 << 1) | (c0 >> 63)];
            let mut em = [c0, c1];
            em[le_off >> 6] |= 1u64 << (le_off & 63);
            let last = [(em[0] >> 1) | (em[1] << 63), em[1] >> 1];
            let qs = [q0 & sm[0], q1 & sm[1]];
            let ql = [q0 & last[0], q1 & last[1]];
            let qe = [(q0 << 1) & em[0], ((q1 << 1) | (q0 >> 63)) & em[1]];
            let fsb = pext64(qs[0], sm[0]) | (pext64(qs[1], sm[1]) << sm[0].count_ones());
            let feb = pext64(qe[0], em[0]) | (pext64(qe[1], em[1]) << em[0].count_ones());
            if !((qs[0] | ql[0]) == q0 && (qs[1] | ql[1]) == q1 && fsb == feb && nq == 2 * fsb.count_ones()) {
                ls = next;
                continue;
            }
            fsb
        } else {
            let mut fs = 0usize;
            let mut bits = 0u64;
            for k in 0..7 {
                let fe = ends[k];
                let len = fe - fs;
                let lst = if len > 0 { fe - 1 } else { fs };
                let qa = unsafe { *lp.add(fs) } == b'"';
                let qb = unsafe { *lp.add(lst) } == b'"';
                bits |= ((qa & qb & (len >= 2)) as u64) << k;
                fs = fe + 1;
            }
            if nq != 2 * bits.count_ones() {
                ls = next;
                continue;
            }
            bits
        };
        let mut a: [&[u8]; 7] = [&[]; 7];
        let mut fs = 0usize;
        for k in 0..7 {
            let fe = ends[k];
            let q = ((fsb >> k) & 1) as usize;
            a[k] = unsafe { std::slice::from_raw_parts(lp.add(fs + q), fe - fs - 2 * q) };
            fs = fe + 1;
        }
        ev_row::<V, K>(a, cx, wc, purs, sparse, maxp);
        ls = next;
    }
}

/// Parses event lines b[s..e) into the partition write-combiner. V selects code variants.
#[inline(never)]
fn parse_ev<const V: u32, const K: usize>(b: &[u8], s: usize, e: usize, cx: &EvCtx, wc: &mut Wc<DRec, K>, purs: &mut [Vec<PRec>], sparse: &mut Vec<SRec>, maxp: &mut usize) {
    let (ws, we, ukey, ekey, vb, psh, vmask) = (cx.ws, cx.we, cx.ukey, cx.ekey, cx.vb, cx.psh, cx.vmask);
    let (evwin, evloc, ingtab, products, unknown_cat) = (cx.evwin, cx.evloc, cx.ingtab, cx.products, cx.unknown_cat);
    for_rows_fast_v::<7, V>(b, s, e, #[inline(always)] |[eid, uid, ts, ing, et, pid, price]| {
        if V & 256 != 0 {
            std::hint::black_box(eid.len() + uid.len() + ts.len() + ing.len() + et.len() + pid.len() + price.len() + eid.as_ptr() as usize);
            return;
        }
        if eid.is_empty() {
            return;
        }
        let ingest_s = unsafe { ingest_win(ing, ingtab, ws) };
        let ingest = ingest_s as i32;
        let mut typ = 3u32;
        let mut tsr = 0u32;
        let mut pr = 0u32;
        let mut cat = 0u32;
        'v: {
            if uid.is_empty() || pid.is_empty() {
                break 'v;
            }
            let Some(t) = (unsafe { ts_win_v::<V>(ts, evwin, evloc) }) else { break 'v };
            // late: ingest - ts > 24 h (an unparseable ingest counts as late)
            if t < ws || t >= we || ingest_s == i32::MAX as i64 || (ingest_s * 1000 + ws) - t > DAY {
                break 'v;
            }
            let ty = match et.len() {
                4 => {
                    let w = u32::from_le_bytes(et.try_into().unwrap());
                    let is_cart = w == u32::from_le_bytes(*b"cart");
                    if !(is_cart | (w == u32::from_le_bytes(*b"view"))) {
                        break 'v;
                    }
                    is_cart as u32
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
            if ty == 2 {
                match parse_price(price) {
                    Some(p) => pr = p,
                    None => break 'v,
                }
                cat = (unsafe { products.get_fast(pid) }).unwrap_or(unknown_cat) as u32;
            }
            tsr = (t - ws) as u32;
            typ = ty;
        }
        let (v, u) = if V & 16 != 0 {
            // both IDs in one AVX2 pass
            let (ne, nu) = (eid.len(), uid.len());
            let pe = eid.as_ptr();
            let pu = uid.as_ptr();
            let (xe, xu) = unsafe { b36v2(pe.add(1), ne.wrapping_sub(1), pu.add(1), nu.wrapping_sub(1), &B36T) };
            let v = match xe {
                Some(x) if ne >= 2 && unsafe { *pe } == b'e' => unscr(x, ekey),
                _ => u64::MAX,
            };
            let u = if typ == 3 {
                0
            } else {
                match xu {
                    Some(x) if nu >= 2 && unsafe { *pu } == b'u' && unscr(x, ukey) < (1 << 31) => unscr(x, ukey) as u32,
                    _ => special_id(uid),
                }
            };
            (v, u)
        } else {
            let u = if typ != 3 { unsafe { dec_uid(uid, ukey) } } else { 0 };
            let v = match unsafe { raw_idv(eid.as_ptr(), eid.len(), b'e', &B36T) } {
                Some(x) => unscr(x, ekey),
                None => u64::MAX,
            };
            (v, u)
        };
        if v < vb {
            let p = (v >> psh) as usize;
            let vl = (v as u32) & vmask;
            wc.push(p, DRec { meta: vl | (typ << 22), uid: u, ts: tsr, ingest });
            if typ == 2 {
                unsafe { purs.get_unchecked_mut(p) }.push(PRec { v_low: vl, price: pr, cat });
            }
            if p > *maxp {
                *maxp = p;
            }
        } else {
            let k = unsafe { key_fast(eid) };
            sparse.push(SRec { eid: k, rec: DRec { meta: typ << 22, uid: u, ts: tsr, ingest }, price: pr, cat });
        }
    });
}

pub(crate) fn fast_main(data: &Path, outdir: &Path, nt: usize, log: &dyn Fn(&str)) -> Result<(), String> {
    if std::env::var("SESS_SLOW").is_ok() {
        return Err("forced".into());
    }
    let manifest = String::from_utf8_lossy(&read_small(&data.join("manifest.json"))).to_string();
    let ws = manifest_time(&manifest, "window_start");
    let we = manifest_time(&manifest, "window_end");
    let cws = ws - WEEK;
    if we - cws >= (1i64 << 31) || we <= ws {
        return Err("window".into());
    }
    let evwin = MonTab::new(ws, we - 1);
    let ckwin = MonTab::new(cws, we - 1);
    let evloc = MonTab::new(ws - DAY, we + DAY);
    let ckloc = MonTab::new(cws - DAY, we + DAY);
    let ingtab = MonTab::new(ws - DAY, we + 6 * DAY);
    if !evwin.complete || !ckwin.complete || !evloc.complete || !ckloc.complete {
        return Err("months".into());
    }
    let (evwin, ckwin, ingtab, evloc, ckloc) = (&evwin, &ckwin, &ingtab, &evloc, &ckloc);
    {
        let dd = cws.div_euclid(DAY);
        let mut y = 1970i64;
        while days_from_civil(y + 1, 1, 1) <= dd {
            y += 1;
        }
        while days_from_civil(y, 1, 1) > dd {
            y -= 1;
        }
        let _ = YT.set(year_tab(y - 1));
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
    if ncat > 1000 {
        return Err("categories".into());
    }
    let mut order: Vec<usize> = (0..ncat).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(&cat_names[b]));
    let mut cat_rank = vec![0usize; ncat];
    for (r, &c) in order.iter().enumerate() {
        cat_rank[c] = r;
    }
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

    // ---- files ----
    let mut files: Vec<(File, u64)> = vec![];
    let mut maps: Vec<Option<memmap2::Mmap>> = vec![];
    let mut chunks: Vec<Chunk> = vec![];
    let mut lchunks: Vec<Chunk> = vec![];
    let mut total_ev: u64 = 0;
    let mut total_ck: u64 = 0;
    let mut biggest_ev: Option<usize> = None;
    let mut biggest_ck: Option<usize> = None;
    for (kind, dir) in [(Kind::Ev, "events"), (Kind::Ck, "clicks"), (Kind::Ln, "links")] {
        for p in list_csv(&data.join(dir)) {
            let Ok(f) = File::open(&p) else { continue };
            let len = f.metadata().map(|m| m.len()).unwrap_or(0);
            if len == 0 {
                continue;
            }
            let fi = files.len();
            if kind == Kind::Ev {
                total_ev += len;
                if biggest_ev.map(|b| files[b].1 < len).unwrap_or(true) {
                    biggest_ev = Some(fi);
                }
            }
            if kind == Kind::Ck {
                total_ck += len;
                if biggest_ck.map(|b| files[b].1 < len).unwrap_or(true) {
                    biggest_ck = Some(fi);
                }
            }
            maps.push(if *K_MAP2 && kind != Kind::Ln { unsafe { memmap2::Mmap::map(&f) }.ok() } else { None });
            files.push((f, len));
            let step = (*K_CHUNK as u64).max(4096);
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
    let Some(bf) = biggest_ev else { return Err("no events".into()) };

    // ---- key recovery from a sample of the largest events file ----
    let (ekey, ukey, est_lines) = {
        use std::os::unix::fs::FileExt;
        let (f, len) = &files[bf];
        let n = (*len).min(1 << 20) as usize;
        let mut b = vec![0u8; n];
        f.read_exact_at(&mut b, 0).map_err(|e| e.to_string())?;
        let mut em = vec![];
        let mut um = vec![];
        let mut lines = 0usize;
        let mut bytes = 0usize;
        let mut it = b.split(|&c| c == b'\n');
        it.next();
        let mut prev_end_ok = true;
        for line in it {
            if !prev_end_ok {
                break;
            }
            prev_end_ok = true;
            lines += 1;
            bytes += line.len() + 1;
            let fs: Vec<&[u8]> = line.split(|&c| c == b',').collect();
            if fs.len() != 7 {
                continue;
            }
            let strip = |x: &[u8]| -> Vec<u8> {
                if x.len() >= 2 && x[0] == b'"' && x[x.len() - 1] == b'"' { x[1..x.len() - 1].to_vec() } else { x.to_vec() }
            };
            let e = strip(fs[0]);
            let u = strip(fs[1]);
            if e.len() >= 2 && e[0] == b'e' {
                if let Some(x) = b36_safe(&e[1..]) {
                    em.push(unmix(x));
                }
            }
            if u.len() >= 2 && u[0] == b'u' {
                if let Some(x) = b36_safe(&u[1..]) {
                    um.push(unmix(x));
                }
            }
        }
        if lines < 100 {
            return Err("sample".into());
        }
        let avg = bytes as f64 / lines as f64;
        let est_lines = (total_ev as f64 / avg) as u64 + 1024;
        let vb = est_lines * 3 / 2 + 4096;
        let Some(ek) = search_key(&em, 1, vb, vb, nt) else { return Err("event key".into()) };
        let ub = est_lines / 15 + 4096;
        let Some(uk) = search_key(&um, 512, ub, ub * 512, nt) else { return Err("user key".into()) };
        (ek, uk, est_lines)
    };
    // click key from a sample of the largest clicks file
    let ckey = {
        use std::os::unix::fs::FileExt;
        let Some(bc) = biggest_ck else { return Err("no clicks".into()) };
        let (f, len) = &files[bc];
        let n = (*len).min(1 << 20) as usize;
        let mut b = vec![0u8; n];
        f.read_exact_at(&mut b, 0).map_err(|e| e.to_string())?;
        let mut cm = vec![];
        let (mut lines, mut bytes) = (0usize, 0usize);
        let mut it = b.split(|&c| c == b'\n');
        it.next();
        for line in it {
            lines += 1;
            bytes += line.len() + 1;
            let fs: Vec<&[u8]> = line.split(|&c| c == b',').collect();
            if fs.len() != 4 {
                continue;
            }
            let x = fs[0];
            let x = if x.len() >= 2 && x[0] == b'"' && x[x.len() - 1] == b'"' { &x[1..x.len() - 1] } else { x };
            if x.len() >= 2 && x[0] == b'k' {
                if let Some(v) = b36_safe(&x[1..]) {
                    cm.push(unmix(v));
                }
            }
        }
        if lines < 100 {
            return Err("click sample".into());
        }
        let est_ck_lines = (total_ck as f64 / (bytes as f64 / lines as f64)) as u64 + 1024;
        let cb = est_ck_lines * 3 / 2 + 4096;
        let Some(ck) = search_key(&cm, 1, cb, cb, nt) else { return Err("click key".into()) };
        ck
    };
    log("keys");
    let vb = est_lines * 3 / 2 + 4096;
    let psh = {
        let mut s = 10u32;
        while (vb >> s) > *K_PARTS as u64 {
            s += 1;
        }
        s
    };
    if psh > 22 {
        return Err("too big".into());
    }
    let nparts = (vb >> psh) as usize + 1;
    let vmask: u32 = (1 << psh) - 1;
    let est = est_lines as usize;
    // records per (thread, partition): 2^psh event indexes plus duplicates, spread over the threads
    let cap = ((1usize << psh) * 11 / 10) / nt + 64;
    let est_ck = (total_ck / 46) as usize;
    let est_users = (est_lines / 31).max(1) as usize;
    let cap_ck = (est_ck * (1usize << CSH) / est_users) * 6 / 5 / nt + 64;
    let ncq_used = (est_users >> CSH) + 2;

    // ---- links ----
    let lparts = run(
        nt,
        lchunks.len(),
        |_| (Vec::<u8>::new(), Vec::<(u64, u64, u64, u64)>::new()),
        |st, t| {
            let c = &lchunks[t];
            let (f, len) = &files[c.file];
            let (s, e) = read_chunk(f, *len, c, &mut st.0);
            let ln = &mut st.1;
            for_rows_fast::<2>(&st.0, s, e, |[a, bb]| {
                if !a.is_empty() && !bb.is_empty() {
                    let (ia, ka) = unsafe { link_id(a, ukey) };
                    let (ib, kb) = unsafe { link_id(bb, ukey) };
                    ln.push((ia, ka, ib, kb));
                }
            });
        },
    );
    log("links parsed");

    // ---- phase A: parse events and clicks (union-find runs alongside) ----
    struct SA {
        buf: Vec<u8>,
        wc: Wc<DRec, 4>,
        wc8: Wc<DRec, 8>,
        purs: Vec<Vec<PRec>>,
        sparse: Vec<SRec>,
        wck: Wc<C16, 4>,
        spck: Vec<C16>,
        ev: Vec<Vec<DRec>>,
        ck: Vec<Vec<C16>>,
        maxp: usize,
    }
    let products = &products;
    let campaigns = &campaigns;
    let evcx = &EvCtx { ws, we, ukey, ekey, vb, psh, vmask, evwin, evloc, ingtab, products, unknown_cat };
    let (pers, sa) = std::thread::scope(|scope| {
        let lparts = &lparts;
        let uf = scope.spawn(move || {
            let t = std::time::Instant::now();
            let r = if std::env::var_os("SESS_UF1").is_some() { build_persons(lparts) } else { build_persons2(lparts) };
            if std::env::var_os("VERBOSE").is_some() {
                eprintln!("union-find {:.3}s classes {}", t.elapsed().as_secs_f64(), r.cname.len());
            }
            r
        });
        let sa = run(
            nt,
            chunks.len(),
            |_| SA {
                buf: vec![],
                wc: if *K_VAR & 8 == 0 { Wc::new_n(nparts, cap, nparts) } else { Wc::empty() },
                wc8: if *K_VAR & 8 != 0 { Wc::new_n(nparts, cap, nparts) } else { Wc::empty() },
                purs: (0..nparts).map(|_| Vec::with_capacity(((1usize << psh) / 25) / nt + 8)).collect(),
                sparse: vec![],
                wck: Wc::new_n(NCQ, cap_ck, ncq_used),
                spck: vec![],
                ev: vec![],
                ck: vec![],
                maxp: 0,
            },
            |st, t| {
                let c = &chunks[t];
                let (f, len) = &files[c.file];
                let (b, s, e): (&[u8], usize, usize) = match &maps[c.file] {
                    Some(mm) => {
                        let m = &mm[..];
                        let (s0, e0) = chunk_lines(m, c);
                        if e0 + 128 <= m.len() {
                            (m, s0, e0)
                        } else {
                            st.buf.clear();
                            st.buf.extend_from_slice(&m[s0..e0]);
                            st.buf.resize(e0 - s0 + 128, 0);
                            (&st.buf[..], 0, e0 - s0)
                        }
                    }
                    None => {
                        let (s, e) = read_chunk(f, *len, c, &mut st.buf);
                        (&st.buf[..], s, e)
                    }
                };
                match c.kind {
                    Kind::Ev => {
                        let var = *K_VAR;
                        if var & 8 != 0 {
                            let wc = &mut st.wc8;
                            match var & 7 {
                                0 => parse_ev::<0, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                1 => parse_ev::<1, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                3 => parse_ev::<3, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                5 => parse_ev::<5, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                7 => parse_ev2::<0, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                _ => parse_ev::<7, 8>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                            }
                        } else {
                            let wc = &mut st.wc;
                            match var & 23 {
                                16 => parse_ev::<16, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                18 => parse_ev::<64, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                20 => parse_ev::<128, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                22 => parse_ev::<256, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                23 => parse_ev2::<0, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                21 => parse_ev::<129, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                17 => parse_ev::<17, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                0 => parse_ev::<0, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                1 => parse_ev::<1, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                2 => parse_ev::<2, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                3 => parse_ev::<3, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                4 => parse_ev::<4, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                5 => parse_ev::<5, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                6 => parse_ev::<6, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                                _ => parse_ev::<7, 4>(b, s, e, evcx, wc, &mut st.purs, &mut st.sparse, &mut st.maxp),
                            }
                        }
                    }
                    Kind::Ck => {
                        let wck = &mut st.wck;
                        let spck = &mut st.spck;
                        for_rows_fast::<4>(b, s, e, #[inline(always)] |[cid, uid, camp, ts]| {
                            if cid.is_empty() || uid.is_empty() {
                                return;
                            }
                            let Some(t) = (unsafe { ts_win(ts, ckwin, ckloc) }) else { return };
                            if t < cws || t >= we {
                                return;
                            }
                            let Some(ci) = (unsafe { campaigns.get_fast(camp) }) else { return };
                            let u = unsafe { dec_uid(uid, ukey) };
                            let c = C16 { cid: unsafe { dec_cid(cid, ckey) }, uid: u, ts: (t - cws) as u32, camp: ci as u32 };
                            if u & SPECIAL == 0 {
                                wck.push(((u >> 9) >> CSH) as usize, c);
                            } else {
                                spck.push(c);
                            }
                        });
                    }
                    Kind::Ln => {}
                }
            },
        );
        (uf.join().unwrap(), sa)
    });
    let mut sa = sa;
    for s in sa.iter_mut() {
        if *K_VAR & 8 != 0 {
            let w = std::mem::replace(&mut s.wc8, Wc::empty());
            s.ev = w.finish();
        } else {
            let w = std::mem::replace(&mut s.wc, Wc::empty());
            s.ev = w.finish();
        }
        let w = std::mem::replace(&mut s.wck, Wc::empty());
        s.ck = w.finish();
    }
    drop(lparts);
    // unmap the inputs on several threads (one serial teardown at exit otherwise)
    std::thread::scope(|sc| {
        for m in maps.drain(..).flatten() {
            sc.spawn(move || drop(m));
        }
    });
    if std::env::var("FAST_PARSEONLY").is_ok() {
        std::process::exit(0);
    }
    let maxp = sa.iter().map(|s| s.maxp).max().unwrap_or(0);
    let pers = &pers;
    log("parsed");

    // ---- sparse events (IDs that do not decode): dedup among themselves ----
    let mut side_pre: Vec<(u64, UEv)> = vec![];
    let mut sparse_map: FxHashMap<u32, (AtomicBool, Vec<UEv>)> = FxHashMap::default();
    let ubits_len = (1usize << 22) / 64;
    let mut sparse_bits = vec![0u64; ubits_len];
    {
        let mut best: FxHashMap<u64, SRec> = FxHashMap::default();
        for s in sa.iter() {
            for r in &s.sparse {
                match best.get_mut(&r.eid) {
                    Some(o) => {
                        if r.rec.ingest < o.rec.ingest {
                            *o = *r;
                        }
                    }
                    None => {
                        best.insert(r.eid, *r);
                    }
                }
            }
        }
        for r in best.values() {
            let typ = (r.rec.meta >> 22) & 3;
            if typ == 3 {
                continue;
            }
            let e = UEv { uid: r.rec.uid, ts: r.rec.ts, tc: typ | (r.cat << 2), price: r.price };
            if r.rec.uid & SPECIAL != 0 {
                side_pre.push((pers.of(r.rec.uid).0, e));
            } else {
                let u = r.rec.uid >> 9;
                sparse_bits[(u >> 6) as usize] |= 1 << (u & 63);
                sparse_map.entry(u).or_insert_with(|| (AtomicBool::new(false), vec![])).1.push(e);
            }
        }
    }

    // ---- clicks: per user range, drop complex persons' clicks to the side, sort per user ----
    let sa_r = &sa;
    let (ckside, cq) = run_map(
        nt,
        NCQ,
        |_| Vec::<(u64, C16)>::new(),
        |side, q| {
            let n: usize = sa_r.iter().map(|s| s.ck[q].len()).sum();
            if n == 0 {
                return ClickQ { cks: vec![], ofs: vec![] };
            }
            let base = (q as u32) << CSH;
            let mut cnt = vec![0u32; (1 << CSH) + 1];
            let mut cplx: Vec<bool> = Vec::with_capacity(n);
            let mut kept = 0usize;
            for s in sa_r.iter() {
                let v = &s.ck[q];
                for (j, c) in v.iter().enumerate() {
                    if j + 8 < v.len() {
                        pers.prefetch(v[j + 8].uid);
                    }
                    let (p, complex) = pers.of(c.uid);
                    cplx.push(complex);
                    if complex {
                        side.push((p, *c));
                    } else {
                        cnt[((c.uid >> 9) - base) as usize + 1] += 1;
                        kept += 1;
                    }
                }
            }
            for i in 1..cnt.len() {
                cnt[i] += cnt[i - 1];
            }
            let ofs = cnt.clone();
            let mut out: Vec<C16> = Vec::with_capacity(kept);
            unsafe { out.set_len(kept) };
            let mut i = 0usize;
            for s in sa_r.iter() {
                for c in &s.ck[q] {
                    if !cplx[i] {
                        let l = ((c.uid >> 9) - base) as usize;
                        out[cnt[l] as usize] = *c;
                        cnt[l] += 1;
                    }
                    i += 1;
                }
            }
            for l in 0..(1usize << CSH) {
                let (a, b) = (ofs[l] as usize, ofs[l + 1] as usize);
                if b - a > 1 {
                    out[a..b].sort_unstable_by_key(|c| c.ts);
                }
            }
            ClickQ { cks: out, ofs }
        },
    );
    log("clicks");
    // the per-thread click buckets are grouped now; release them before phase B allocates
    let mut sa = sa;
    for x in sa.iter_mut() {
        x.ck = vec![];
    }
    let sa = sa;
    let sa_r = &sa;

    // ---- phase B: event-index order, one user block at a time ----
    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;
    let sh = Shared {
        pers,
        cq: &cq,
        sparse: &sparse_map,
        sparse_bits: &sparse_bits,
        ukey,
        scx: SessCx { ckey, ws, cws, ndays, organic, cat_rank: &cat_rank },
    };
    let np = maxp + 1;
    let nseg = (nt * *K_SEGF).min(np).max(1);
    // Segment boundaries, nudged to partition boundaries that no user seems to straddle (the
    // valid record with the largest index before the boundary and the one with the smallest index
    // after it belong to different users). Only an efficiency heuristic: users that do straddle a
    // boundary are still merged across segments below.
    let bounds: Vec<usize> = {
        let edge = |p: usize, want_max: bool| -> u32 {
            let mut best: Option<(u32, u32)> = None;
            for x in sa.iter() {
                for r in &x.ev[p] {
                    if (r.meta >> 22) & 3 == 3 || r.uid & SPECIAL != 0 {
                        continue;
                    }
                    let vl = r.meta & vmask;
                    let better = match best {
                        None => true,
                        Some((bv, _)) => if want_max { vl > bv } else { vl < bv },
                    };
                    if better {
                        best = Some((vl, r.uid >> 9));
                    }
                }
            }
            best.map(|b| b.1).unwrap_or(NOUSER)
        };
        let mut b = vec![0usize; nseg + 1];
        b[nseg] = np;
        for si in 1..nseg {
            let nominal = si * np / nseg;
            let mut q = nominal.max(b[si - 1] + 1);
            let limit = (q + np / nseg).min(np - 1);
            while q < limit {
                let a = edge(q - 1, true);
                let c = edge(q, false);
                if a != c || a == NOUSER {
                    break;
                }
                q += 1;
            }
            b[si] = q.min(np);
        }
        for si in 1..=nseg {
            if b[si] < b[si - 1] {
                b[si] = b[si - 1];
            }
        }
        b
    };
    let bounds = &bounds;
    let seg_lo = |s: usize| bounds[s];
    struct SB {
        slot: Vec<u32>,
        pbuf: Vec<PRec>,
        cur: Vec<UEv>,
        us: UserSc,
        out: Out,
        pieces: Vec<(usize, u8, u32, Vec<UEv>)>,
        meta: Vec<(usize, u32, u32)>,
        bad: bool,
    }
    let new_out = || Out { rows: Vec::with_capacity(est / nt / 8 + 64), camp: vec![[0; 4]; ncamp * ndays], side: vec![] };
    let new_us = || {
        let mut us = UserSc::default();
        us.sc.catsum = vec![0; ncat];
        us.sc.covered = vec![(0, 0); ncamp + 1];
        us
    };
    let sh_r = &sh;
    let sb = run(
        nt,
        nseg,
        |_| SB { slot: vec![0u32; 1 << psh], pbuf: vec![], cur: vec![], us: new_us(), out: new_out(), pieces: vec![], meta: vec![], bad: false },
        |st, s| {
            if st.bad {
                return;
            }
            let (p0, p1) = (seg_lo(s), seg_lo(s + 1));
            let mut cur_user = NOUSER;
            let mut first_user = NOUSER;
            let mut in_head = s > 0;
            st.cur.clear();
            for p in p0..p1 {
                let parts: Vec<&[DRec]> = sa_r.iter().map(|x| &x.ev[p][..]).collect();
                let slot = &mut st.slot;
                for (ti, sl) in parts.iter().enumerate() {
                    for (i, r) in sl.iter().enumerate() {
                        let vl = (r.meta & vmask) as usize;
                        let code = ((ti as u32) << 26) | (i as u32 + 1);
                        let sv = unsafe { *slot.get_unchecked(vl) };
                        if sv == 0 {
                            unsafe { *slot.get_unchecked_mut(vl) = code };
                        } else {
                            let o = &parts[(sv >> 26) as usize][((sv & ((1 << 26) - 1)) - 1) as usize];
                            if r.ingest < o.ingest {
                                unsafe { *slot.get_unchecked_mut(vl) = code };
                            }
                        }
                    }
                }
                st.pbuf.clear();
                for x in sa_r.iter() {
                    st.pbuf.extend_from_slice(&x.purs[p]);
                }
                st.pbuf.sort_unstable_by_key(|r| r.v_low);
                let mut pc = 0usize;
                let span = 1usize << psh;
                for vl in 0..span {
                    let sv = unsafe { *slot.get_unchecked(vl) };
                    if sv == 0 {
                        continue;
                    }
                    unsafe { *slot.get_unchecked_mut(vl) = 0 };
                    let r = parts[(sv >> 26) as usize][((sv & ((1 << 26) - 1)) - 1) as usize];
                    let typ = (r.meta >> 22) & 3;
                    if typ == 3 {
                        continue;
                    }
                    let mut e = UEv { uid: r.uid, ts: r.ts, tc: typ, price: 0 };
                    if typ == 2 {
                        while st.pbuf[pc].v_low < vl as u32 {
                            pc += 1;
                        }
                        let pr = st.pbuf[pc];
                        e.price = pr.price;
                        e.tc |= pr.cat << 2;
                    }
                    if r.uid & SPECIAL != 0 {
                        st.out.side.push((pers.of(r.uid).0, e));
                        continue;
                    }
                    let u = r.uid >> 9;
                    if u != cur_user {
                        pers.prefetch(r.uid);
                        if cur_user != NOUSER {
                            if u < cur_user {
                                st.bad = true;
                                return;
                            }
                            if in_head {
                                st.pieces.push((s, 0, cur_user, std::mem::take(&mut st.cur)));
                                in_head = false;
                            } else {
                                finish_user(cur_user, &mut st.cur, sh_r, &mut st.us, &mut st.out);
                                st.cur.clear();
                            }
                        } else {
                            first_user = u;
                        }
                        cur_user = u;
                    }
                    st.cur.push(e);
                }
            }
            if cur_user != NOUSER {
                if in_head {
                    st.pieces.push((s, 0, cur_user, std::mem::take(&mut st.cur)));
                } else if s + 1 < nseg {
                    st.pieces.push((s, 1, cur_user, std::mem::take(&mut st.cur)));
                } else {
                    finish_user(cur_user, &mut st.cur, sh_r, &mut st.us, &mut st.out);
                    st.cur.clear();
                }
            }
            st.meta.push((s, first_user, cur_user));
        },
    );
    if sb.iter().any(|s| s.bad) {
        return Err("user order".into());
    }
    // segment order check and boundary merge
    let mut meta: Vec<(usize, u32, u32)> = sb.iter().flat_map(|s| s.meta.iter().copied()).collect();
    meta.sort_unstable();
    let mut prev = 0u32;
    let mut any = false;
    for &(_, f, l) in &meta {
        if f == NOUSER {
            continue;
        }
        if any && f < prev {
            return Err("segment order".into());
        }
        prev = l;
        any = true;
    }
    let mut sb = sb;
    let mut pieces: Vec<(usize, u8, u32, Vec<UEv>)> = sb.iter_mut().flat_map(|s| std::mem::take(&mut s.pieces)).collect();
    pieces.sort_unstable_by_key(|x| (x.0, x.1));
    let mut groups: Vec<(u32, Vec<UEv>)> = vec![];
    for (_, _, u, ev) in pieces {
        match groups.last_mut() {
            Some(g) if g.0 == u => g.1.extend_from_slice(&ev),
            _ => groups.push((u, ev)),
        }
    }
    if std::env::var_os("VERBOSE").is_some() {
        let tot: usize = groups.iter().map(|g| g.1.len()).sum();
        let big = groups.iter().map(|g| g.1.len()).max().unwrap_or(0);
        eprintln!("boundary groups {} events {} largest {}", groups.len(), tot, big);
    }
    let groups_r = &groups;
    let (fix, _) = run_map(
        nt,
        groups.len(),
        |_| (new_us(), new_out()),
        |st, g| {
            let mut ev = groups_r[g].1.clone();
            finish_user(groups_r[g].0, &mut ev, sh_r, &mut st.0, &mut st.1);
        },
    );
    // users whose only valid events had IDs that do not decode
    let mut left_us = new_us();
    let mut left_out = new_out();
    for (&u, ent) in sparse_map.iter() {
        if !ent.0.load(AO::Relaxed) {
            let mut ev = vec![];
            finish_user(u, &mut ev, &sh, &mut left_us, &mut left_out);
        }
    }
    log("users");

    // ---- side path: complex persons ----
    let mut outs: Vec<Out> = sb.into_iter().map(|s| s.out).collect();
    outs.extend(fix.into_iter().map(|s| s.1));
    outs.push(left_out);
    // all side events and clicks, bucketed by person hash; every bucket sorts and runs its own
    // persons, so nothing here is serial
    let mut ev_srcs: Vec<Vec<(u64, UEv)>> = vec![side_pre];
    for o in outs.iter_mut() {
        ev_srcs.push(std::mem::take(&mut o.side));
    }
    let ck_srcs: Vec<Vec<(u64, C16)>> = ckside;
    let spck: Vec<Vec<C16>> = sa.iter().map(|x| x.spck.clone()).collect();
    drop(sa);
    let nsb = 64usize;
    let hb = |p: u64| (mix(p) >> 58) as usize;
    let ev_srcs_r = &ev_srcs;
    let ck_srcs_r = &ck_srcs;
    let spck_r = &spck;
    let (sout, _) = run_map(
        nt,
        nsb,
        |_| (new_us(), new_out(), Vec::<(u64, UEv)>::new(), Vec::<(u64, C16)>::new(), Vec::<UEv>::new(), Vec::<C16>::new()),
        |st, b| {
            let (us, out, ev, ck, e2, c2) = (&mut st.0, &mut st.1, &mut st.2, &mut st.3, &mut st.4, &mut st.5);
            ev.clear();
            ck.clear();
            for v in ev_srcs_r {
                ev.extend(v.iter().filter(|x| hb(x.0) == b));
            }
            if ev.is_empty() {
                return;
            }
            for v in ck_srcs_r {
                ck.extend(v.iter().filter(|x| hb(x.0) == b));
            }
            for v in spck_r {
                for c in v {
                    let p = pers.of(c.uid).0;
                    if hb(p) == b {
                        ck.push((p, *c));
                    }
                }
            }
            ev.sort_unstable_by_key(|x| (x.0, x.1.ts));
            ck.sort_unstable_by_key(|x| (x.0, x.1.ts));
            let mut i = 0;
            while i < ev.len() {
                let p = ev[i].0;
                let mut j = i + 1;
                while j < ev.len() && ev[j].0 == p {
                    j += 1;
                }
                e2.clear();
                e2.extend(ev[i..j].iter().map(|x| x.1));
                c2.clear();
                if p & (CLASS_TAG | SPEC_TAG) == 0 || (p & CLASS_TAG != 0 && pers.cuser[(p & !CLASS_TAG) as usize] != NOUSER) {
                    // simple person routed here (pathological grouping): clicks live with its user
                    let u = if p & CLASS_TAG != 0 { pers.cuser[(p & !CLASS_TAG) as usize] } else { (p as u32) >> 9 };
                    let q = (u >> CSH) as usize;
                    let lu = (u & ((1 << CSH) - 1)) as usize;
                    if q < cq.len() && !cq[q].ofs.is_empty() {
                        for c in &cq[q].cks[cq[q].ofs[lu] as usize..cq[q].ofs[lu + 1] as usize] {
                            if pers.of(c.uid).0 == p {
                                c2.push(*c);
                            }
                        }
                    }
                } else {
                    let a = ck.partition_point(|x| x.0 < p);
                    let bb = ck.partition_point(|x| x.0 <= p);
                    c2.extend(ck[a..bb].iter().map(|x| x.1));
                }
                let name = pers.name(p, ukey);
                sess_one(name, e2, c2, &sh_r.scx, &mut us.sc, &mut out.rows, &mut out.camp);
                i = j;
            }
        },
    );
    outs.extend(sout.into_iter().map(|s| s.1));
    log("side");

    // ---- output ----
    let mut camp = vec![[0i64; 4]; ncamp * ndays];
    for o in &outs {
        for (a, b) in camp.iter_mut().zip(o.camp.iter()) {
            for t in 0..4 {
                a[t] += b[t];
            }
        }
    }
    let ud = outdir.join("user_daily");
    let cd = outdir.join("campaign_daily");
    std::fs::create_dir_all(&ud).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&cd).map_err(|e| e.to_string())?;
    std::thread::scope(|scope| {
        let mut g = 0usize;
        for o in outs.iter() {
            if o.rows.is_empty() && g > 0 {
                continue;
            }
            let ud = &ud;
            let cat_names = &cat_names;
            let gg = g;
            scope.spawn(move || {
                if std::env::var_os("SESS_OLDPQ").is_some() {
                    write_user(&ud.join(format!("part-{gg:03}.parquet")), &o.rows, cat_names)
                } else {
                    write_user_pq(&ud.join(format!("part-{gg:03}.parquet")), &o.rows, cat_names).unwrap()
                }
            });
            g += 1;
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
        if std::env::var_os("SESS_OLDPQ").is_some() {
            write_camp(&cd.join("part-000.parquet"), &c_id, &c_ch, &c_day, &c_v);
        } else {
            write_camp_pq(&cd.join("part-000.parquet"), &c_id, &c_ch, &c_day, &c_v).unwrap();
        }
    });
    log("done");
    Ok(())
}


/// micro-benchmark of the field decoders (FAST_BENCH=<n>), run under callgrind
pub(crate) fn bench(which: u32, data: &Path) {
    let manifest = String::from_utf8_lossy(&read_small(&data.join("manifest.json"))).to_string();
    let ws = manifest_time(&manifest, "window_start");
    let we = manifest_time(&manifest, "window_end");
    {
        let dd = (ws - WEEK).div_euclid(DAY);
        let mut y = 1970i64;
        while days_from_civil(y + 1, 1, 1) <= dd { y += 1; }
        while days_from_civil(y, 1, 1) > dd { y -= 1; }
        let _ = YT.set(year_tab(y - 1));
    }
    let evwin = MonTab::new(ws, we - 1);
    let evloc = MonTab::new(ws - DAY, we + DAY);
    let ingtab = MonTab::new(ws - DAY, we + 6 * DAY);
    let mut buf: Vec<u8> = vec![];
    for p in list_csv(&data.join("events")) {
        let b = std::fs::read(&p).unwrap();
        if let Some(h) = memchr(b'\n', &b) { buf.extend_from_slice(&b[h + 1..]); }
    }
    let n0 = buf.len();
    buf.resize(n0 + 256, 0);
    let mut rows: Vec<[(u32, u32); 7]> = vec![];
    for_rows_fast::<7>(&buf, 0, n0, |a| {
        let mut r = [(0u32, 0u32); 7];
        for k in 0..7 { r[k] = ((a[k].as_ptr() as usize - buf.as_ptr() as usize) as u32, a[k].len() as u32); }
        rows.push(r);
    });
    let f = |r: &[(u32, u32); 7], k: usize| -> &[u8] { &buf[r[k].0 as usize..(r[k].0 + r[k].1) as usize] };
    let t0 = std::time::Instant::now();
    let mut acc = 0u64;
    for _ in 0..3 {
        for r in &rows {
            let x = match which {
                1 => unsafe { raw_idv(f(r, 0).as_ptr(), r[0].1 as usize, b'e', &B36T) }.map(|x| unscr(x, 77)).unwrap_or(0),
                2 => unsafe { raw_id(f(r, 0), b'e') }.map(|x| unscr(x, 77)).unwrap_or(0),
                3 => unsafe { key_fast(f(r, 0)) },
                4 => (unsafe { ts_win(f(r, 2), &evwin, &evloc) }).unwrap_or(0) as u64,
                5 => (unsafe { parse_ts_fast(f(r, 2)) }).unwrap_or(0) as u64,
                6 => (unsafe { ingest_win(f(r, 3), &ingtab, ws) }) as u64,
                7 => (unsafe { parse_ingest(f(r, 3)) }).unwrap_or(0) as u64,
                8 => (unsafe { dec_uid(f(r, 1), 77) }) as u64,
                10 => { let t = f(r, 2); if t.len() == 20 { (unsafe { f1_tab(t.as_ptr(), &evwin) }).unwrap_or(0) as u64 } else { 0 } }
                11 => { let t = f(r, 2); if t.len() == 20 { 1 } else { 0 } }
                14 => { let t = f(r, 2); if t.len() == 20 { probe_f1(t.as_ptr(), &evwin).unwrap_or(0) as u64 } else { 0 } }
                15 => probe_b36v(f(r, 0).as_ptr(), 7).unwrap_or(0),
                12 => { let t = f(r, 2); if t.len() == 13 { (unsafe { digits13(t.as_ptr()) }).unwrap_or(0) as u64 } else { 0 } }
                13 => { let t = f(r, 2); if t.len() == 25 { (unsafe { parse_ts_fast(t) }).unwrap_or(0) as u64 } else { 0 } }
                16 => { let t = f(r, 2); if t.len() == 25 { (unsafe { f2_tab(t.as_ptr(), &evloc) }).unwrap_or(0) as u64 } else { 0 } }
                17 => { let t = f(r, 2); if t.len() == 25 { let a = unsafe { f2_tab(t.as_ptr(), &evloc) }; let b = unsafe { parse_ts_fast(t) }; let ok = match (a, b) { (Some(x), Some(y)) => x == y, (None, None) => true, (None, Some(y)) => y < ws - DAY || y >= we + DAY, _ => false }; if !ok { eprintln!("F2 MISMATCH {:?} {:?} {:?}", String::from_utf8_lossy(t), a, b); } 0 } else { 0 } }
                _ => r[0].1 as u64,
            };
            acc = acc.wrapping_add(std::hint::black_box(x));
        }
    }
    eprintln!("bench {which}: rows {} acc {acc} {:?}", rows.len(), t0.elapsed());
}

#[inline(never)]
pub(crate) fn probe_b36v(p: *const u8, k: usize) -> Option<u64> {
    unsafe { b36v(p, k, &B36T) }
}
#[inline(never)]
pub(crate) fn probe_f1(p: *const u8, t: &MonTab) -> Option<i64> {
    unsafe { f1_tab(p, t) }
}
pub(crate) fn probe_f1_call(p: *const u8) {
    let t = std::hint::black_box(MonTab::new(1_800_000_000_000, 1_801_000_000_000));
    eprintln!("{:?}", probe_f1(p, &t));
}

/// FAST_FUZZ: the inline decoders against the general ones
pub(crate) fn fuzz_fast() {
    let mut rng: u64 = 0x1234_5678_9abc_def1;
    let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut bad = 0u64;
    // windows across years incl. leap Feb and year ends
    let starts = ["2025-01-01T00:00:00Z", "2027-12-20T00:00:00Z", "2028-02-20T00:00:00Z", "2028-12-25T00:00:00Z", "2026-06-24T00:00:00Z"];
    for st in starts {
        let ws = parse_ts(st.as_bytes()).unwrap();
        let we = ws + 14 * DAY;
        {
            let dd = (ws - WEEK).div_euclid(DAY);
            let mut y = 1970i64;
            while days_from_civil(y + 1, 1, 1) <= dd { y += 1; }
            while days_from_civil(y, 1, 1) > dd { y -= 1; }
            // YT is set once per process; only the first window uses the fast table path in parse_ts_fast
            let _ = YT.set(year_tab(y - 1));
        }
        let win = MonTab::new(ws, we - 1);
        let loc = MonTab::new(ws - DAY, we + DAY);
        let ing = MonTab::new(ws - DAY, we + 6 * DAY);
        for _ in 0..400_000 {
            let t = ws - 3 * DAY + (next() % (20 * DAY as u64)) as i64;
            let t = t - t.rem_euclid(1000) + if next() % 3 == 0 { (next() % 1000) as i64 } else { 0 };
            let days = t.div_euclid(DAY);
            let msd = t.rem_euclid(DAY);
            // civil from days
            let mut y = 1970 + days / 366;
            while days_from_civil(y + 1, 1, 1) <= days { y += 1; }
            while days_from_civil(y, 1, 1) > days { y -= 1; }
            let mut mo = 1;
            while mo < 12 && days_from_civil(y, mo + 1, 1) <= days { mo += 1; }
            let d = days - days_from_civil(y, mo, 1) + 1;
            let (h, mi, s) = (msd / 3_600_000, msd / 60_000 % 60, msd / 1000 % 60);
            let mut b: Vec<u8> = match next() % 3 {
                0 => format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z").into_bytes(),
                1 => format!("{}", t).into_bytes(),
                _ => {
                    let oh = (next() % 24) as i64;
                    let om = (next() % 60) as i64;
                    let sign = if next() % 2 == 0 { 1 } else { -1 };
                    let lt = t + sign * (oh * 60 + om) * 60_000;
                    let ldays = lt.div_euclid(DAY);
                    let lms = lt.rem_euclid(DAY);
                    let mut y = 1970 + ldays / 366;
                    while days_from_civil(y + 1, 1, 1) <= ldays { y += 1; }
                    while days_from_civil(y, 1, 1) > ldays { y -= 1; }
                    let mut mo = 1;
                    while mo < 12 && days_from_civil(y, mo + 1, 1) <= ldays { mo += 1; }
                    let d = ldays - days_from_civil(y, mo, 1) + 1;
                    format!("{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}{}{oh:02}:{om:02}", lms / 3_600_000, lms / 60_000 % 60, lms / 1000 % 60, if sign > 0 { '+' } else { '-' }).into_bytes()
                }
            };
            // corruptions
            match next() % 8 {
                0 => { let i = (next() as usize) % b.len(); b[i] = (next() % 128) as u8; }
                1 => { let i = (next() as usize) % b.len(); b[i] = b"0123456789:-TZ+"[(next() % 15) as usize]; }
                2 => { if b.len() == 20 { b[8] = b'3'; b[9] = b'0' + (next() % 3) as u8; } }
                3 => { if b.len() == 20 { b[11] = b'2'; b[12] = b'0' + (next() % 6) as u8; } }
                _ => {}
            }
            let mut pad = b.clone();
            pad.resize(b.len() + 64, b'7');
            let s = &pad[..b.len()];
            let r = unsafe { parse_ts_fast(s) };
            let f = unsafe { ts_win(s, &win, &loc) };
            let rin = r.filter(|&x| x >= ws && x < we);
            let fin = f.filter(|&x| x >= ws && x < we);
            if rin != fin {
                bad += 1;
                if bad < 10 { eprintln!("TS MISMATCH {:?} ref {:?} fast {:?}", String::from_utf8_lossy(s), r, f); }
            }
            if b.len() == 20 && r.is_some() && b[19] == b'Z' {
                let gi = unsafe { ingest_win(s, &ing, ws) };
                let ri = (r.unwrap().saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64);
                if gi != ri {
                    bad += 1;
                    if bad < 10 { eprintln!("INGEST MISMATCH {:?} ref {} fast {}", String::from_utf8_lossy(s), ri, gi); }
                }
            }
        }
    }
    // base36
    let digs = b"0123456789abcdefghijklmnopqrstuvwxyzA_ -";
    for _ in 0..3_000_000 {
        let k = 1 + (next() % 10) as usize;
        let mut b: Vec<u8> = (0..k).map(|_| digs[(next() % if next() % 10 == 0 { 40 } else { 36 }) as usize]).collect();
        let mut pad = b.clone();
        pad.resize(k + 32, b'9');
        let f = unsafe { b36v(pad.as_ptr(), k, &B36T) };
        let r = (|| {
            if k > 8 || (k > 1 && b[0] == b'0') { return None; }
            let mut v: u64 = 0;
            for &c in &b {
                let d = match c { b'0'..=b'9' => c - b'0', b'a'..=b'z' => c - b'a' + 10, _ => return None } as u64;
                v = v * 36 + d;
            }
            if v > MASK40 { None } else { Some(v) }
        })();
        let s = unsafe { b36(pad.as_ptr(), k) };
        if f != r || s != r {
            bad += 1;
            if bad < 20 { eprintln!("B36 MISMATCH {:?} ref {:?} sse {:?} swar {:?}", String::from_utf8_lossy(&b), r, f, s); }
        }
        b.clear();
    }
    // IdTable fast lookup
    let mut camps = IdTable::new(b"c_");
    for i in [1usize, 2, 7, 10, 99, 100, 1999, 2000, 123456] {
        camps.insert_if_absent(format!("c_{i}").as_bytes(), i as u16);
    }
    camps.insert_if_absent(b"C_5", 77);
    camps.insert_if_absent(b"weird", 78);
    let atoms: [&[u8]; 16] = [b"c_", b"C_", b"c", b"_", b"0", b"1", b"2", b"7", b"9", b"99", b"100", b"x", b" ", b"5", b"weird", b"1999"];
    for _ in 0..2_000_000 {
        let mut b: Vec<u8> = vec![];
        for _ in 0..(1 + next() % 4) {
            b.extend_from_slice(atoms[(next() % 16) as usize]);
        }
        let mut pad = b.clone();
        pad.resize(b.len() + 16, b'3');
        let f = unsafe { camps.get_fast(&pad[..b.len()]) };
        let r = camps.get(&b);
        if f != r {
            bad += 1;
            if bad < 20 { eprintln!("IDTAB MISMATCH {:?} {:?} {:?}", String::from_utf8_lossy(&b), f, r); }
        }
    }
    eprintln!("fuzz_fast done, mismatches: {bad}");
}


fn write_user_pq(path: &Path, rows: &[UserRow], cat_names: &[Vec<u8>]) -> std::io::Result<()> {
    use crate::pq::{Col, Kind as K};
    let n = rows.len();
    let mut uid = Col::new("user_id", K::Str, n * 13);
    let mut day = Col::new("day", K::Date, n * 4);
    let mut ses = Col::new("sessions", K::I64, n * 8);
    let mut evc = Col::new("events", K::I64, n * 8);
    let mut rev = Col::new("revenue_cents", K::I64, n * 8);
    let mut top = Col::new("top_category", K::OptStr, n * 6);
    top.defs.reserve(n);
    for r in rows {
        let k = r.person;
        if k & FALLBACK == 0 {
            let mut b = [0u8; 9];
            let mut len = 0;
            while len < 9 {
                let c = ((k >> (7 * (8 - len))) & 0x7F) as u8;
                if c == 0 {
                    break;
                }
                b[len] = c;
                len += 1;
            }
            uid.put_str(&b[..len]);
        } else {
            uid.put_str(&key_bytes(k));
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
    crate::pq::write(path, n, &[uid, day, ses, evc, rev, top])
}

fn write_camp_pq(path: &Path, ids: &[Vec<u8>], ch: &[Vec<u8>], day: &[i32], v: &[[i64; 4]]) -> std::io::Result<()> {
    use crate::pq::{Col, Kind as K};
    let n = ids.len();
    let mut c0 = Col::new("campaign_id", K::Str, n * 12);
    let mut c1 = Col::new("channel", K::Str, n * 12);
    let mut c2 = Col::new("day", K::Date, n * 4);
    let mut c3 = Col::new("sessions", K::I64, n * 8);
    let mut c4 = Col::new("purchasing_sessions", K::I64, n * 8);
    let mut c5 = Col::new("revenue_cents", K::I64, n * 8);
    let mut c6 = Col::new("assisted_sessions", K::I64, n * 8);
    for i in 0..n {
        c0.put_str(&ids[i]);
        c1.put_str(&ch[i]);
        c2.put_i32(day[i]);
        c3.put_i64(v[i][0]);
        c4.put_i64(v[i][1]);
        c5.put_i64(v[i][2]);
        c6.put_i64(v[i][3]);
    }
    crate::pq::write(path, n, &[c0, c1, c2, c3, c4, c5, c6])
}

pub(crate) fn pq_empty_test(dir: &Path) {
    write_user_pq(&dir.join("u.parquet"), &[], &[]).unwrap();
    write_camp_pq(&dir.join("c.parquet"), &[], &[], &[], &[]).unwrap();
    let rows = vec![UserRow { person: key(b"u1"), day: 20000, sessions: 1, events: 2, revenue: 0, top: None }; 17];
    write_user_pq(&dir.join("n.parquet"), &rows, &[b"x".to_vec()]).unwrap();
}
