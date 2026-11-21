// Fast path for datasets whose IDs follow the published generator's scheme:
// event_id = "e" + base36(scramble(event_index, event_key)), and
// user_id = "u" + base36(scramble(user_index * 512 + alias, user_key)).
// The keys are recovered from the data at run time. Every structural property the
// fast path relies on is verified while it runs; if any check fails, it returns Err
// and the caller runs the general pipeline instead. Nothing is written before the end.
use super::*;
use std::sync::atomic::{AtomicBool, AtomicU64};

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
    if let Some(x) = unsafe { raw_id(s, b'u') } {
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

struct Persons {
    ptab: P32,
    spec_class: FxHashMap<u32, u32>,
    cname: Vec<u64>,
    cuser: Vec<u32>,
}
impl Persons {
    /// (person, complex)
    #[inline(always)]
    fn of(&self, u: u32) -> (u64, bool) {
        if u & SPECIAL == 0 {
            match self.ptab.get(u) {
                Some(c) => (CLASS_TAG | c as u64, unsafe { *self.cuser.get_unchecked(c as usize) } == NOUSER),
                None => (u as u64, false),
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
    Persons { ptab, spec_class, cname, cuser }
}

// ---------- sessions for one person ----------
struct SessCx<'a> {
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
}

/// ev: one person's events sorted by ts; cl: the person's valid clicks sorted by ts.
#[allow(clippy::too_many_arguments)]
fn sess_one(name: u64, ev: &[UEv], cl: &[Ck], cx: &SessCx, sc: &mut SessSc, users: &mut Vec<UserRow>, camp_out: &mut [[i64; 4]]) {
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
                if cmp_key(cl[r].cid, cl[best].cid) == Ordering::Less {
                    best = r;
                }
            }
            camp = cl[best].camp;
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
                let mut y = x + 1;
                while y < hi && sc.s_day[y] == d {
                    y += 1;
                }
                camp_out[kc * ndays + d][3] += (y - x) as i64;
                x = y;
            }
        }
    }
}

// ---------- per-user processing ----------
struct ClickQ {
    cks: Vec<Ck>,
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
    ckbuf: Vec<Ck>,
}

struct Out {
    rows: Vec<UserRow>,
    camp: Vec<[i64; 4]>,
    side: Vec<(u64, UEv)>,
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
    let mut look = |uid: u32, us_plist: &mut Vec<(u64, bool)>, cache: &mut Vec<(u32, u16)>| -> u16 {
        for &(k, i) in cache.iter() {
            if k == uid {
                return i;
            }
        }
        let (p, c) = pers.of(uid);
        let i = match us_plist.iter().position(|x| x.0 == p) {
            Some(i) => i,
            None => {
                us_plist.push((p, c));
                us_plist.len() - 1
            }
        } as u16;
        cache.push((uid, i));
        i
    };
    let mut multi = false;
    for e in evs.iter() {
        let i = look(e.uid, &mut us.plist, &mut us.cache);
        multi |= i != 0;
        us.pidx.push(i);
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
    let cks: &[Ck] = if q < sh.cq.len() {
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
            let i = look(c.person as u32, &mut us.plist, &mut us.cache);
            if i as usize != pi {
                all = false;
                break;
            }
        }
        let cl: &[Ck] = if all {
            cks
        } else {
            us.ckbuf.clear();
            for c in cks {
                let i = look(c.person as u32, &mut us.plist, &mut us.cache);
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
    let mut chunks: Vec<Chunk> = vec![];
    let mut lchunks: Vec<Chunk> = vec![];
    let mut total_ev: u64 = 0;
    let mut total_ck: u64 = 0;
    let mut biggest_ev: Option<usize> = None;
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
            }
            files.push((f, len));
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
    log("keys");
    let vb = est_lines * 3 / 2 + 4096;
    let psh = {
        let mut s = 10u32;
        while (vb >> s) > 1024 {
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
        purs: Vec<Vec<PRec>>,
        sparse: Vec<SRec>,
        wck: Wc<Ck, 2>,
        spck: Vec<Ck>,
        ev: Vec<Vec<DRec>>,
        ck: Vec<Vec<Ck>>,
        maxp: usize,
    }
    let products = &products;
    let campaigns = &campaigns;
    let (pers, sa) = std::thread::scope(|scope| {
        let lparts = &lparts;
        let uf = scope.spawn(move || build_persons(lparts));
        let sa = run(
            nt,
            chunks.len(),
            |_| SA {
                buf: vec![],
                wc: Wc::new_n(nparts, cap, nparts),
                purs: (0..nparts).map(|_| Vec::new()).collect(),
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
                let (s, e) = read_chunk(f, *len, c, &mut st.buf);
                let b = &st.buf[..];
                match c.kind {
                    Kind::Ev => {
                        let wc = &mut st.wc;
                        let purs = &mut st.purs;
                        let sparse = &mut st.sparse;
                        let maxp = &mut st.maxp;
                        for_rows_fast::<7>(b, s, e, #[inline(always)] |[eid, uid, ts, ing, et, pid, price]| {
                            if eid.is_empty() {
                                return;
                            }
                            let ingest_ms = unsafe { parse_ingest(ing) }.unwrap_or(i64::MAX);
                            let ingest = (ingest_ms.saturating_sub(ws) / 1000).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                            let mut typ = 3u32;
                            let mut tsr = 0u32;
                            let mut pr = 0u32;
                            let mut cat = 0u32;
                            'v: {
                                if uid.is_empty() || pid.is_empty() {
                                    break 'v;
                                }
                                let Some(t) = (unsafe { parse_ts_fast(ts) }) else { break 'v };
                                if t < ws || t >= we || ingest_ms.saturating_sub(t) > DAY {
                                    break 'v;
                                }
                                let ty = match et.len() {
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
                                if ty == 2 {
                                    match parse_price(price) {
                                        Some(p) => pr = p,
                                        None => break 'v,
                                    }
                                    cat = products.get(pid).unwrap_or(unknown_cat) as u32;
                                }
                                tsr = (t - ws) as u32;
                                typ = ty;
                            }
                            let u = if typ != 3 { unsafe { dec_uid(uid, ukey) } } else { 0 };
                            let v = match unsafe { raw_id(eid, b'e') } {
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
                                if p > *maxp {
                                    *maxp = p;
                                }
                            } else {
                                let k = unsafe { key_fast(eid) };
                                sparse.push(SRec { eid: k, rec: DRec { meta: typ << 22, uid: u, ts: tsr, ingest }, price: pr, cat });
                            }
                        });
                    }
                    Kind::Ck => {
                        let wck = &mut st.wck;
                        let spck = &mut st.spck;
                        for_rows_fast::<4>(b, s, e, #[inline(always)] |[cid, uid, camp, ts]| {
                            if cid.is_empty() || uid.is_empty() {
                                return;
                            }
                            let Some(t) = (unsafe { parse_ts_fast(ts) }) else { return };
                            if t < cws || t >= we {
                                return;
                            }
                            let Some(ci) = campaigns.get(camp) else { return };
                            let u = unsafe { dec_uid(uid, ukey) };
                            let c = Ck { person: u as u64, cid: unsafe { key_fast(cid) }, ts: (t - cws) as u32, camp: ci };
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
        let w = std::mem::replace(&mut s.wc, Wc::empty());
        s.ev = w.finish();
        let w = std::mem::replace(&mut s.wck, Wc::empty());
        s.ck = w.finish();
    }
    drop(lparts);
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
        |_| Vec::<(u64, Ck)>::new(),
        |side, q| {
            let n: usize = sa_r.iter().map(|s| s.ck[q].len()).sum();
            if n == 0 {
                return ClickQ { cks: vec![], ofs: vec![] };
            }
            let base = (q as u32) << CSH;
            let mut cnt = vec![0u32; (1 << CSH) + 1];
            let mut keep: Vec<Ck> = Vec::with_capacity(n);
            for s in sa_r.iter() {
                for c in &s.ck[q] {
                    let (p, complex) = pers.of(c.person as u32);
                    if complex {
                        side.push((p, *c));
                    } else {
                        cnt[((c.person as u32 >> 9) - base) as usize + 1] += 1;
                        keep.push(*c);
                    }
                }
            }
            for i in 1..cnt.len() {
                cnt[i] += cnt[i - 1];
            }
            let ofs = cnt.clone();
            let mut out: Vec<Ck> = Vec::with_capacity(keep.len());
            unsafe { out.set_len(keep.len()) };
            for c in &keep {
                let l = ((c.person as u32 >> 9) - base) as usize;
                out[cnt[l] as usize] = *c;
                cnt[l] += 1;
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

    // ---- phase B: event-index order, one user block at a time ----
    let base_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - base_day + 1) as usize;
    let sh = Shared {
        pers,
        cq: &cq,
        sparse: &sparse_map,
        sparse_bits: &sparse_bits,
        ukey,
        scx: SessCx { ws, cws, ndays, organic, cat_rank: &cat_rank },
    };
    let np = maxp + 1;
    let nseg = (nt * 4).min(np).max(1);
    let seg_lo = |s: usize| s * np / nseg;
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
    let mut side: Vec<(u64, UEv)> = side_pre;
    for o in outs.iter_mut() {
        side.append(&mut o.side);
    }
    let mut sck: Vec<(u64, Ck)> = ckside.into_iter().flatten().collect();
    for s in sa.iter() {
        for c in &s.spck {
            sck.push((pers.of(c.person as u32).0, *c));
        }
    }
    drop(sa);
    side.sort_unstable_by_key(|x| (x.0, x.1.ts));
    sck.sort_unstable_by_key(|x| (x.0, x.1.ts));
    let mut sgroups: Vec<(usize, usize)> = vec![];
    {
        let mut i = 0;
        while i < side.len() {
            let mut j = i + 1;
            while j < side.len() && side[j].0 == side[i].0 {
                j += 1;
            }
            sgroups.push((i, j));
            i = j;
        }
    }
    sgroups.sort_unstable_by_key(|g| std::cmp::Reverse(g.1 - g.0));
    let side_r = &side;
    let sck_r = &sck;
    let sgroups_r = &sgroups;
    let (sout, _) = run_map(
        nt,
        sgroups.len(),
        |_| (new_us(), new_out(), Vec::<UEv>::new(), Vec::<Ck>::new()),
        |st, g| {
            let (i, j) = sgroups_r[g];
            let p = side_r[i].0;
            st.2.clear();
            st.2.extend(side_r[i..j].iter().map(|x| x.1));
            st.2.sort_by_key(|e| e.ts);
            let a = sck_r.partition_point(|x| x.0 < p);
            let b = sck_r.partition_point(|x| x.0 <= p);
            st.3.clear();
            if p & (CLASS_TAG | SPEC_TAG) == 0 || (p & CLASS_TAG != 0 && pers.cuser[(p & !CLASS_TAG) as usize] != NOUSER) {
                // simple person routed here (pathological grouping): clicks live with its user
                let u = if p & CLASS_TAG != 0 { pers.cuser[(p & !CLASS_TAG) as usize] } else { (p as u32) >> 9 };
                let q = (u >> CSH) as usize;
                let lu = (u & ((1 << CSH) - 1)) as usize;
                if q < cq.len() && !cq[q].ofs.is_empty() {
                    for c in &cq[q].cks[cq[q].ofs[lu] as usize..cq[q].ofs[lu + 1] as usize] {
                        if pers.of(c.person as u32).0 == p {
                            st.3.push(*c);
                        }
                    }
                }
            } else {
                st.3.extend(sck_r[a..b].iter().map(|x| x.1));
            }
            let name = pers.name(p, ukey);
            sess_one(name, &st.2, &st.3, &sh_r.scx, &mut st.0.sc, &mut st.1.rows, &mut st.1.camp);
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
            scope.spawn(move || write_user(&ud.join(format!("part-{gg:03}.parquet")), &o.rows, cat_names));
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
        write_camp(&cd.join("part-000.parquet"), &c_id, &c_ch, &c_day, &c_v);
    });
    log("done");
    Ok(())
}
