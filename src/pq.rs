// Minimal Parquet writer: one row group, one PLAIN uncompressed data page (v1) per column.
// Columns: required/optional BYTE_ARRAY (UTF8), required INT32 (DATE), required INT64.

pub(crate) enum Kind {
    Str,
    OptStr,
    Date,
    I64,
}

pub(crate) struct Col {
    pub name: &'static str,
    pub kind: Kind,
    /// PLAIN-encoded non-null values
    pub data: Vec<u8>,
    /// definition levels (OptStr only): 1 = present
    pub defs: Vec<u8>,
}

impl Col {
    pub fn new(name: &'static str, kind: Kind, cap: usize) -> Col {
        Col { name, kind, data: Vec::with_capacity(cap), defs: vec![] }
    }
    #[inline(always)]
    pub fn put_str(&mut self, s: &[u8]) {
        self.data.extend_from_slice(&(s.len() as u32).to_le_bytes());
        self.data.extend_from_slice(s);
        if let Kind::OptStr = self.kind {
            self.defs.push(1);
        }
    }
    #[inline(always)]
    pub fn put_null(&mut self) {
        self.defs.push(0);
    }
    #[inline(always)]
    pub fn put_i32(&mut self, v: i32) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }
    #[inline(always)]
    pub fn put_i64(&mut self, v: i64) {
        self.data.extend_from_slice(&v.to_le_bytes());
    }
}

struct T {
    b: Vec<u8>,
    last: Vec<i16>,
}
impl T {
    fn new() -> T {
        T { b: vec![], last: vec![0] }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn zz(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn field(&mut self, id: i16, ty: u8) {
        let d = id - *self.last.last().unwrap();
        if d > 0 && d <= 15 {
            self.b.push(((d as u8) << 4) | ty);
        } else {
            self.b.push(ty);
            self.zz(id as i64);
        }
        *self.last.last_mut().unwrap() = id;
    }
    fn i32f(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.zz(v as i64);
    }
    fn i64f(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.zz(v);
    }
    fn strf(&mut self, id: i16, s: &[u8]) {
        self.field(id, 8);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn begin(&mut self, id: i16) {
        self.field(id, 12);
        self.last.push(0);
    }
    fn end(&mut self) {
        self.b.push(0);
        self.last.pop();
    }
    fn list(&mut self, id: i16, ety: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.b.push(((n as u8) << 4) | ety);
        } else {
            self.b.push(0xF0 | ety);
            self.varint(n as u64);
        }
    }
    /// struct element inside a list
    fn elem_begin(&mut self) {
        self.last.push(0);
    }
}

pub(crate) fn write(path: &std::path::Path, nrows: usize, cols: &[Col]) -> std::io::Result<()> {
    use std::io::Write;
    let mut out: Vec<u8> = Vec::with_capacity(cols.iter().map(|c| c.data.len() + c.defs.len() / 8 + 64).sum::<usize>() + 4096);
    out.extend_from_slice(b"PAR1");
    let mut metas: Vec<(i64, i64)> = vec![]; // (page offset, chunk size)
    for c in cols {
        let off = out.len() as i64;
        let mut body: Vec<u8> = vec![];
        if let Kind::OptStr = c.kind {
            // RLE/bit-packed hybrid, bit width 1, one bit-packed run
            let groups = c.defs.len().div_ceil(8);
            let mut lv: Vec<u8> = vec![];
            let mut h = ((groups as u64) << 1) | 1;
            while h >= 0x80 {
                lv.push((h as u8) | 0x80);
                h >>= 7;
            }
            lv.push(h as u8);
            for g in 0..groups {
                let mut byte = 0u8;
                for i in 0..8 {
                    if let Some(&d) = c.defs.get(g * 8 + i) {
                        byte |= d << i;
                    }
                }
                lv.push(byte);
            }
            body.extend_from_slice(&(lv.len() as u32).to_le_bytes());
            body.extend_from_slice(&lv);
        }
        let mut ph = T::new();
        ph.i32f(1, 0); // DATA_PAGE
        let size = (body.len() + c.data.len()) as i32;
        ph.i32f(2, size);
        ph.i32f(3, size);
        ph.begin(5);
        ph.i32f(1, nrows as i32);
        ph.i32f(2, 0); // PLAIN
        ph.i32f(3, 3); // RLE
        ph.i32f(4, 3); // RLE
        ph.end();
        ph.b.push(0);
        out.extend_from_slice(&ph.b);
        out.extend_from_slice(&body);
        out.extend_from_slice(&c.data);
        metas.push((off, out.len() as i64 - off));
    }
    let mut m = T::new();
    m.i32f(1, 1);
    m.list(2, 12, cols.len() + 1);
    // root
    m.elem_begin();
    m.strf(4, b"schema");
    m.i32f(5, cols.len() as i32);
    m.end();
    for c in cols {
        m.elem_begin();
        let (ty, rep, conv) = match c.kind {
            Kind::Str => (6, 0, Some(0)),
            Kind::OptStr => (6, 1, Some(0)),
            Kind::Date => (1, 0, Some(6)),
            Kind::I64 => (2, 0, None),
        };
        m.i32f(1, ty);
        m.i32f(3, rep);
        m.strf(4, c.name.as_bytes());
        if let Some(cv) = conv {
            m.i32f(6, cv);
        }
        match c.kind {
            Kind::Str | Kind::OptStr => {
                m.begin(10);
                m.begin(1); // STRING
                m.end();
                m.end();
            }
            Kind::Date => {
                m.begin(10);
                m.begin(6); // DATE
                m.end();
                m.end();
            }
            Kind::I64 => {}
        }
        m.end();
    }
    m.i64f(3, nrows as i64);
    m.list(4, 12, 1);
    m.elem_begin();
    m.list(1, 12, cols.len());
    let mut total = 0i64;
    for (c, &(off, sz)) in cols.iter().zip(metas.iter()) {
        total += sz;
        m.elem_begin();
        m.i64f(2, off);
        m.begin(3);
        let ty = match c.kind {
            Kind::Str | Kind::OptStr => 6,
            Kind::Date => 1,
            Kind::I64 => 2,
        };
        m.i32f(1, ty);
        m.list(2, 5, 2);
        m.zz(0);
        m.zz(3);
        m.list(3, 8, 1);
        m.varint(c.name.len() as u64);
        m.b.extend_from_slice(c.name.as_bytes());
        m.i32f(4, 0); // UNCOMPRESSED
        m.i64f(5, nrows as i64);
        m.i64f(6, sz);
        m.i64f(7, sz);
        m.i64f(9, off);
        m.end();
        m.end();
    }
    m.i64f(2, total);
    m.i64f(3, nrows as i64);
    m.end();
    m.strf(6, b"sess");
    m.b.push(0);
    out.extend_from_slice(&m.b);
    out.extend_from_slice(&(m.b.len() as u32).to_le_bytes());
    out.extend_from_slice(b"PAR1");
    let mut f = std::fs::File::create(path)?;
    f.write_all(&out)?;
    Ok(())
}
