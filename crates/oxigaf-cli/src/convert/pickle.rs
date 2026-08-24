//! A pure-Rust unpickler for the object graphs FLAME `.pkl` models contain.
//!
//! It implements the pickle opcodes of protocols 0 through 5 and reconstructs
//! `numpy.ndarray`, `numpy.dtype`, `chumpy.ch.Ch` and `scipy.sparse` payloads
//! into [`NpyArray`] values. Opcodes that cannot be honoured without a Python
//! interpreter (extension registry, persistent ids, out-of-band buffers) return
//! an explicit error naming the opcode instead of guessing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{anyhow, bail, Context, Result};

use super::npy::{self, NpyArray};

/// A value on the unpickling stack.
#[derive(Clone)]
pub enum Value {
    /// Python `None`.
    None,
    /// Python `bool`.
    Bool(bool),
    /// Python `int`.
    Int(i64),
    /// Python `float`.
    Float(f64),
    /// Python 2 `str` / Python 3 `bytes`.
    Bytes(Rc<Vec<u8>>),
    /// Python `unicode` / Python 3 `str`.
    Str(Rc<String>),
    /// Python `tuple`.
    Tuple(Rc<Vec<Value>>),
    /// Python `list` (or `set`, which is treated as a list).
    List(Rc<RefCell<Vec<Value>>>),
    /// Python `dict`, kept as insertion-ordered pairs.
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    /// A resolved global reference (`module`, `name`).
    Global(Rc<(String, String)>),
    /// A reconstructed instance.
    Object(Rc<RefCell<Object>>),
    /// Stack marker pushed by the `MARK` opcode.
    Mark,
}

/// A reconstructed Python object.
pub struct Object {
    /// Defining module of the class.
    pub module: String,
    /// Class name.
    pub name: String,
    /// Constructor arguments.
    pub args: Vec<Value>,
    /// `__setstate__` payload.
    pub state: Option<Value>,
    /// Items assigned through `SETITEM`/`SETITEMS`.
    pub entries: Vec<(Value, Value)>,
}

impl Value {
    /// Interpret the value as text (Python 2 `str` is decoded as latin-1).
    pub fn as_text(&self) -> Option<String> {
        match self {
            Value::Str(s) => Some(s.as_ref().clone()),
            Value::Bytes(b) => Some(b.iter().map(|&c| char::from(c)).collect()),
            _ => None,
        }
    }

    /// Numeric value of an `int`, `float` or `bool` entry.
    fn as_number(&self) -> Option<f64> {
        match self {
            Value::Int(v) => Some(*v as f64),
            Value::Float(v) => Some(*v),
            Value::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Raw bytes of a `str`/`bytes` value (text is latin-1 encoded).
    fn as_raw_bytes(&self) -> Option<Vec<u8>> {
        match self {
            Value::Bytes(b) => Some(b.as_ref().clone()),
            Value::Str(s) => {
                let mut out = Vec::with_capacity(s.len());
                for ch in s.chars() {
                    let code = u32::from(ch);
                    if code > 0xff {
                        return None;
                    }
                    out.push(code as u8);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

/// Decode bytes as latin-1 text.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// Build a fresh object value.
fn new_object(module: String, name: String, args: Vec<Value>) -> Value {
    Value::Object(Rc::new(RefCell::new(Object {
        module,
        name,
        args,
        state: None,
        entries: Vec::new(),
    })))
}

/// Assign `target[key] = value` for dict-like values.
fn set_item(target: &Value, key: Value, value: Value) -> Result<()> {
    match target {
        Value::Dict(d) => {
            d.try_borrow_mut()
                .map_err(|_| anyhow!("pickle: recursive dict update"))?
                .push((key, value));
            Ok(())
        }
        Value::Object(o) => {
            o.try_borrow_mut()
                .map_err(|_| anyhow!("pickle: recursive object update"))?
                .entries
                .push((key, value));
            Ok(())
        }
        _ => bail!("pickle: SETITEM applied to a value that is not dict-like"),
    }
}

/// Interpret a hexadecimal digit.
fn hex_digit(byte: Option<u8>) -> Result<u8> {
    let b = byte.context("pickle: truncated \\x escape in STRING literal")?;
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => bail!("pickle: invalid hex digit in STRING literal"),
    }
}

/// Decode a protocol-0 quoted string literal into raw bytes.
fn decode_repr_string(text: &[u8]) -> Result<Vec<u8>> {
    let mut trimmed = text;
    while let Some((&first, rest)) = trimmed.split_first() {
        if first.is_ascii_whitespace() {
            trimmed = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = trimmed.split_last() {
        if last.is_ascii_whitespace() {
            trimmed = rest;
        } else {
            break;
        }
    }
    if trimmed.len() < 2 {
        bail!("pickle: malformed STRING literal");
    }
    let quote = trimmed[0];
    if (quote != b'\'' && quote != b'"') || trimmed[trimmed.len() - 1] != quote {
        bail!("pickle: malformed STRING literal");
    }
    let body = &trimmed[1..trimmed.len() - 1];
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        let escape = *body
            .get(i)
            .context("pickle: truncated escape in STRING literal")?;
        i += 1;
        match escape {
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'r' => out.push(b'\r'),
            b'x' => {
                let hi = hex_digit(body.get(i).copied())?;
                let lo = hex_digit(body.get(i + 1).copied())?;
                out.push(hi * 16 + lo);
                i += 2;
            }
            b'0'..=b'7' => {
                let mut value = u32::from(escape - b'0');
                let mut taken = 1;
                while taken < 3 {
                    match body.get(i) {
                        Some(&d) if (b'0'..=b'7').contains(&d) => {
                            value = value * 8 + u32::from(d - b'0');
                            i += 1;
                            taken += 1;
                        }
                        _ => break,
                    }
                }
                out.push((value & 0xff) as u8);
            }
            other => out.push(other),
        }
    }
    Ok(out)
}

/// Decode a protocol-0 `UNICODE` argument (raw-unicode-escape).
fn decode_unicode_escape(text: &[u8]) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i] == b'\\' && i + 1 < text.len() {
            let width = match text[i + 1] {
                b'u' => 4,
                b'U' => 8,
                b'\\' => {
                    out.push('\\');
                    i += 2;
                    continue;
                }
                _ => {
                    out.push(char::from(text[i]));
                    i += 1;
                    continue;
                }
            };
            let mut code: u32 = 0;
            for offset in 0..width {
                code = code * 16 + u32::from(hex_digit(text.get(i + 2 + offset).copied())?);
            }
            let ch =
                char::from_u32(code).context("pickle: invalid code point in UNICODE literal")?;
            out.push(ch);
            i += 2 + width;
        } else {
            out.push(char::from(text[i]));
            i += 1;
        }
    }
    Ok(out)
}

/// Decode a little-endian two's-complement integer of up to 8 bytes.
fn long_from_bytes(bytes: &[u8]) -> Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        bail!("pickle: integer of {} bytes exceeds i64", bytes.len());
    }
    let mut raw: u64 = 0;
    for (i, &b) in bytes.iter().enumerate() {
        raw |= u64::from(b) << (8 * i);
    }
    let bits = 8 * bytes.len();
    if bits < 64 {
        let shift = 64 - bits;
        Ok(((raw << shift) as i64) >> shift)
    } else {
        Ok(raw as i64)
    }
}

/// The pickle stack machine.
struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u64, Value>,
}

impl<'a> Unpickler<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("pickle: offset overflow")?;
        let slice: &'a [u8] = self
            .data
            .get(self.pos..end)
            .with_context(|| format!("pickle: unexpected end of stream at {}", self.pos))?;
        self.pos = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8> {
        let slice = self.take(1)?;
        slice
            .first()
            .copied()
            .context("pickle: unexpected end of stream")
    }

    fn line(&mut self) -> Result<&'a [u8]> {
        let rest: &'a [u8] = self
            .data
            .get(self.pos..)
            .context("pickle: unexpected end of stream")?;
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .context("pickle: unterminated text argument")?;
        self.pos += end + 1;
        Ok(&rest[..end])
    }

    fn read_len(&mut self, width: usize) -> Result<usize> {
        let bytes = self.take(width)?;
        let mut value: u64 = 0;
        for (i, &b) in bytes.iter().enumerate() {
            value |= u64::from(b) << (8 * i);
        }
        usize::try_from(value).context("pickle: length does not fit in memory")
    }

    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle: stack underflow")
    }

    fn top(&self) -> Result<Value> {
        self.stack.last().cloned().context("pickle: empty stack")
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mut mark = None;
        for (i, value) in self.stack.iter().enumerate().rev() {
            if matches!(value, Value::Mark) {
                mark = Some(i);
                break;
            }
        }
        let index = mark.context("pickle: unbalanced MARK")?;
        let items = self.stack.split_off(index + 1);
        self.stack.pop();
        Ok(items)
    }

    fn memo_put(&mut self, key: u64) -> Result<()> {
        let value = self.top()?;
        self.memo.insert(key, value);
        Ok(())
    }

    fn memo_get(&mut self, key: u64) -> Result<()> {
        let value = self
            .memo
            .get(&key)
            .cloned()
            .with_context(|| format!("pickle: memo entry {key} is missing"))?;
        self.push(value);
        Ok(())
    }

    fn pairs_from(items: Vec<Value>) -> Result<Vec<(Value, Value)>> {
        if items.len() % 2 != 0 {
            bail!("pickle: dict opcode with an odd number of values");
        }
        let mut pairs = Vec::with_capacity(items.len() / 2);
        let mut iter = items.into_iter();
        while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
            pairs.push((key, value));
        }
        Ok(pairs)
    }

    fn extend_sequence(target: &Value, items: Vec<Value>) -> Result<()> {
        match target {
            Value::List(l) => {
                l.try_borrow_mut()
                    .map_err(|_| anyhow!("pickle: recursive list update"))?
                    .extend(items);
                Ok(())
            }
            _ => bail!("pickle: APPEND applied to a value that is not a list"),
        }
    }

    /// Apply a `REDUCE`-style call.
    fn reduce(&mut self, callable: Value, argtuple: Value) -> Result<Value> {
        let args: Vec<Value> = match &argtuple {
            Value::Tuple(t) => t.as_ref().clone(),
            Value::None => Vec::new(),
            _ => bail!("pickle: REDUCE expects a tuple of arguments"),
        };
        let (module, name) = match &callable {
            Value::Global(g) => (g.0.clone(), g.1.clone()),
            _ => bail!("pickle: REDUCE callable is not a global reference"),
        };

        // `_codecs.encode(text, 'latin1')` restores Python 2 byte strings.
        if (module == "_codecs" || module == "codecs") && name == "encode" {
            let text = args
                .first()
                .context("pickle: _codecs.encode without an argument")?;
            let bytes = text
                .as_raw_bytes()
                .context("pickle: _codecs.encode applied to a non-text value")?;
            return Ok(Value::Bytes(Rc::new(bytes)));
        }

        // `copyreg._reconstructor(cls, base, state)` builds a bare instance.
        if (module == "copyreg" || module == "copy_reg") && name == "_reconstructor" {
            let mut iter = args.into_iter();
            let class = iter
                .next()
                .context("pickle: copyreg._reconstructor without a class")?;
            let (cls_module, cls_name) = match &class {
                Value::Global(g) => (g.0.clone(), g.1.clone()),
                _ => bail!("pickle: copyreg._reconstructor without a class reference"),
            };
            let rest: Vec<Value> = iter.collect();
            return Ok(new_object(cls_module, cls_name, rest));
        }

        Ok(new_object(module, name, args))
    }

    /// Execute the stream until `STOP`.
    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.byte()?;
            match op {
                // --- Control ---
                b'(' => self.push(Value::Mark),
                b'.' => return self.pop(),
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_mark()?;
                }
                b'2' => {
                    let top = self.top()?;
                    self.push(top);
                }
                b'\x80' => {
                    let proto = self.byte()?;
                    if proto > 5 {
                        bail!("pickle: unsupported protocol version {proto}");
                    }
                }
                b'\x95' => {
                    self.take(8)?;
                }

                // --- Scalars ---
                b'N' => self.push(Value::None),
                b'\x88' => self.push(Value::Bool(true)),
                b'\x89' => self.push(Value::Bool(false)),
                b'I' => {
                    let text = latin1(self.line()?);
                    let trimmed = text.trim();
                    let value = match trimmed {
                        "01" => Value::Bool(true),
                        "00" => Value::Bool(false),
                        other => Value::Int(
                            other
                                .parse::<i64>()
                                .with_context(|| format!("pickle: bad INT '{other}'"))?,
                        ),
                    };
                    self.push(value);
                }
                b'J' => {
                    let bytes = self.take(4)?;
                    let value = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    self.push(Value::Int(i64::from(value)));
                }
                b'K' => {
                    let value = self.byte()?;
                    self.push(Value::Int(i64::from(value)));
                }
                b'M' => {
                    let bytes = self.take(2)?;
                    let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                    self.push(Value::Int(i64::from(value)));
                }
                b'L' => {
                    let text = latin1(self.line()?);
                    let trimmed = text.trim().trim_end_matches('L');
                    self.push(Value::Int(
                        trimmed
                            .parse::<i64>()
                            .with_context(|| format!("pickle: bad LONG '{trimmed}'"))?,
                    ));
                }
                b'\x8a' | b'\x8b' => {
                    let width = if op == b'\x8a' { 1 } else { 4 };
                    let len = self.read_len(width)?;
                    let bytes = self.take(len)?;
                    self.push(Value::Int(long_from_bytes(bytes)?));
                }
                b'F' => {
                    let text = latin1(self.line()?);
                    let trimmed = text.trim();
                    self.push(Value::Float(
                        trimmed
                            .parse::<f64>()
                            .with_context(|| format!("pickle: bad FLOAT '{trimmed}'"))?,
                    ));
                }
                b'G' => {
                    let bytes = self.take(8)?;
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(bytes);
                    self.push(Value::Float(f64::from_be_bytes(buf)));
                }

                // --- Strings and bytes ---
                b'S' => {
                    let raw = decode_repr_string(self.line()?)?;
                    self.push(Value::Bytes(Rc::new(raw)));
                }
                b'T' | b'U' => {
                    let width = if op == b'T' { 4 } else { 1 };
                    let len = self.read_len(width)?;
                    let bytes = self.take(len)?.to_vec();
                    self.push(Value::Bytes(Rc::new(bytes)));
                }
                b'B' | b'C' | b'\x8e' | b'\x96' => {
                    let width = match op {
                        b'B' => 4,
                        b'C' => 1,
                        _ => 8,
                    };
                    let len = self.read_len(width)?;
                    let bytes = self.take(len)?.to_vec();
                    self.push(Value::Bytes(Rc::new(bytes)));
                }
                b'V' => {
                    let text = decode_unicode_escape(self.line()?)?;
                    self.push(Value::Str(Rc::new(text)));
                }
                b'X' | b'\x8c' | b'\x8d' => {
                    let width = match op {
                        b'X' => 4,
                        b'\x8c' => 1,
                        _ => 8,
                    };
                    let len = self.read_len(width)?;
                    let bytes = self.take(len)?;
                    let text = std::str::from_utf8(bytes)
                        .context("pickle: invalid UTF-8 in a unicode literal")?;
                    self.push(Value::Str(Rc::new(text.to_string())));
                }

                // --- Containers ---
                b')' => self.push(Value::Tuple(Rc::new(Vec::new()))),
                b't' => {
                    let items = self.pop_mark()?;
                    self.push(Value::Tuple(Rc::new(items)));
                }
                b'\x85' | b'\x86' | b'\x87' => {
                    let arity = match op {
                        b'\x85' => 1,
                        b'\x86' => 2,
                        _ => 3,
                    };
                    if self.stack.len() < arity {
                        bail!("pickle: stack underflow in TUPLE{arity}");
                    }
                    let items = self.stack.split_off(self.stack.len() - arity);
                    self.push(Value::Tuple(Rc::new(items)));
                }
                b']' | b'\x8f' => self.push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b'l' | b'\x91' => {
                    let items = self.pop_mark()?;
                    self.push(Value::List(Rc::new(RefCell::new(items))));
                }
                b'a' => {
                    let item = self.pop()?;
                    let target = self.top()?;
                    Self::extend_sequence(&target, vec![item])?;
                }
                b'e' | b'\x90' => {
                    let items = self.pop_mark()?;
                    let target = self.top()?;
                    Self::extend_sequence(&target, items)?;
                }
                b'}' => self.push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = Self::pairs_from(items)?;
                    self.push(Value::Dict(Rc::new(RefCell::new(pairs))));
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let target = self.top()?;
                    set_item(&target, key, value)?;
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let pairs = Self::pairs_from(items)?;
                    let target = self.top()?;
                    for (key, value) in pairs {
                        set_item(&target, key, value)?;
                    }
                }

                // --- Globals and instances ---
                b'c' => {
                    let module = latin1(self.line()?);
                    let name = latin1(self.line()?);
                    self.push(Value::Global(Rc::new((module, name))));
                }
                b'\x93' => {
                    let name = self
                        .pop()?
                        .as_text()
                        .context("pickle: STACK_GLOBAL without a name")?;
                    let module = self
                        .pop()?
                        .as_text()
                        .context("pickle: STACK_GLOBAL without a module")?;
                    self.push(Value::Global(Rc::new((module, name))));
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let value = self.reduce(callable, args)?;
                    self.push(value);
                }
                b'\x81' | b'\x92' => {
                    if op == b'\x92' {
                        self.pop()?; // keyword arguments are not used by numpy
                    }
                    let args = self.pop()?;
                    let class = self.pop()?;
                    let value = self.reduce(class, args)?;
                    self.push(value);
                }
                b'o' => {
                    let mut items = self.pop_mark()?.into_iter();
                    let class = items.next().context("pickle: OBJ without a class")?;
                    let args: Vec<Value> = items.collect();
                    let value = self.reduce(class, Value::Tuple(Rc::new(args)))?;
                    self.push(value);
                }
                b'i' => {
                    let module = latin1(self.line()?);
                    let name = latin1(self.line()?);
                    let args = self.pop_mark()?;
                    self.push(new_object(module, name, args));
                }
                b'b' => {
                    let state = self.pop()?;
                    let target = self.top()?;
                    match &target {
                        Value::Object(o) => {
                            o.try_borrow_mut()
                                .map_err(|_| anyhow!("pickle: recursive BUILD"))?
                                .state = Some(state);
                        }
                        _ => bail!("pickle: BUILD applied to a value that is not an object"),
                    }
                }

                // --- Memo ---
                b'p' => {
                    let text = latin1(self.line()?);
                    let key: u64 = text
                        .trim()
                        .parse()
                        .with_context(|| format!("pickle: bad PUT index '{text}'"))?;
                    self.memo_put(key)?;
                }
                b'q' | b'r' => {
                    let width = if op == b'q' { 1 } else { 4 };
                    let key = self.read_len(width)? as u64;
                    self.memo_put(key)?;
                }
                b'\x94' => {
                    let key = self.memo.len() as u64;
                    self.memo_put(key)?;
                }
                b'g' => {
                    let text = latin1(self.line()?);
                    let key: u64 = text
                        .trim()
                        .parse()
                        .with_context(|| format!("pickle: bad GET index '{text}'"))?;
                    self.memo_get(key)?;
                }
                b'h' | b'j' => {
                    let width = if op == b'h' { 1 } else { 4 };
                    let key = self.read_len(width)? as u64;
                    self.memo_get(key)?;
                }

                // --- Deliberately unsupported ---
                b'P' | b'Q' => {
                    bail!("pickle: persistent ids (opcode 0x{op:02x}) require a Python interpreter")
                }
                b'\x82' | b'\x83' | b'\x84' => {
                    bail!("pickle: extension registry opcodes (0x{op:02x}) are not supported")
                }
                b'\x97' | b'\x98' => {
                    bail!("pickle: out-of-band buffers (opcode 0x{op:02x}) are not supported")
                }
                other => bail!(
                    "pickle: unsupported opcode 0x{other:02x} at offset {}",
                    self.pos - 1
                ),
            }
        }
    }
}

/// Look up a dict entry by its (latin-1 decoded) key.
fn lookup(entries: &[(Value, Value)], key: &str) -> Option<Value> {
    entries
        .iter()
        .find(|(k, _)| k.as_text().as_deref() == Some(key))
        .map(|(_, v)| v.clone())
}

/// Extract dict entries from a `__setstate__` payload.
fn dict_entries_of(value: &Value) -> Option<Vec<(Value, Value)>> {
    match value {
        Value::Dict(d) => d
            .try_borrow()
            .ok()
            .map(|entries| entries.iter().cloned().collect()),
        // `object.__reduce_ex__` may produce `(instance_dict, slots_dict)`.
        Value::Tuple(t) => t.iter().find_map(dict_entries_of),
        _ => None,
    }
}

/// Read a shape tuple.
fn shape_from_value(value: &Value) -> Result<Vec<usize>> {
    let items: Vec<Value> = match value {
        Value::Tuple(t) => t.as_ref().clone(),
        Value::List(l) => l
            .try_borrow()
            .map_err(|_| anyhow!("pickle: shape list is already borrowed"))?
            .iter()
            .cloned()
            .collect(),
        Value::Int(n) => vec![Value::Int(*n)],
        _ => bail!("array shape is not a tuple"),
    };
    let mut shape = Vec::with_capacity(items.len());
    for dim in items {
        match dim {
            Value::Int(n) if n >= 0 => shape.push(n as usize),
            _ => bail!("array shape contains a non-integer dimension"),
        }
    }
    Ok(shape)
}

/// Normalise a numpy dtype descriptor to an explicit byte order.
fn normalise_descr(descr: &str) -> Result<String> {
    let mut chars = descr.chars();
    let first = chars.next().context("empty numpy dtype descriptor")?;
    let native = if cfg!(target_endian = "big") {
        '>'
    } else {
        '<'
    };
    let (order, base) = if matches!(first, '<' | '>' | '|' | '=') {
        let base: String = chars.collect();
        let order = if first == '=' { native } else { first };
        (order, base)
    } else {
        (native, descr.to_string())
    };
    if base.is_empty() {
        bail!("numpy dtype '{descr}' has no type code");
    }
    let size = npy::item_size(&format!("|{base}"))?;
    let order = if size == 1 { '|' } else { order };
    Ok(format!("{order}{base}"))
}

/// Resolve a `numpy.dtype` value into a descriptor string.
fn descr_from_value(value: &Value) -> Result<String> {
    if let Some(text) = value.as_text() {
        return normalise_descr(&text);
    }
    let Value::Object(obj) = value else {
        bail!("numpy dtype is neither a string nor an object");
    };
    let obj = obj
        .try_borrow()
        .map_err(|_| anyhow!("pickle: dtype object is already borrowed"))?;
    if obj.name != "dtype" {
        bail!("unexpected numpy dtype class {}.{}", obj.module, obj.name);
    }
    let base = obj
        .args
        .first()
        .and_then(Value::as_text)
        .context("numpy dtype has no type-code argument")?;
    // __setstate__ = (version, byteorder, subarray, names, fields, ...)
    let order = obj.state.as_ref().and_then(|state| match state {
        Value::Tuple(t) => t.get(1).and_then(Value::as_text),
        _ => None,
    });
    let descr = match order.and_then(|o| o.chars().next()) {
        Some(order) => {
            let stripped = base.trim_start_matches(|c| matches!(c, '<' | '>' | '|' | '='));
            format!("{order}{stripped}")
        }
        None => base,
    };
    normalise_descr(&descr)
}

/// Rebuild a `numpy.ndarray` from its `__setstate__` payload.
fn ndarray_from_object(obj: &Object) -> Result<NpyArray> {
    let state = obj
        .state
        .as_ref()
        .context("numpy array pickle has no __setstate__ payload")?;
    let items: Vec<Value> = match state {
        Value::Tuple(t) => t.as_ref().clone(),
        Value::List(l) => l
            .try_borrow()
            .map_err(|_| anyhow!("pickle: array state is already borrowed"))?
            .iter()
            .cloned()
            .collect(),
        _ => bail!("numpy array state is not a tuple"),
    };
    // (version, shape, dtype, is_fortran, data); very old pickles omit the version.
    let fields = match items.len() {
        5 => &items[1..5],
        4 => &items[0..4],
        other => bail!("unexpected numpy array state arity {other}"),
    };
    let shape = shape_from_value(&fields[0])?;
    let descr = descr_from_value(&fields[1])?;
    let fortran = matches!(fields[2], Value::Bool(true) | Value::Int(1));
    let raw = fields[3]
        .as_raw_bytes()
        .context("numpy array payload is not a byte string (object dtypes are unsupported)")?;
    let item = npy::item_size(&descr)?;
    let count: usize = shape.iter().product();
    let expected = count
        .checked_mul(item)
        .context("numpy array size overflows")?;
    if raw.len() != expected {
        bail!(
            "numpy array payload is {} bytes but shape {shape:?} with dtype '{descr}' needs {expected}",
            raw.len()
        );
    }
    let data = if fortran {
        npy::fortran_to_c(&raw, &shape, item)
    } else {
        raw
    };
    Ok(NpyArray { descr, shape, data })
}

/// Densify a compressed sparse matrix into a row-major float64 array.
pub fn densify(
    rows: usize,
    cols: usize,
    values: &[f64],
    indices: &[i64],
    indptr: &[i64],
    column_major: bool,
) -> Result<NpyArray> {
    let cells = rows
        .checked_mul(cols)
        .context("sparse matrix is too large to densify")?;
    let mut dense = vec![0.0f64; cells];
    let outer = if column_major { cols } else { rows };
    if indptr.len() < outer + 1 {
        bail!(
            "sparse indptr has {} entries, expected {}",
            indptr.len(),
            outer + 1
        );
    }
    for o in 0..outer {
        let start = usize::try_from(indptr[o])
            .map_err(|_| anyhow!("sparse indptr contains a negative offset"))?;
        let end = usize::try_from(indptr[o + 1])
            .map_err(|_| anyhow!("sparse indptr contains a negative offset"))?;
        if start > end || end > indices.len() || end > values.len() {
            bail!("sparse indptr range {start}..{end} is out of bounds");
        }
        for k in start..end {
            let inner = usize::try_from(indices[k])
                .map_err(|_| anyhow!("sparse matrix contains a negative index"))?;
            let (row, col) = if column_major { (inner, o) } else { (o, inner) };
            if row >= rows || col >= cols {
                bail!("sparse index ({row}, {col}) is outside a {rows}x{cols} matrix");
            }
            dense[row * cols + col] += values[k];
        }
    }
    npy::from_f64(vec![rows, cols], &dense)
}

/// Read a Python `list`/`tuple` of numbers as `f64` values.
fn number_sequence(value: &Value) -> Option<Vec<f64>> {
    let items: Vec<Value> = match value {
        Value::List(l) => l.try_borrow().ok()?.iter().cloned().collect(),
        Value::Tuple(t) => t.as_ref().clone(),
        _ => return None,
    };
    items.iter().map(Value::as_number).collect()
}

/// Read a numeric payload that may be an array or a plain Python sequence.
fn float_values(value: &Value, what: &str) -> Result<Vec<f64>> {
    if let Some(array) = value_to_array(value)? {
        return npy::to_f64_vec(&array);
    }
    number_sequence(value)
        .with_context(|| format!("sparse '{what}' is neither an array nor a sequence of numbers"))
}

/// Read an integer payload that may be an array or a plain Python sequence.
fn integer_values(value: &Value, what: &str) -> Result<Vec<i64>> {
    if let Some(array) = value_to_array(value)? {
        return npy::to_i64_vec(&array);
    }
    let numbers = number_sequence(value).with_context(|| {
        format!("sparse '{what}' is neither an array nor a sequence of numbers")
    })?;
    Ok(numbers.iter().map(|value| *value as i64).collect())
}

/// Rebuild a SciPy compressed sparse matrix from its instance dict.
fn sparse_from_entries(module: &str, name: &str, entries: &[(Value, Value)]) -> Result<NpyArray> {
    let data_value = lookup(entries, "data").context("sparse matrix has no 'data' array")?;
    let indices_value =
        lookup(entries, "indices").context("sparse matrix has no 'indices' array")?;
    let indptr_value = lookup(entries, "indptr").context("sparse matrix has no 'indptr' array")?;
    let shape_value = lookup(entries, "_shape")
        .or_else(|| lookup(entries, "shape"))
        .context("sparse matrix has no shape")?;

    let shape = shape_from_value(&shape_value)?;
    if shape.len() != 2 {
        bail!("sparse matrix shape {shape:?} is not two-dimensional");
    }
    let (rows, cols) = (shape[0], shape[1]);

    let values = float_values(&data_value, "data")?;
    let idx = integer_values(&indices_value, "indices")?;
    let ptr = integer_values(&indptr_value, "indptr")?;

    // CSC stores one indptr entry per column, CSR one per row. Deciding this
    // wrongly silently transposes the matrix, so the class name and the
    // indptr length must agree.
    let lower = name.to_ascii_lowercase();
    let column_major = if lower.contains("csc") {
        if ptr.len() != cols + 1 {
            bail!(
                "{module}.{name}: indptr has {} entries, expected {} columns + 1",
                ptr.len(),
                cols
            );
        }
        true
    } else if lower.contains("csr") {
        if ptr.len() != rows + 1 {
            bail!(
                "{module}.{name}: indptr has {} entries, expected {} rows + 1",
                ptr.len(),
                rows
            );
        }
        false
    } else if ptr.len() == cols + 1 && rows != cols {
        true
    } else if ptr.len() == rows + 1 {
        false
    } else {
        bail!(
            "unsupported sparse format {module}.{name} (indptr has {} entries for a {rows}x{cols} matrix)",
            ptr.len()
        );
    };

    densify(rows, cols, &values, &idx, &ptr, column_major)
}

/// Convert a decoded pickle value into an array, if it holds one.
fn value_to_array(value: &Value) -> Result<Option<NpyArray>> {
    let Value::Object(cell) = value else {
        return Ok(None);
    };
    let obj = cell
        .try_borrow()
        .map_err(|_| anyhow!("pickle: object is already borrowed"))?;

    if obj.module.contains("numpy") && (obj.name == "_reconstruct" || obj.name == "ndarray") {
        return ndarray_from_object(&obj).map(Some);
    }

    let state_entries = obj.state.as_ref().and_then(dict_entries_of);
    if obj.module.starts_with("scipy.sparse") {
        let entries =
            state_entries.context("scipy sparse matrix pickle has no instance dictionary")?;
        return sparse_from_entries(&obj.module, &obj.name, &entries).map(Some);
    }

    if let Some(entries) = state_entries {
        // chumpy `Ch` (and similar thin wrappers) keep the payload in `x`.
        if let Some(inner) = lookup(&entries, "x") {
            if let Some(array) = value_to_array(&inner)? {
                return Ok(Some(array));
            }
        }
        // An unlabelled compressed sparse matrix.
        if lookup(&entries, "indptr").is_some() {
            return sparse_from_entries(&obj.module, &obj.name, &entries).map(Some);
        }
        let keys: Vec<String> = entries.iter().filter_map(|(k, _)| k.as_text()).collect();
        bail!(
            "unsupported pickled class {}.{} (state keys: [{}])",
            obj.module,
            obj.name,
            keys.join(", ")
        );
    }

    bail!(
        "unsupported pickled class {}.{} (no instance dictionary)",
        obj.module,
        obj.name
    )
}

/// Return the entries of the top-level mapping.
fn root_entries(root: &Value) -> Result<Vec<(Value, Value)>> {
    match root {
        Value::Dict(d) => {
            let entries = d
                .try_borrow()
                .map_err(|_| anyhow!("pickle: top-level dict is already borrowed"))?;
            Ok(entries.iter().cloned().collect())
        }
        Value::Object(o) => {
            let obj = o
                .try_borrow()
                .map_err(|_| anyhow!("pickle: top-level object is already borrowed"))?;
            if !obj.entries.is_empty() {
                return Ok(obj.entries.clone());
            }
            if let Some(entries) = obj.state.as_ref().and_then(dict_entries_of) {
                return Ok(entries);
            }
            bail!(
                "pickle: top-level object {}.{} is not a mapping",
                obj.module,
                obj.name
            )
        }
        _ => bail!("pickle: expected a top-level dict of FLAME components"),
    }
}

/// Decode a pickled Python `dict` of arrays into named [`NpyArray`] values.
pub fn load_arrays(data: &[u8]) -> Result<HashMap<String, NpyArray>> {
    if data.len() < 2 {
        bail!("pickle file is too small to contain a model");
    }
    let mut machine = Unpickler::new(data);
    let root = machine.run()?;
    let entries = root_entries(&root)?;

    let mut arrays = HashMap::new();
    for (key, value) in entries {
        let Some(name) = key.as_text() else {
            continue;
        };
        match value_to_array(&value) {
            Ok(Some(array)) => {
                arrays.insert(name, array);
            }
            Ok(None) => tracing::debug!("Skipping non-array pickle entry '{name}'"),
            Err(err) => {
                if super::REQUIRED_COMPONENTS.contains(&name.as_str()) {
                    return Err(err).with_context(|| {
                        format!("Failed to decode required FLAME component '{name}'")
                    });
                }
                tracing::debug!("Skipping pickle entry '{name}': {err:#}");
            }
        }
    }

    if arrays.is_empty() {
        bail!("pickle file contains no numeric arrays");
    }
    Ok(arrays)
}
