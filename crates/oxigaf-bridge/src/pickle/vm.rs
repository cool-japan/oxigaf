//! A non-executing Python pickle unpickler (protocols 0-5).
//!
//! "Non-executing" is the whole point: every opcode that would normally call
//! into Python (`REDUCE`, `NEWOBJ`, `INST`, `BUILD`, …) is recorded
//! structurally as a [`Value::Reduce`] / [`Value::Object`] instead of being
//! dispatched. Nothing this module does depends on the contents of the
//! pickle beyond parsing it, so a hostile `.pt` cannot make it execute
//! anything; the worst it can do is make it allocate, which the depth and
//! stack limits below bound.
//!
//! Coverage is the full opcode set through protocol 5, minus the two
//! framing-only opcodes that carry no value (`FRAME` is honoured by
//! skipping, `NEXT_BUFFER`/`READONLY_BUFFER` are out-of-band buffers, which
//! `torch.save` does not emit).

use super::error::{PickleError, Result};
use super::value::Value;
use std::collections::HashMap;

/// Maximum nesting depth of `MARK`-delimited constructs.
///
/// A pickle is a flat instruction stream, so this bounds recursion in the
/// *value tree*, not in this parser (which is iterative). Real checkpoints
/// nest a handful of levels; the limit exists so a crafted file cannot
/// build an arbitrarily deep tree that later overflows the stack when it is
/// dropped or walked.
const MAX_MARK_DEPTH: usize = 256;

/// Maximum number of values live on the unpickler stack at once.
const MAX_STACK: usize = 1 << 22;

/// Resolves a `PERSID` / `BINPERSID` identifier to a value.
///
/// A pickle by itself cannot express "the bytes of this tensor's storage";
/// PyTorch stores those as separate members of the surrounding ZIP and
/// refers to them by persistent id. [`super::torch`] implements this against
/// the ZIP container.
pub trait PersistentIdResolver {
    /// Resolve one persistent id.
    ///
    /// # Errors
    ///
    /// Returns an error if the identifier does not name a resource this
    /// resolver can supply.
    fn resolve(&mut self, id: &Value) -> Result<Value>;
}

/// A resolver that records the identifier verbatim, for pickles that carry
/// no external resources (a bare `.pkl`, e.g. a FLAME model).
pub struct OpaquePersistentIds;

impl PersistentIdResolver for OpaquePersistentIds {
    fn resolve(&mut self, id: &Value) -> Result<Value> {
        Ok(Value::PersistentId(Box::new(id.clone())))
    }
}

/// Unpickle a complete pickle stream, resolving persistent ids through
/// `resolver`.
///
/// # Errors
///
/// Returns [`PickleError`] for a truncated stream, an unknown or
/// out-of-protocol opcode, a stack underflow, or a limit violation.
pub fn load_with<R: PersistentIdResolver>(data: &[u8], resolver: &mut R) -> Result<Value> {
    Unpickler::new(data, resolver).run()
}

/// Unpickle a stream that refers to no external resources.
///
/// # Errors
///
/// As [`load_with`].
pub fn load(data: &[u8]) -> Result<Value> {
    load_with(data, &mut OpaquePersistentIds)
}

struct Unpickler<'a, R: PersistentIdResolver> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
    /// Next key `MEMOIZE` will use.
    ///
    /// `MEMOIZE` stores at "the next available index", which CPython tracks
    /// as a monotonically increasing counter -- *not* as the memo's current
    /// size. Deriving it from `memo.len()` would repeat a key the moment a
    /// stream mixes `MEMOIZE` with an explicit `BINPUT` (protocol 4 files
    /// written by a mix of picklers do), silently overwriting an entry a
    /// later `BINGET` still needs.
    next_memo_key: u32,
    resolver: &'a mut R,
}

impl<'a, R: PersistentIdResolver> Unpickler<'a, R> {
    fn new(data: &'a [u8], resolver: &'a mut R) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
            next_memo_key: 0,
            resolver,
        }
    }

    // --- primitive readers -------------------------------------------------

    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .data
            .get(self.pos)
            .ok_or(PickleError::Truncated { offset: self.pos })?;
        self.pos += 1;
        Ok(byte)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(PickleError::Truncated { offset: self.pos })?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or(PickleError::Truncated { offset: self.pos })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let b = self.read_exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let b = self.read_exact(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(self.read_i32()? as u32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let b = self.read_exact(8)?;
        let mut array = [0u8; 8];
        array.copy_from_slice(b);
        Ok(u64::from_le_bytes(array))
    }

    /// Reads a length prefix and converts it to a `usize`, rejecting a value
    /// that exceeds the remaining input. Without that check a crafted
    /// 8-byte length would make the reader try to allocate exabytes before
    /// discovering the stream is truncated.
    fn read_len(&mut self, raw: u64) -> Result<usize> {
        let len = usize::try_from(raw).map_err(|_| PickleError::Truncated { offset: self.pos })?;
        if len > self.data.len().saturating_sub(self.pos) {
            return Err(PickleError::Truncated { offset: self.pos });
        }
        Ok(len)
    }

    /// Reads up to and including the next `\n`, returning the line without
    /// its terminator. Used by the protocol-0 text opcodes.
    fn read_line(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        let rel = self.data[start..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or(PickleError::Truncated { offset: start })?;
        self.pos = start + rel + 1;
        let mut line = &self.data[start..start + rel];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        Ok(line)
    }

    fn read_line_str(&mut self) -> Result<String> {
        let line = self.read_line()?;
        Ok(String::from_utf8_lossy(line).into_owned())
    }

    // --- stack helpers -----------------------------------------------------

    fn push(&mut self, value: Value) -> Result<()> {
        if self.stack.len() >= MAX_STACK {
            return Err(PickleError::LimitExceeded {
                what: "unpickler stack depth",
                limit: MAX_STACK,
            });
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or(PickleError::StackUnderflow { offset: self.pos })
    }

    fn peek_mut(&mut self) -> Result<&mut Value> {
        self.stack
            .last_mut()
            .ok_or(PickleError::StackUnderflow { offset: self.pos })
    }

    fn mark(&mut self) -> Result<()> {
        if self.marks.len() >= MAX_MARK_DEPTH {
            return Err(PickleError::LimitExceeded {
                what: "MARK nesting depth",
                limit: MAX_MARK_DEPTH,
            });
        }
        self.marks.push(self.stack.len());
        Ok(())
    }

    /// Pops everything above the most recent `MARK`, consuming the mark.
    fn pop_to_mark(&mut self) -> Result<Vec<Value>> {
        let at = self
            .marks
            .pop()
            .ok_or(PickleError::StackUnderflow { offset: self.pos })?;
        if at > self.stack.len() {
            return Err(PickleError::StackUnderflow { offset: self.pos });
        }
        Ok(self.stack.split_off(at))
    }

    fn memo_put(&mut self, key: u32) -> Result<()> {
        let value = self
            .stack
            .last()
            .ok_or(PickleError::StackUnderflow { offset: self.pos })?
            .clone();
        self.memo.insert(key, value);
        // An explicit PUT at or beyond the counter advances it, so a later
        // MEMOIZE cannot land on a key a PUT already claimed.
        self.next_memo_key = self.next_memo_key.max(key.saturating_add(1));
        Ok(())
    }

    fn memo_get(&mut self, key: u32) -> Result<()> {
        let value = self
            .memo
            .get(&key)
            .ok_or(PickleError::MemoMiss { key })?
            .clone();
        self.push(value)
    }

    // --- main loop ---------------------------------------------------------

    fn run(mut self) -> Result<Value> {
        loop {
            let offset = self.pos;
            let opcode = self.read_u8()?;
            if self.step(opcode, offset)? {
                break;
            }
        }
        let result = self.pop()?;
        if !self.stack.is_empty() {
            return Err(PickleError::UnbalancedStack {
                remaining: self.stack.len(),
            });
        }
        Ok(result)
    }

    /// Executes one opcode. Returns `Ok(true)` when `STOP` was reached.
    fn step(&mut self, opcode: u8, offset: usize) -> Result<bool> {
        match opcode {
            // --- protocol / framing ---
            // PROTO
            0x80 => {
                let version = self.read_u8()?;
                if version > 5 {
                    return Err(PickleError::UnsupportedProtocol { version });
                }
            }
            // FRAME: a length-prefixed hint only; the payload is ordinary
            // opcodes, so nothing needs to be skipped.
            0x95 => {
                self.read_u64()?;
            }
            // STOP
            b'.' => return Ok(true),

            // --- constants ---
            b'N' => self.push(Value::None)?,
            0x88 => self.push(Value::Bool(true))?,
            0x89 => self.push(Value::Bool(false))?,

            // --- integers ---
            // INT (protocol 0): also carries the textual booleans "01"/"00".
            b'I' => {
                let line = self.read_line_str()?;
                let value = match line.as_str() {
                    "01" => Value::Bool(true),
                    "00" => Value::Bool(false),
                    text => Value::Int(text.parse::<i64>().map_err(|_| {
                        PickleError::MalformedLiteral {
                            opcode: "INT",
                            offset,
                        }
                    })?),
                };
                self.push(value)?;
            }
            // BININT
            b'J' => {
                let value = self.read_i32()?;
                self.push(Value::Int(i64::from(value)))?;
            }
            // BININT1
            b'K' => {
                let value = self.read_u8()?;
                self.push(Value::Int(i64::from(value)))?;
            }
            // BININT2
            b'M' => {
                let value = self.read_u16()?;
                self.push(Value::Int(i64::from(value)))?;
            }
            // LONG (protocol 0): a decimal literal with a trailing 'L'.
            b'L' => {
                let mut line = self.read_line_str()?;
                if line.ends_with('L') {
                    line.pop();
                }
                let value = line
                    .parse::<i64>()
                    .map_err(|_| PickleError::MalformedLiteral {
                        opcode: "LONG",
                        offset,
                    })?;
                self.push(Value::Int(value))?;
            }
            // LONG1 / LONG4
            0x8a | 0x8b => {
                let raw_len = if opcode == 0x8a {
                    u64::from(self.read_u8()?)
                } else {
                    u64::from(self.read_u32()?)
                };
                let len = self.read_len(raw_len)?;
                let bytes = self.read_exact(len)?;
                self.push(decode_signed_le(bytes))?;
            }

            // --- floats ---
            // FLOAT (protocol 0)
            b'F' => {
                let line = self.read_line_str()?;
                let value = line
                    .parse::<f64>()
                    .map_err(|_| PickleError::MalformedLiteral {
                        opcode: "FLOAT",
                        offset,
                    })?;
                self.push(Value::Float(value))?;
            }
            // BINFLOAT: big-endian, unlike every other binary opcode.
            b'G' => {
                let b = self.read_exact(8)?;
                let mut array = [0u8; 8];
                array.copy_from_slice(b);
                self.push(Value::Float(f64::from_be_bytes(array)))?;
            }

            // --- strings and bytes ---
            // STRING (protocol 0): a repr-quoted Python 2 str.
            b'S' => {
                let line = self.read_line()?;
                self.push(Value::Bytes(decode_repr_string(line, offset)?))?;
            }
            // BINSTRING / SHORT_BINSTRING (Python 2 str)
            b'T' | b'U' => {
                let raw_len = if opcode == b'T' {
                    u64::from(self.read_u32()?)
                } else {
                    u64::from(self.read_u8()?)
                };
                let len = self.read_len(raw_len)?;
                let bytes = self.read_exact(len)?.to_vec();
                self.push(Value::Bytes(bytes))?;
            }
            // BINBYTES / SHORT_BINBYTES / BINBYTES8
            b'B' | b'C' | 0x8e => {
                let raw_len = match opcode {
                    b'B' => u64::from(self.read_u32()?),
                    b'C' => u64::from(self.read_u8()?),
                    _ => self.read_u64()?,
                };
                let len = self.read_len(raw_len)?;
                let bytes = self.read_exact(len)?.to_vec();
                self.push(Value::Bytes(bytes))?;
            }
            // BYTEARRAY8
            0x96 => {
                let raw_len = self.read_u64()?;
                let len = self.read_len(raw_len)?;
                let bytes = self.read_exact(len)?.to_vec();
                self.push(Value::Bytes(bytes))?;
            }
            // UNICODE (protocol 0): raw-unicode-escape encoded.
            b'V' => {
                let line = self.read_line()?;
                self.push(Value::Str(decode_raw_unicode_escape(line)))?;
            }
            // BINUNICODE / SHORT_BINUNICODE / BINUNICODE8
            b'X' | 0x8c | 0x8d => {
                let raw_len = match opcode {
                    b'X' => u64::from(self.read_u32()?),
                    0x8c => u64::from(self.read_u8()?),
                    _ => self.read_u64()?,
                };
                let len = self.read_len(raw_len)?;
                let bytes = self.read_exact(len)?;
                let text = std::str::from_utf8(bytes)
                    .map_err(|source| PickleError::InvalidUtf8 { offset, source })?;
                self.push(Value::Str(text.to_string()))?;
            }

            // --- tuples ---
            b')' => self.push(Value::Tuple(Vec::new()))?,
            0x85..=0x87 => {
                let arity = (opcode - 0x84) as usize;
                let at = self
                    .stack
                    .len()
                    .checked_sub(arity)
                    .ok_or(PickleError::StackUnderflow { offset })?;
                let items = self.stack.split_off(at);
                self.push(Value::Tuple(items))?;
            }
            // TUPLE
            b't' => {
                let items = self.pop_to_mark()?;
                self.push(Value::Tuple(items))?;
            }

            // --- lists ---
            b']' => self.push(Value::List(Vec::new()))?,
            // LIST
            b'l' => {
                let items = self.pop_to_mark()?;
                self.push(Value::List(items))?;
            }
            // APPEND
            b'a' => {
                let item = self.pop()?;
                append_items(self.peek_mut()?, vec![item], offset)?;
            }
            // APPENDS
            b'e' => {
                let items = self.pop_to_mark()?;
                append_items(self.peek_mut()?, items, offset)?;
            }

            // --- dicts ---
            b'}' => self.push(Value::Dict(Vec::new()))?,
            // DICT
            b'd' => {
                let flat = self.pop_to_mark()?;
                self.push(Value::Dict(pair_up(flat, offset)?))?;
            }
            // SETITEM
            b's' => {
                let value = self.pop()?;
                let key = self.pop()?;
                set_items(self.peek_mut()?, vec![(key, value)], offset)?;
            }
            // SETITEMS
            b'u' => {
                let flat = self.pop_to_mark()?;
                let pairs = pair_up(flat, offset)?;
                set_items(self.peek_mut()?, pairs, offset)?;
            }

            // --- sets ---
            // EMPTY_SET
            0x8f => self.push(Value::Set(Vec::new()))?,
            // FROZENSET
            0x91 => {
                let items = self.pop_to_mark()?;
                self.push(Value::Set(items))?;
            }
            // ADDITEMS
            0x90 => {
                let items = self.pop_to_mark()?;
                match self.peek_mut()? {
                    Value::Set(existing) => existing.extend(items),
                    other => {
                        return Err(PickleError::WrongTarget {
                            opcode: "ADDITEMS",
                            found: other.type_name(),
                            offset,
                        })
                    }
                }
            }

            // --- memo ---
            // PUT (protocol 0)
            b'p' => {
                let line = self.read_line_str()?;
                let key =
                    line.trim()
                        .parse::<u32>()
                        .map_err(|_| PickleError::MalformedLiteral {
                            opcode: "PUT",
                            offset,
                        })?;
                self.memo_put(key)?;
            }
            // BINPUT
            b'q' => {
                let key = self.read_u8()?;
                self.memo_put(u32::from(key))?;
            }
            // LONG_BINPUT
            b'r' => {
                let key = self.read_u32()?;
                self.memo_put(key)?;
            }
            // MEMOIZE: stores at the next counter value, which `memo_put`
            // then advances (see `next_memo_key`).
            0x94 => {
                let key = self.next_memo_key;
                self.memo_put(key)?;
            }
            // GET (protocol 0)
            b'g' => {
                let line = self.read_line_str()?;
                let key =
                    line.trim()
                        .parse::<u32>()
                        .map_err(|_| PickleError::MalformedLiteral {
                            opcode: "GET",
                            offset,
                        })?;
                self.memo_get(key)?;
            }
            // BINGET
            b'h' => {
                let key = self.read_u8()?;
                self.memo_get(u32::from(key))?;
            }
            // LONG_BINGET
            b'j' => {
                let key = self.read_u32()?;
                self.memo_get(key)?;
            }

            // --- stack manipulation ---
            b'(' => self.mark()?,
            // POP: drops the top of stack, or the innermost mark if the
            // stack has not grown since it was set.
            b'0' => {
                if self.marks.last() == Some(&self.stack.len()) {
                    self.marks.pop();
                } else {
                    self.pop()?;
                }
            }
            // POP_MARK
            b'1' => {
                self.pop_to_mark()?;
            }
            // DUP
            b'2' => {
                let top = self
                    .stack
                    .last()
                    .ok_or(PickleError::StackUnderflow { offset })?
                    .clone();
                self.push(top)?;
            }

            // --- globals and object construction ---
            // GLOBAL
            b'c' => {
                let module = self.read_line_str()?;
                let name = self.read_line_str()?;
                self.push(Value::Global { module, name })?;
            }
            // STACK_GLOBAL
            0x93 => {
                let name = self.pop()?;
                let module = self.pop()?;
                let (module, name) = match (module.as_text(), name.as_text()) {
                    (Some(module), Some(name)) => (module, name),
                    _ => {
                        return Err(PickleError::WrongTarget {
                            opcode: "STACK_GLOBAL",
                            found: "non-string",
                            offset,
                        })
                    }
                };
                self.push(Value::Global { module, name })?;
            }
            // REDUCE
            b'R' => {
                let args = self.pop()?;
                let callable = self.pop()?;
                self.push(Value::Reduce {
                    callable: Box::new(callable),
                    args: Box::new(args),
                })?;
            }
            // NEWOBJ
            0x81 => {
                let args = self.pop()?;
                let class = self.pop()?;
                self.push(new_object(class, args))?;
            }
            // NEWOBJ_EX: (class, args, kwargs). The kwargs dict is merged
            // into the recorded state so a `__setstate__`-less class built
            // purely from keywords is still readable.
            0x92 => {
                let kwargs = self.pop()?;
                let args = self.pop()?;
                let class = self.pop()?;
                let mut object = new_object(class, args);
                if let (Value::Object { state, .. }, Value::Dict(entries)) = (&mut object, &kwargs)
                {
                    if !entries.is_empty() {
                        *state = Some(Box::new(kwargs.clone()));
                    }
                }
                self.push(object)?;
            }
            // OBJ: like REDUCE but the callable sits below a MARK.
            b'o' => {
                let mut items = self.pop_to_mark()?;
                if items.is_empty() {
                    return Err(PickleError::StackUnderflow { offset });
                }
                let class = items.remove(0);
                self.push(new_object(class, Value::Tuple(items)))?;
            }
            // INST: GLOBAL + OBJ in one opcode (protocol 0).
            b'i' => {
                let module = self.read_line_str()?;
                let name = self.read_line_str()?;
                let args = self.pop_to_mark()?;
                self.push(new_object(
                    Value::Global { module, name },
                    Value::Tuple(args),
                ))?;
            }
            // BUILD
            b'b' => {
                let state = self.pop()?;
                let target = self.pop()?;
                self.push(apply_build(target, state))?;
            }

            // --- persistent ids ---
            // PERSID (protocol 0)
            b'P' => {
                let id = Value::Bytes(self.read_line()?.to_vec());
                let resolved = self.resolver.resolve(&id)?;
                self.push(resolved)?;
            }
            // BINPERSID
            b'Q' => {
                let id = self.pop()?;
                let resolved = self.resolver.resolve(&id)?;
                self.push(resolved)?;
            }

            // --- extension registry (protocol 2), never emitted by torch.save ---
            0x82..=0x84 => {
                return Err(PickleError::UnsupportedOpcode { opcode, offset });
            }

            other => {
                return Err(PickleError::UnsupportedOpcode {
                    opcode: other,
                    offset,
                })
            }
        }
        Ok(false)
    }
}

/// Wraps `class` and `args` as a fresh, state-less object.
fn new_object(class: Value, args: Value) -> Value {
    Value::Object {
        class: Box::new(class),
        args: Box::new(args),
        state: None,
        list_items: Vec::new(),
        dict_items: Vec::new(),
    }
}

/// Applies a `BUILD` state to `target`, promoting a bare `REDUCE` (or any
/// other value) to an [`Value::Object`] so the state has somewhere to live.
fn apply_build(target: Value, state: Value) -> Value {
    match target {
        Value::Object {
            class,
            args,
            list_items,
            dict_items,
            ..
        } => Value::Object {
            class,
            args,
            state: Some(Box::new(state)),
            list_items,
            dict_items,
        },
        Value::Reduce { callable, args } => Value::Object {
            class: callable,
            args,
            state: Some(Box::new(state)),
            list_items: Vec::new(),
            dict_items: Vec::new(),
        },
        other => Value::Object {
            class: Box::new(other),
            args: Box::new(Value::Tuple(Vec::new())),
            state: Some(Box::new(state)),
            list_items: Vec::new(),
            dict_items: Vec::new(),
        },
    }
}

/// `APPEND`/`APPENDS` target handling: a real list grows, a list *subclass*
/// (an `Object`) records the items separately.
fn append_items(target: &mut Value, items: Vec<Value>, offset: usize) -> Result<()> {
    match target {
        Value::List(existing) => existing.extend(items),
        Value::Set(existing) => existing.extend(items),
        Value::Object { list_items, .. } => list_items.extend(items),
        other => {
            return Err(PickleError::WrongTarget {
                opcode: "APPEND(S)",
                found: other.type_name(),
                offset,
            })
        }
    }
    Ok(())
}

/// `SETITEM`/`SETITEMS` target handling, mirroring [`append_items`].
fn set_items(target: &mut Value, pairs: Vec<(Value, Value)>, offset: usize) -> Result<()> {
    match target {
        Value::Dict(existing) => existing.extend(pairs),
        Value::Object { dict_items, .. } => dict_items.extend(pairs),
        other => {
            return Err(PickleError::WrongTarget {
                opcode: "SETITEM(S)",
                found: other.type_name(),
                offset,
            })
        }
    }
    Ok(())
}

/// Turns a flat `[k, v, k, v, …]` run into pairs.
fn pair_up(flat: Vec<Value>, offset: usize) -> Result<Vec<(Value, Value)>> {
    if !flat.len().is_multiple_of(2) {
        return Err(PickleError::WrongTarget {
            opcode: "DICT/SETITEMS",
            found: "odd number of key/value entries",
            offset,
        });
    }
    let mut pairs = Vec::with_capacity(flat.len() / 2);
    let mut iter = flat.into_iter();
    while let (Some(key), Some(value)) = (iter.next(), iter.next()) {
        pairs.push((key, value));
    }
    Ok(pairs)
}

/// Decodes a `LONG1`/`LONG4` payload: little-endian two's complement of
/// arbitrary width. Values that fit an `i64` become [`Value::Int`]; wider
/// ones are kept verbatim as [`Value::BigInt`] rather than being truncated.
fn decode_signed_le(bytes: &[u8]) -> Value {
    if bytes.is_empty() {
        return Value::Int(0);
    }
    let negative = bytes[bytes.len() - 1] & 0x80 != 0;
    // Sign-extend into 8 bytes when it fits; otherwise the value genuinely
    // needs more than 64 bits only if a significant (non sign-fill) byte
    // sits beyond the 8th.
    let significant = bytes.len()
        - bytes
            .iter()
            .rev()
            .take_while(|&&b| b == if negative { 0xff } else { 0x00 })
            .count();
    if significant <= 8 {
        let mut buf = [if negative { 0xff } else { 0x00 }; 8];
        let take = bytes.len().min(8);
        buf[..take].copy_from_slice(&bytes[..take]);
        let value = i64::from_le_bytes(buf);
        // A positive value whose top significant byte has its sign bit set
        // needs a 9th zero byte to stay positive, so it does not fit.
        if significant == 8 && !negative && value < 0 {
            return Value::BigInt(bytes.to_vec());
        }
        return Value::Int(value);
    }
    Value::BigInt(bytes.to_vec())
}

/// Decodes the repr-quoted payload of a protocol-0 `STRING` opcode.
///
/// The payload is a Python string literal: a quote character, escaped
/// content, and the matching closing quote. Only the escapes CPython's own
/// pickler emits (`\\`, `\'`, `\"`, `\n`, `\r`, `\t`, and `\xNN`) are
/// recognized; anything else is an error rather than being silently taken
/// literally, so a corrupt file is reported instead of yielding wrong bytes.
fn decode_repr_string(line: &[u8], offset: usize) -> Result<Vec<u8>> {
    let malformed = || PickleError::MalformedLiteral {
        opcode: "STRING",
        offset,
    };
    if line.len() < 2 {
        return Err(malformed());
    }
    let quote = line[0];
    if (quote != b'\'' && quote != b'"') || line[line.len() - 1] != quote {
        return Err(malformed());
    }
    let body = &line[1..line.len() - 1];

    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        if body[i] != b'\\' {
            out.push(body[i]);
            i += 1;
            continue;
        }
        let escape = *body.get(i + 1).ok_or_else(malformed)?;
        match escape {
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'0' => out.push(0),
            b'x' => {
                let hex = body.get(i + 2..i + 4).ok_or_else(malformed)?;
                let text = std::str::from_utf8(hex).map_err(|_| malformed())?;
                out.push(u8::from_str_radix(text, 16).map_err(|_| malformed())?);
                i += 2;
            }
            _ => return Err(malformed()),
        }
        i += 2;
    }
    Ok(out)
}

/// Decodes a protocol-0 `UNICODE` payload, which CPython writes with the
/// `raw-unicode-escape` codec: ASCII passes through, and anything else
/// appears as `\uXXXX`.
fn decode_raw_unicode_escape(line: &[u8]) -> String {
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < line.len() {
        if line[i] == b'\\' && line.get(i + 1) == Some(&b'u') {
            if let Some(hex) = line.get(i + 2..i + 6) {
                if let Some(ch) = std::str::from_utf8(hex)
                    .ok()
                    .and_then(|t| u32::from_str_radix(t, 16).ok())
                    .and_then(char::from_u32)
                {
                    out.push(ch);
                    i += 6;
                    continue;
                }
            }
        }
        out.push(line[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a protocol-2 pickle by hand. Keeping these as byte literals
    /// (rather than fixtures produced by a Python that may not be installed)
    /// means the tests run everywhere and document the wire format.
    fn proto2(body: &[u8]) -> Vec<u8> {
        let mut out = vec![0x80, 0x02];
        out.extend_from_slice(body);
        out.push(b'.');
        out
    }

    #[test]
    fn test_empty_dict() {
        let value = load(&proto2(b"}")).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Dict(Vec::new()));
    }

    #[test]
    fn test_dict_with_short_binunicode_keys() {
        // } q\0 X\4\0\0\0 "conv" K 7 s  ->  {"conv": 7}
        let mut body = vec![b'}', b'q', 0x00, b'X', 4, 0, 0, 0];
        body.extend_from_slice(b"conv");
        body.extend_from_slice(&[b'K', 7, b's']);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value.get("conv"), Some(&Value::Int(7)));
    }

    #[test]
    fn test_setitems_with_mark() {
        let mut body = vec![b'}', b'('];
        body.extend_from_slice(&[b'K', 1, b'K', 10, b'K', 2, b'K', 20, b'u']);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(
            value,
            Value::Dict(vec![
                (Value::Int(1), Value::Int(10)),
                (Value::Int(2), Value::Int(20)),
            ])
        );
    }

    #[test]
    fn test_tuple_opcodes() {
        let value = load(&proto2(&[b'K', 1, b'K', 2, b'K', 3, 0x87]))
            .expect("test: unpickle should succeed");
        assert_eq!(
            value,
            Value::Tuple(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );

        let value = load(&proto2(b")")).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Tuple(Vec::new()));

        let mut body = vec![b'('];
        body.extend_from_slice(&[b'K', 1, b'K', 2, b'K', 3, b'K', 4, b't']);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value.as_seq().map(<[Value]>::len), Some(4));
    }

    #[test]
    fn test_memo_roundtrip() {
        // ] q\0 K 5 a  -- memoize the list, then h\0 must retrieve it, so
        // appending through the retrieved reference is observable on the
        // single value left on the stack.
        let body = vec![b']', b'q', 0x00, b'K', 5, b'a'];
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value, Value::List(vec![Value::Int(5)]));

        // K 7 q\1 0 ] h\1 a  -- push 7, memoize it under key 1, POP it,
        // then retrieve it from the memo and append. The memo must survive
        // the value leaving the stack.
        let body = vec![b'K', 7, b'q', 0x01, b'0', b']', b'h', 0x01, b'a'];
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value, Value::List(vec![Value::Int(7)]));
    }

    #[test]
    fn test_memoize_does_not_collide_with_an_explicit_put() {
        // Regression test: MEMOIZE stores at "the next available index",
        // which CPython tracks as a monotonic counter. Deriving that index
        // from `memo.len()` instead is wrong whenever an explicit PUT
        // leaves a *gap* in the key space, because len then undercounts the
        // highest key in use and MEMOIZE eventually walks onto it.
        //
        // Here `BINPUT 3` claims key 3 while len is only 1, so the len-based
        // scheme hands out 1, 2, then 3 -- clobbering the BINPUT's value on
        // its third step (and, since len no longer grows, every step after).
        //
        //   K 10 q\3  -> key 3 = 10          (len 1, counter 4)
        //   0         -> POP
        //   K 20 \x94 -> len-based key 1; correct key 4
        //   0         -> POP
        //   K 30 \x94 -> len-based key 2; correct key 5
        //   0         -> POP
        //   K 40 \x94 -> len-based key 3 == COLLISION; correct key 6
        //   0         -> POP
        //   ] h\3 a   -> [10] if key 3 survived, [40] if it was clobbered
        let body = vec![
            b'K', 10, b'q', 0x03, b'0', // BINPUT 3, POP
            b'K', 20, 0x94, b'0', // MEMOIZE, POP
            b'K', 30, 0x94, b'0', // MEMOIZE, POP
            b'K', 40, 0x94, b'0', // MEMOIZE, POP
            b']', b'h', 0x03, b'a', // [] <- memo[3]
        ];
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(
            value,
            Value::List(vec![Value::Int(10)]),
            "MEMOIZE must not overwrite a key an explicit PUT already claimed"
        );

        // The counter must also keep advancing past the PUT, so the three
        // MEMOIZEd values land on distinct, fresh keys.
        let body = vec![
            b'K', 10, b'q', 0x03, b'0', //
            b'K', 20, 0x94, b'0', //
            b'K', 30, 0x94, b'0', //
            b']', b'h', 0x04, b'a', b'h', 0x05, b'a',
        ];
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value, Value::List(vec![Value::Int(20), Value::Int(30)]));
    }

    #[test]
    fn test_memo_miss_is_an_error_not_a_panic() {
        let err = load(&proto2(&[b'h', 0x07])).expect_err("missing memo key must error");
        assert!(matches!(err, PickleError::MemoMiss { key: 7 }));
    }

    #[test]
    fn test_global_and_reduce_are_recorded_not_executed() {
        // c os\n system\n ) R  -- the canonical "arbitrary code execution"
        // pickle. It must decode to inert data, never dispatch anything.
        let mut body = Vec::new();
        body.extend_from_slice(b"cos\nsystem\n");
        body.extend_from_slice(b")R");
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        match value {
            Value::Reduce { callable, args } => {
                assert_eq!(
                    *callable,
                    Value::Global {
                        module: "os".into(),
                        name: "system".into()
                    }
                );
                assert_eq!(*args, Value::Tuple(Vec::new()));
            }
            other => panic!("expected an inert Reduce record, got {other:?}"),
        }
    }

    #[test]
    fn test_build_attaches_state_to_reduce() {
        let mut body = Vec::new();
        body.extend_from_slice(b"cmymod\nMyClass\n");
        body.extend_from_slice(&[b')', b'R', b'}', b'q', 0x00]);
        body.extend_from_slice(&[b'X', 1, 0, 0, 0, b'a', b'K', 3, b's', b'b']);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value.class_path(), Some(("mymod", "MyClass")));
        assert_eq!(value.get("a"), Some(&Value::Int(3)));
    }

    #[test]
    fn test_binfloat_is_big_endian() {
        let mut body = vec![b'G'];
        body.extend_from_slice(&1.5f64.to_be_bytes());
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Float(1.5));
    }

    #[test]
    fn test_long1_signed_decoding() {
        // LONG1 with a single 0xff byte is -1, not 255.
        let value = load(&proto2(&[0x8a, 1, 0xff])).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Int(-1));

        // A zero-length LONG1 is Python's canonical encoding of 0.
        let value = load(&proto2(&[0x8a, 0])).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Int(0));

        // 2^64 needs 9 bytes and must not be truncated into an i64.
        let mut body = vec![0x8a, 9];
        body.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 1]);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert!(matches!(value, Value::BigInt(_)));
    }

    #[test]
    fn test_protocol_0_text_opcodes() {
        // No PROTO opcode at all: a protocol-0 stream.
        let mut body = Vec::new();
        body.extend_from_slice(b"(lp0\nI42\naS'hi'\np1\naF1.5\na.");
        let value = load(&body).expect("test: unpickle should succeed");
        assert_eq!(
            value,
            Value::List(vec![
                Value::Int(42),
                Value::Bytes(b"hi".to_vec()),
                Value::Float(1.5),
            ])
        );
    }

    #[test]
    fn test_string_opcode_hex_escapes() {
        let decoded = decode_repr_string(br"'a\x00\n\\b'", 0).expect("test: decode should succeed");
        assert_eq!(decoded, b"a\x00\n\\b");
    }

    #[test]
    fn test_string_opcode_rejects_unknown_escape() {
        assert!(decode_repr_string(br"'a\q'", 0).is_err());
        assert!(decode_repr_string(b"'unterminated", 0).is_err());
    }

    #[test]
    fn test_protocol_5_framing_and_memoize() {
        // PROTO 5, FRAME <len>, then ordinary opcodes with MEMOIZE.
        let mut body = vec![0x80, 0x05, 0x95];
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&[b'K', 9, 0x94, b'.']);
        let value = load(&body).expect("test: unpickle should succeed");
        assert_eq!(value, Value::Int(9));
    }

    #[test]
    fn test_truncated_stream_errors_instead_of_panicking() {
        // A BINUNICODE claiming 4 GiB of payload in a 12-byte file.
        let mut body = vec![0x80, 0x02, b'X'];
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(b"abc");
        let err = load(&body).expect_err("truncated payload must error");
        assert!(matches!(err, PickleError::Truncated { .. }));

        // A stream that simply ends before STOP.
        let err = load(&[0x80, 0x02, b'}']).expect_err("missing STOP must error");
        assert!(matches!(err, PickleError::Truncated { .. }));
    }

    #[test]
    fn test_stack_underflow_errors_instead_of_panicking() {
        let err = load(&proto2(b"R")).expect_err("REDUCE on an empty stack must error");
        assert!(matches!(err, PickleError::StackUnderflow { .. }));

        let err = load(&proto2(b"t")).expect_err("TUPLE with no MARK must error");
        assert!(matches!(err, PickleError::StackUnderflow { .. }));
    }

    #[test]
    fn test_unknown_opcode_errors() {
        let err = load(&proto2(&[0x01])).expect_err("unknown opcode must error");
        assert!(matches!(
            err,
            PickleError::UnsupportedOpcode { opcode: 1, .. }
        ));
    }

    #[test]
    fn test_unsupported_protocol_errors() {
        let err = load(&[0x80, 0x09, b'N', b'.']).expect_err("protocol 9 must be rejected");
        assert!(matches!(
            err,
            PickleError::UnsupportedProtocol { version: 9 }
        ));
    }

    #[test]
    fn test_persistent_ids_go_through_the_resolver() {
        struct Counting(usize);
        impl PersistentIdResolver for Counting {
            fn resolve(&mut self, id: &Value) -> Result<Value> {
                self.0 += 1;
                Ok(Value::Str(format!("resolved:{id}")))
            }
        }

        let mut body = vec![b'X', 3, 0, 0, 0];
        body.extend_from_slice(b"abc");
        body.push(b'Q');
        let mut resolver = Counting(0);
        let value =
            load_with(&proto2(&body), &mut resolver).expect("test: unpickle should succeed");
        assert_eq!(resolver.0, 1);
        assert_eq!(value, Value::Str("resolved:'abc'".into()));
    }

    #[test]
    fn test_mark_depth_is_bounded() {
        // A crafted stream of nothing but MARK must be refused rather than
        // growing the mark stack without bound.
        let body = vec![b'('; MAX_MARK_DEPTH + 8];
        let err = load(&proto2(&body)).expect_err("unbounded MARK nesting must error");
        assert!(matches!(err, PickleError::LimitExceeded { .. }));
    }

    #[test]
    fn test_newobj_records_class_and_args() {
        let mut body = Vec::new();
        body.extend_from_slice(b"cmymod\nMyClass\n");
        body.extend_from_slice(&[b'K', 1, 0x85, 0x81]);
        let value = load(&proto2(&body)).expect("test: unpickle should succeed");
        assert_eq!(value.class_path(), Some(("mymod", "MyClass")));
        assert_eq!(value.ctor_args(), Some(&[Value::Int(1)][..]));
    }
}
