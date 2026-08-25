//! A tiny protocol-2 pickle *writer*, used only by this module's tests.
//!
//! The pickle reader's tests need real pickle streams. Shipping binary
//! fixtures would make them opaque and would tie the crate's test suite to a
//! Python installation that may not exist on a given machine; emitting the
//! opcodes here keeps every test hermetic and makes the wire format legible
//! at the call site.
//!
//! This is `#[cfg(test)]`-only: nothing in the shipped crate writes pickles.

use super::value::Value;

/// Accumulates protocol-2 opcodes.
pub struct PickleBuilder {
    bytes: Vec<u8>,
}

impl PickleBuilder {
    /// Start a stream, emitting the `PROTO 2` header.
    pub fn new() -> Self {
        Self {
            bytes: vec![0x80, 0x02],
        }
    }

    /// `EMPTY_DICT`
    pub fn empty_dict(&mut self) -> &mut Self {
        self.bytes.push(b'}');
        self
    }

    /// `MARK`
    pub fn mark(&mut self) -> &mut Self {
        self.bytes.push(b'(');
        self
    }

    /// `TUPLE` (pops back to the most recent `MARK`)
    pub fn tuple(&mut self) -> &mut Self {
        self.bytes.push(b't');
        self
    }

    /// `BINUNICODE`
    pub fn unicode(&mut self, text: &str) -> &mut Self {
        self.bytes.push(b'X');
        self.bytes
            .extend_from_slice(&(text.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(text.as_bytes());
        self
    }

    /// `BININT`-family, choosing the narrowest encoding.
    pub fn int(&mut self, value: i64) -> &mut Self {
        match value {
            0..=0xff => self.bytes.extend_from_slice(&[b'K', value as u8]),
            0x100..=0xffff => {
                self.bytes.push(b'M');
                self.bytes.extend_from_slice(&(value as u16).to_le_bytes());
            }
            _ => match i32::try_from(value) {
                Ok(narrow) => {
                    self.bytes.push(b'J');
                    self.bytes.extend_from_slice(&narrow.to_le_bytes());
                }
                // Beyond i32, CPython emits LONG1 (little-endian two's
                // complement). Falling back to `value as i32` here would
                // silently truncate, which once made a test asserting an
                // oversized-shape rejection instead assert a shape of 0.
                Err(_) => {
                    self.bytes.push(0x8a);
                    self.bytes.push(8);
                    self.bytes.extend_from_slice(&value.to_le_bytes());
                }
            },
        }
        self
    }

    /// `NEWTRUE` / `NEWFALSE`
    pub fn bool(&mut self, value: bool) -> &mut Self {
        self.bytes.push(if value { 0x88 } else { 0x89 });
        self
    }

    /// `NONE`
    pub fn none(&mut self) -> &mut Self {
        self.bytes.push(b'N');
        self
    }

    /// `BINFLOAT` (big-endian, per the format)
    pub fn float(&mut self, value: f64) -> &mut Self {
        self.bytes.push(b'G');
        self.bytes.extend_from_slice(&value.to_be_bytes());
        self
    }

    /// `SHORT_BINSTRING`: a Python 2 `str`, which is how a protocol-2
    /// pickle written by Python 2 stores both names and raw array bytes.
    pub fn py2_str(&mut self, bytes: &[u8]) -> &mut Self {
        if let Ok(len) = u8::try_from(bytes.len()) {
            self.bytes.push(b'U');
            self.bytes.push(len);
        } else {
            self.bytes.push(b'T');
            self.bytes
                .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        }
        self.bytes.extend_from_slice(bytes);
        self
    }

    /// `GLOBAL`
    pub fn global(&mut self, module: &str, name: &str) -> &mut Self {
        self.bytes.push(b'c');
        self.bytes.extend_from_slice(module.as_bytes());
        self.bytes.push(b'\n');
        self.bytes.extend_from_slice(name.as_bytes());
        self.bytes.push(b'\n');
        self
    }

    /// `REDUCE`
    pub fn reduce(&mut self) -> &mut Self {
        self.bytes.push(b'R');
        self
    }

    /// `BUILD`
    pub fn build_state(&mut self) -> &mut Self {
        self.bytes.push(b'b');
        self
    }

    /// `BINPERSID`
    pub fn binpersid(&mut self) -> &mut Self {
        self.bytes.push(b'Q');
        self
    }

    /// `SETITEM`
    pub fn setitem(&mut self) -> &mut Self {
        self.bytes.push(b's');
        self
    }

    /// A `MARK`-delimited tuple of non-negative integers.
    pub fn int_tuple(&mut self, values: &[usize]) -> &mut Self {
        self.mark();
        for &value in values {
            self.int(value as i64);
        }
        self.tuple()
    }

    /// Finish the stream with `STOP`.
    pub fn finish(mut self) -> Vec<u8> {
        self.bytes.push(b'.');
        self.bytes
    }
}

impl Default for PickleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: build a stream with `f` and terminate it.
pub fn pickle(f: impl FnOnce(&mut PickleBuilder)) -> Vec<u8> {
    let mut builder = PickleBuilder::new();
    f(&mut builder);
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickle::vm;

    #[test]
    fn test_builder_round_trips_through_the_reader() {
        // The writer exists to feed the reader, so the two must agree; a
        // bug in the writer would otherwise show up as a phantom reader bug
        // in every test that uses it.
        let bytes = pickle(|p| {
            p.empty_dict();
            p.unicode("name");
            p.py2_str(b"value");
            p.setitem();
            p.unicode("nums");
            p.int_tuple(&[1, 2, 3]);
            p.setitem();
            p.unicode("flag");
            p.bool(true);
            p.setitem();
            p.unicode("x");
            p.float(0.5);
            p.setitem();
            p.unicode("nil");
            p.none();
            p.setitem();
        });

        let value = vm::load(&bytes).expect("test: unpickle should succeed");
        assert_eq!(
            value.get("name").and_then(Value::as_text).as_deref(),
            Some("value")
        );
        assert_eq!(
            value.get("nums").and_then(Value::as_seq),
            Some(&[Value::Int(1), Value::Int(2), Value::Int(3)][..])
        );
        assert_eq!(value.get("flag"), Some(&Value::Bool(true)));
        assert_eq!(value.get("x").and_then(Value::as_f64), Some(0.5));
        assert_eq!(value.get("nil"), Some(&Value::None));
    }

    #[test]
    fn test_builder_int_widths() {
        for probe in [
            0i64,
            1,
            255,
            256,
            65535,
            65536,
            1_000_000,
            i64::from(i32::MAX),
            // Beyond i32 the builder must switch to LONG1 rather than
            // truncating: a silent `as i32` here once turned an
            // oversized-shape test into a zero-shape one that passed
            // vacuously.
            i64::from(i32::MAX) + 1,
            1 << 34,
            i64::MAX,
            -1,
            i64::from(i32::MIN),
            i64::MIN,
        ] {
            let bytes = pickle(|p| {
                p.int(probe);
            });
            let value = vm::load(&bytes).expect("test: unpickle should succeed");
            assert_eq!(value, Value::Int(probe), "width mismatch for {probe}");
        }
    }
}
