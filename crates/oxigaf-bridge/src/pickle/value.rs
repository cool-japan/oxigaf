//! The value tree an unpickled Python object graph decodes into.
//!
//! The unpickler in [`super::vm`] is deliberately *non-executing*: `REDUCE`,
//! `NEWOBJ`, `BUILD` and friends are recorded structurally rather than
//! dispatched to Rust implementations of the named Python callables. That
//! keeps the VM total and side-effect free -- a malicious pickle can make
//! this crate allocate, but never call anything -- and it lets each format
//! interpreter ([`super::torch`], [`super::numpy`]) decide for itself which
//! constructors it recognizes and reject the rest with a specific error.

use std::fmt;

/// A decoded Python value.
///
/// Python 2 `str` (the `STRING`/`BINSTRING`/`SHORT_BINSTRING` opcodes) and
/// Python 3 `bytes` both decode to [`Value::Bytes`], because a protocol-2
/// pickle written by Python 2 -- which is what the FLAME `.pkl` models are --
/// stores raw array data in `str` objects. Use [`Value::as_text`] where a
/// human-readable name is expected; it accepts either representation.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Python `None`.
    None,
    /// Python `bool`.
    Bool(bool),
    /// Python `int` that fits in an `i64`.
    Int(i64),
    /// Python `int` too large for an `i64`, as little-endian two's complement
    /// bytes exactly as the `LONG1`/`LONG4` opcode carried them.
    BigInt(Vec<u8>),
    /// Python `float`.
    Float(f64),
    /// Python 3 `bytes`, or a Python 2 `str`.
    Bytes(Vec<u8>),
    /// Python 3 `str` (a `UNICODE`/`BINUNICODE`/`SHORT_BINUNICODE` opcode).
    Str(String),
    /// Python `tuple`.
    Tuple(Vec<Value>),
    /// Python `list`.
    List(Vec<Value>),
    /// Python `dict`, in insertion order.
    Dict(Vec<(Value, Value)>),
    /// Python `set` or `frozenset`.
    Set(Vec<Value>),
    /// A dotted global reference (`GLOBAL` / `STACK_GLOBAL`), e.g.
    /// `numpy.core.multiarray._reconstruct`. Never resolved to a callable.
    Global {
        /// Module path, e.g. `torch._utils`.
        module: String,
        /// Attribute name within the module, e.g. `_rebuild_tensor_v2`.
        name: String,
    },
    /// A recorded `callable(*args)` from `REDUCE`, not evaluated.
    Reduce {
        /// The callable, usually a [`Value::Global`].
        callable: Box<Value>,
        /// The argument tuple.
        args: Box<Value>,
    },
    /// An object under construction: the result of `NEWOBJ`/`NEWOBJ_EX`/
    /// `OBJ`/`INST`, or of a `REDUCE` that a later `BUILD` attached state to.
    Object {
        /// The class or callable the object was constructed from.
        class: Box<Value>,
        /// The positional construction arguments.
        args: Box<Value>,
        /// The `__setstate__` argument supplied by `BUILD`, if any.
        state: Option<Box<Value>>,
        /// Items appended by `APPEND`/`APPENDS` (list subclasses).
        list_items: Vec<Value>,
        /// Items set by `SETITEM`/`SETITEMS` (dict subclasses).
        dict_items: Vec<(Value, Value)>,
    },
    /// A `PERSID`/`BINPERSID` reference. The container format decides what
    /// the identifier means -- for a PyTorch `.pt` it names a storage blob
    /// inside the surrounding ZIP.
    PersistentId(Box<Value>),
}

impl Value {
    /// Short type name, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Bool(_) => "bool",
            Self::Int(_) | Self::BigInt(_) => "int",
            Self::Float(_) => "float",
            Self::Bytes(_) => "bytes",
            Self::Str(_) => "str",
            Self::Tuple(_) => "tuple",
            Self::List(_) => "list",
            Self::Dict(_) => "dict",
            Self::Set(_) => "set",
            Self::Global { .. } => "global",
            Self::Reduce { .. } => "reduce",
            Self::Object { .. } => "object",
            Self::PersistentId(_) => "persistent id",
        }
    }

    /// Interprets the value as text, accepting both a Python 3 `str` and a
    /// Python 2 `str`/`bytes`.
    ///
    /// Byte strings are decoded as UTF-8 when they are valid UTF-8 and as
    /// Latin-1 otherwise, mirroring `pickle.load(f, encoding="latin1")` --
    /// the encoding the FLAME conversion has always used. Layer names and
    /// dict keys are ASCII in practice, so the two agree there; the fallback
    /// only matters for a stray high byte, which Latin-1 maps losslessly
    /// rather than failing.
    pub fn as_text(&self) -> Option<String> {
        match self {
            Self::Str(s) => Some(s.clone()),
            Self::Bytes(b) => Some(match std::str::from_utf8(b) {
                Ok(s) => s.to_string(),
                Err(_) => b.iter().map(|&c| c as char).collect(),
            }),
            _ => None,
        }
    }

    /// The value as an integer, accepting `int` and `bool`.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    /// The value as a non-negative `usize`.
    pub fn as_usize(&self) -> Option<usize> {
        usize::try_from(self.as_i64()?).ok()
    }

    /// The value as a float, accepting `float`, `int` and `bool`.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            other => other.as_i64().map(|i| i as f64),
        }
    }

    /// The elements of a `tuple` or `list`.
    pub fn as_seq(&self) -> Option<&[Value]> {
        match self {
            Self::Tuple(v) | Self::List(v) => Some(v),
            _ => None,
        }
    }

    /// The raw bytes of a Python 2 `str` / Python 3 `bytes`.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// The entries of a `dict`, or of an object whose `BUILD` state was a
    /// plain `__dict__` (the default `__reduce_ex__` behaviour for a class
    /// without `__setstate__`, which is how `scipy.sparse` matrices and
    /// `chumpy` arrays pickle).
    pub fn as_mapping(&self) -> Option<&[(Value, Value)]> {
        match self {
            Self::Dict(entries) => Some(entries),
            Self::Object {
                state: Some(state),
                dict_items,
                ..
            } => match state.as_ref() {
                Self::Dict(entries) => Some(entries),
                // `object.__reduce_ex__` may hand `(instance_dict, slots_dict)`.
                Self::Tuple(parts) => parts.first()?.as_mapping(),
                _ => Some(dict_items),
            },
            Self::Object { dict_items, .. } if !dict_items.is_empty() => Some(dict_items),
            _ => None,
        }
    }

    /// Looks a key up in a mapping (see [`Value::as_mapping`]), comparing
    /// keys through [`Value::as_text`] so a Python 2 `str` key matches a
    /// Rust `&str` just as a Python 3 `str` key does.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_mapping()?
            .iter()
            .find(|(k, _)| k.as_text().as_deref() == Some(key))
            .map(|(_, v)| v)
    }

    /// The `module.name` this value refers to, if it is a [`Value::Global`],
    /// or the class of an [`Value::Object`] / callable of a
    /// [`Value::Reduce`], following one level of nesting.
    ///
    /// `NEWOBJ` produces `Object { class: <the class global> }` while a
    /// plain `REDUCE` on `copyreg._reconstructor` produces
    /// `Object { class: Global(copyreg, _reconstructor), args: (RealClass, ...) }`;
    /// this resolves the former directly and the latter through its first
    /// argument, so callers can ask "what class is this?" without caring
    /// which construction opcode the pickler chose.
    pub fn class_path(&self) -> Option<(&str, &str)> {
        match self {
            Self::Global { module, name } => Some((module, name)),
            Self::Reduce { callable, .. } => callable.class_path(),
            Self::Object { class, args, .. } => match class.class_path() {
                Some(("copyreg", "_reconstructor")) | Some(("copy_reg", "_reconstructor")) => {
                    args.as_seq()?.first()?.class_path()
                }
                other => other,
            },
            _ => None,
        }
    }

    /// The positional arguments an object or reduce was constructed with.
    pub fn ctor_args(&self) -> Option<&[Value]> {
        match self {
            Self::Reduce { args, .. } | Self::Object { args, .. } => args.as_seq(),
            _ => None,
        }
    }

    /// The `BUILD` state attached to an object, if any.
    pub fn state(&self) -> Option<&Value> {
        match self {
            Self::Object { state, .. } => state.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    /// A short, non-recursive rendering suitable for error messages: it
    /// names the shape of the value rather than dumping a multi-megabyte
    /// tensor graph into a log line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::BigInt(b) => write!(f, "<{}-byte int>", b.len()),
            Self::Float(x) => write!(f, "{x}"),
            Self::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            Self::Str(s) => write!(f, "'{s}'"),
            Self::Tuple(v) => write!(f, "<tuple of {}>", v.len()),
            Self::List(v) => write!(f, "<list of {}>", v.len()),
            Self::Dict(v) => write!(f, "<dict of {}>", v.len()),
            Self::Set(v) => write!(f, "<set of {}>", v.len()),
            Self::Global { module, name } => write!(f, "{module}.{name}"),
            Self::Reduce { callable, .. } => write!(f, "{callable}(...)"),
            Self::Object { class, .. } => write!(f, "<{class} object>"),
            Self::PersistentId(id) => write!(f, "<persistent id {id}>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_as_text_accepts_py2_str_and_py3_str() {
        assert_eq!(Value::Str("abc".into()).as_text().as_deref(), Some("abc"));
        assert_eq!(
            Value::Bytes(b"abc".to_vec()).as_text().as_deref(),
            Some("abc")
        );
        // Latin-1 fallback for bytes that are not valid UTF-8, matching
        // `pickle.load(f, encoding="latin1")`.
        assert_eq!(Value::Bytes(vec![0xe9]).as_text().as_deref(), Some("é"));
        assert_eq!(Value::Int(3).as_text(), None);
    }

    #[test]
    fn test_get_matches_bytes_keys() {
        let dict = Value::Dict(vec![
            (Value::Bytes(b"v_template".to_vec()), Value::Int(1)),
            (Value::Str("f".into()), Value::Int(2)),
        ]);
        assert_eq!(dict.get("v_template"), Some(&Value::Int(1)));
        assert_eq!(dict.get("f"), Some(&Value::Int(2)));
        assert_eq!(dict.get("missing"), None);
    }

    #[test]
    fn test_class_path_follows_copyreg_reconstructor() {
        // `copyreg._reconstructor(RealClass, object, None)` is how a plain
        // old-style class pickles; asking for "the class" must see through
        // it rather than reporting `copyreg._reconstructor`.
        let obj = Value::Object {
            class: Box::new(Value::Global {
                module: "copyreg".into(),
                name: "_reconstructor".into(),
            }),
            args: Box::new(Value::Tuple(vec![
                Value::Global {
                    module: "chumpy.ch".into(),
                    name: "Ch".into(),
                },
                Value::None,
                Value::None,
            ])),
            state: None,
            list_items: Vec::new(),
            dict_items: Vec::new(),
        };
        assert_eq!(obj.class_path(), Some(("chumpy.ch", "Ch")));
    }

    #[test]
    fn test_as_mapping_reads_object_dict_state() {
        let obj = Value::Object {
            class: Box::new(Value::Global {
                module: "scipy.sparse.csc".into(),
                name: "csc_matrix".into(),
            }),
            args: Box::new(Value::Tuple(vec![])),
            state: Some(Box::new(Value::Dict(vec![(
                Value::Bytes(b"_shape".to_vec()),
                Value::Tuple(vec![Value::Int(5), Value::Int(3)]),
            )]))),
            list_items: Vec::new(),
            dict_items: Vec::new(),
        };
        assert_eq!(
            obj.get("_shape"),
            Some(&Value::Tuple(vec![Value::Int(5), Value::Int(3)]))
        );
    }
}
