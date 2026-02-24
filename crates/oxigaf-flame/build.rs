/// Detect nightly Rust compiler and emit `cfg(nightly)`.
///
/// This allows the `simd` feature to gracefully degrade on stable Rust:
/// - On nightly: `#![feature(portable_simd)]` is enabled, SIMD code compiles.
/// - On stable: `simd` feature is accepted but SIMD code is compiled out.
///
/// This ensures `--all-features` works on both stable and nightly.
fn main() {
    // Register `nightly` as a known cfg to suppress unexpected_cfgs warnings
    println!("cargo::rustc-check-cfg=cfg(nightly)");

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if let Ok(output) = std::process::Command::new(&rustc).arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout);
        if version.contains("nightly") || version.contains("dev") {
            println!("cargo:rustc-cfg=nightly");
        }
    }
    // Re-run only if the compiler changes
    println!("cargo:rerun-if-env-changed=RUSTC");
}
