# 🛢️ Oiler

```
█████ ████ ██   ████▛ ████▙
██ ██  ██  ██   ██    ██ ██
██ ██  ██  ██   ████  ████▛
██ ██  ██  ██   ██    ██ █▖
█████ ████ ████ ████▙ ██ ██
```

A lightweight and ergonomic **linear algebra library** for Rust.

---

## ✨ Features

- 📐 **Matrices**
  - Gaussian elimination
  - CR decomposition
  - More coming soon...

- 📏 **Vectors**
  - Dot product
  - Basic operations

---

## 🚀 Quick Start

Add Oiler to your project:

```bash
cargo add oiler
```

Or install the latest version directly from GitHub:

```bash
cargo add --git https://github.com/NewDawn0/oiler.git
```

---

## 🧪 Example

```rust
use oiler::linalg::prelude::*;

fn main() {
    let a = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let b = Vector::new([5.0, 11.0]);

    let x = a.gaussian_eliminate(&b);

    println!("Solution: {}", x);
}
```

---

## 📦 Roadmap

- [ ] More matrix decompositions
- [ ] Sparse matrix support
- [ ] Performance optimizations
- [ ] Documentation improvements

---

## 📄 License

Licensed under the **MIT License**.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or start a discussion.

---

> Built with ❤️ in Rust
