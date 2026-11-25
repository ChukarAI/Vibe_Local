//! Deterministic built-in micro-eval engine.
//!
//! This is NOT an LLM: it is a real, fully deterministic extractive reasoner
//! used (a) as the fallback chat backend when `llama-server` is not installed
//! and (b) as a measurable workload for pipeline/VibeMark validation. Every
//! response is computed from the prompt with no randomness, and throughput
//! numbers measured against it are labeled `engine = "builtin-microeval"`.
//!
//! Capabilities: arithmetic expression evaluation (shunting-yard), keyword
//! extraction, extractive sentence ranking and structured echo.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MicroEval {
    pub max_context_tokens: usize,
}

impl Default for MicroEval {
    fn default() -> Self {
        Self {
            max_context_tokens: 8192,
        }
    }
}

/// Split text into deterministic word/punctuation tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Dimension of the deterministic builtin embedding vectors.
pub const DET_EMBED_DIM: usize = 64;

/// Deterministic, L2-normalized bag-of-tokens embedding.
///
/// This is NOT a semantic model embedding: it is a stable hash projection
/// (FNV-1a token hashes scattered into `dim` buckets with signed updates),
/// used (a) as the builtin fallback for `POST /v1/embeddings` when no
/// llama-server is present and (b) as the offline verifier for the Vibe
/// Grid `embed_batch` workload. Identical input always yields identical
/// output on every platform, and the result is honestly labeled
/// `builtin-microeval` wherever it surfaces.
pub fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    assert!(dim > 0, "embedding dimension must be positive");
    let mut vec = vec![0f32; dim];
    let tokens = tokenize(&text.to_lowercase());
    if tokens.is_empty() {
        // Hash the raw text so even punctuation-only input is non-zero.
        scatter(&mut vec, vibe_local_core::util::fnv1a32(text.as_bytes()));
    } else {
        for tok in &tokens {
            scatter(&mut vec, vibe_local_core::util::fnv1a32(tok.as_bytes()));
        }
    }
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

/// Scatter one token hash into a vector: four signed bucket updates driven
/// by a splitmix64 expansion of the 32-bit hash (fully deterministic).
fn scatter(vec: &mut [f32], hash: u32) {
    let dim = vec.len() as u64;
    let mut state = u64::from(hash) ^ 0x9e37_79b9_7f4a_7c15;
    for _ in 0..4 {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let idx = (z % dim) as usize;
        if z & 0x8000_0000_0000_0000 != 0 {
            vec[idx] += 1.0;
        } else {
            vec[idx] -= 1.0;
        }
    }
}

/// Evaluate an arithmetic expression with + - * / % ^ and parentheses.
/// Returns None for anything that is not a pure arithmetic expression.
pub fn eval_arithmetic(expr: &str) -> Option<f64> {
    let tokens = tokenize(expr);
    if tokens.is_empty() {
        return None;
    }
    // Shunting-yard to RPN. Precedence is doubled so unary minus/plus can
    // sit between multiplicative and power operators: `-2^2` = -(2^2) and
    // `2^-3^2` = 2^(-(3^2)), matching mathematical convention.
    let mut output: Vec<String> = Vec::new();
    let mut ops: Vec<String> = Vec::new();
    let prec = |op: &str| match op {
        "+" | "-" => 2,
        "*" | "/" | "%" => 4,
        "u-" | "u+" => 5,
        "^" => 6,
        _ => 0,
    };
    let mut prev_was_operand = false;
    for tok in &tokens {
        let is_number = tok
            .chars()
            .next()
            .map(|c| c.is_ascii_digit() || c == '.')
            .unwrap_or(false)
            && tok.chars().all(|c| c.is_ascii_digit() || c == '.');
        if is_number {
            if prev_was_operand {
                return None; // "2 3" is not an expression
            }
            output.push(tok.clone());
            prev_was_operand = true;
        } else if tok == "(" {
            ops.push(tok.clone());
            prev_was_operand = false;
        } else if tok == ")" {
            let mut found = false;
            while let Some(op) = ops.pop() {
                if op == "(" {
                    found = true;
                    break;
                }
                output.push(op);
            }
            if !found {
                return None;
            }
            prev_was_operand = true;
        } else if matches!(tok.as_str(), "+" | "-" | "*" | "/" | "%" | "^") {
            // Unary minus/plus: prefix operators bind to the NEXT operand,
            // so they never pop pending operators (popping `^` before a
            // unary `-` would turn `2^-3` into `(2^0)-3`).
            if !prev_was_operand && (tok == "-" || tok == "+") {
                output.push("0".to_string());
                ops.push(format!("u{tok}"));
                prev_was_operand = false;
                continue;
            } else if !prev_was_operand {
                return None;
            }
            while let Some(top) = ops.last() {
                if top == "(" {
                    break;
                }
                let right_assoc = tok == "^";
                if prec(top) > prec(tok) || (prec(top) == prec(tok) && !right_assoc) {
                    output.push(ops.pop().unwrap());
                } else {
                    break;
                }
            }
            ops.push(tok.clone());
            prev_was_operand = false;
        } else {
            return None; // letters/other symbols -> not arithmetic
        }
    }
    if !prev_was_operand {
        return None;
    }
    while let Some(op) = ops.pop() {
        if op == "(" {
            return None;
        }
        output.push(op);
    }
    // Evaluate RPN.
    let mut stack: Vec<f64> = Vec::new();
    for tok in &output {
        if let Ok(n) = tok.parse::<f64>() {
            stack.push(n);
            continue;
        }
        let b = stack.pop()?;
        let a = stack.pop()?;
        let v = match tok.as_str() {
            "+" | "u+" => a + b,
            "-" | "u-" => a - b,
            "*" => a * b,
            "/" => {
                if b == 0.0 {
                    return None;
                }
                a / b
            }
            "%" => {
                if b == 0.0 {
                    return None;
                }
                a % b
            }
            "^" => a.powf(b),
            _ => return None,
        };
        if !v.is_finite() {
            return None;
        }
        stack.push(v);
    }
    if stack.len() == 1 {
        Some(stack[0])
    } else {
        None
    }
}

fn strip_math_framing(prompt: &str) -> &str {
    let t = prompt.trim();
    let lower = t.to_lowercase();
    for prefix in ["what is ", "calculate ", "compute ", "solve ", "eval "] {
        // Prefixes are ASCII, so a lowercase match implies the same byte
        // prefix in the original string (safe slicing).
        if lower.starts_with(prefix) {
            return &t[prefix.len()..];
        }
    }
    t
}

fn stopwords() -> &'static [&'static str] {
    &[
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "to", "of",
        "in", "on", "for", "with", "as", "by", "at", "it", "this", "that", "these", "those", "i",
        "you", "he", "she", "we", "they", "me", "my", "your", "do", "does", "did", "can", "could",
        "will", "would", "should", "please", "what", "which", "who", "how", "why", "when", "where",
    ]
}

/// Deterministic keyword extraction ranked by frequency then first position.
pub fn keywords(text: &str, limit: usize) -> Vec<String> {
    let tokens = tokenize(text);
    let mut freq: HashMap<String, (usize, usize)> = HashMap::new();
    for (i, tok) in tokens.iter().enumerate() {
        let lower = tok.to_lowercase();
        if lower.len() < 3
            || stopwords().contains(&lower.as_str())
            || !lower.chars().next().unwrap().is_alphabetic()
        {
            continue;
        }
        let e = freq.entry(lower).or_insert((0, i));
        e.0 += 1;
    }
    let mut ranked: Vec<(String, usize, usize)> =
        freq.into_iter().map(|(k, (f, p))| (k, f, p)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
    ranked.into_iter().take(limit).map(|(k, _, _)| k).collect()
}

fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let t = current.trim().to_string();
            if !t.is_empty() {
                out.push(t);
            }
            current.clear();
        }
    }
    let t = current.trim().to_string();
    if !t.is_empty() {
        out.push(t);
    }
    out
}

impl MicroEval {
    /// Ingest context (real work: tokenization + indexing), returns token count.
    pub fn ingest(&self, prompt: &str) -> usize {
        let tokens = tokenize(prompt);
        tokens.len().min(self.max_context_tokens)
    }

    /// Generate a deterministic response as a token vector.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Vec<String> {
        let text = self.generate_text(prompt);
        tokenize(&text).into_iter().take(max_tokens).collect()
    }

    /// Full deterministic response text.
    pub fn generate_text(&self, prompt: &str) -> String {
        let trimmed = prompt.trim();
        if trimmed.is_empty() {
            return "No input provided. Send a prompt to evaluate.".to_string();
        }
        // 1) Arithmetic path.
        let candidate = strip_math_framing(trimmed).trim_end_matches(['?', '=', '.']);
        if let Some(value) = eval_arithmetic(candidate) {
            if value == value.trunc() && value.abs() < 1e15 {
                return format!("{} = {}", candidate.trim(), value as i64);
            }
            return format!("{} = {:.6}", candidate.trim(), value);
        }
        // 2) Extractive path: rank prompt sentences by keyword overlap with
        //    the final question sentence.
        let sents = sentences(trimmed);
        let kws = keywords(trimmed, 8);
        if sents.len() >= 3 && !kws.is_empty() {
            let question = sents.last().unwrap().to_lowercase();
            let mut scored: Vec<(usize, usize, &String)> = sents
                .iter()
                .enumerate()
                .take(sents.len() - 1)
                .map(|(i, s)| {
                    let lower = s.to_lowercase();
                    let score = kws
                        .iter()
                        .filter(|k| lower.contains(k.as_str()) && question.contains(k.as_str()))
                        .count();
                    (score, i, s)
                })
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            if scored[0].0 > 0 {
                let picked: Vec<&str> = scored.iter().take(2).map(|(_, _, s)| s.as_str()).collect();
                return format!(
                    "Extractive answer (deterministic micro-eval): {}",
                    picked.join(" ")
                );
            }
        }
        // 3) Structured echo path.
        let token_count = self.ingest(trimmed);
        let kw_list = if kws.is_empty() {
            "none".to_string()
        } else {
            kws.join(", ")
        };
        format!(
            "Micro-eval analysis: received {token_count} tokens. Key terms: {kw_list}. \
             This deterministic engine extracts and computes; install llama-server \
             (VIBE_LLAMA_SERVER) for full LLM inference."
        )
    }
}

/// Remove `<think>...</think>` blocks (including an unclosed trailing block).
pub fn strip_think(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_embedding_properties() {
        let a = deterministic_embedding("the quick brown fox", DET_EMBED_DIM);
        let b = deterministic_embedding("the quick brown fox", DET_EMBED_DIM);
        assert_eq!(a, b, "same input must embed identically");
        assert_eq!(a.len(), DET_EMBED_DIM);
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        let c = deterministic_embedding("completely different words here", DET_EMBED_DIM);
        assert_ne!(a, c);
        // Case-insensitive and dimension-configurable.
        assert_eq!(
            a,
            deterministic_embedding("THE QUICK BROWN FOX", DET_EMBED_DIM)
        );
        assert_eq!(deterministic_embedding("x", 8).len(), 8);
        // Punctuation-only input still produces a usable vector.
        let p = deterministic_embedding("!!!", DET_EMBED_DIM);
        assert!(p.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn arithmetic_correct() {
        assert_eq!(eval_arithmetic("2+3*4"), Some(14.0));
        assert_eq!(eval_arithmetic("(2+3)*4"), Some(20.0));
        assert_eq!(eval_arithmetic("10/4"), Some(2.5));
        assert_eq!(eval_arithmetic("2^10"), Some(1024.0));
        assert_eq!(eval_arithmetic("-5+3"), Some(-2.0));
        assert_eq!(eval_arithmetic("7%3"), Some(1.0));
        assert_eq!(eval_arithmetic("1/0"), None);
        assert_eq!(eval_arithmetic("hello world"), None);
        assert_eq!(eval_arithmetic("2 3"), None);
        assert_eq!(eval_arithmetic("(2+3"), None);
        assert_eq!(eval_arithmetic(""), None);
    }

    #[test]
    fn unary_after_operator() {
        // Regression: unary minus/plus must bind to the next operand, not
        // pop pending higher-precedence operators.
        assert_eq!(eval_arithmetic("2*-3"), Some(-6.0));
        assert_eq!(eval_arithmetic("2^-3"), Some(0.125));
        assert_eq!(eval_arithmetic("10/-2"), Some(-5.0));
        assert_eq!(eval_arithmetic("5--3"), Some(8.0));
        assert_eq!(eval_arithmetic("5+-3"), Some(2.0));
        assert_eq!(eval_arithmetic("2*-3^2"), Some(-18.0));
        assert_eq!(eval_arithmetic("-2^2"), Some(-4.0));
        assert_eq!(eval_arithmetic("2^-3^2"), Some(2f64.powf(-9.0)));
        assert_eq!(eval_arithmetic("(-2+3)*4"), Some(4.0));
        assert_eq!(eval_arithmetic("--3"), Some(3.0));
        assert_eq!(eval_arithmetic("2^+3"), Some(8.0));
        assert_eq!(eval_arithmetic("3+4*2/(1-5)^2"), Some(3.5));
    }

    #[test]
    fn math_prompt_path() {
        let e = MicroEval::default();
        assert_eq!(e.generate_text("What is 12 * 12?"), "12 * 12 = 144");
        assert_eq!(e.generate_text("calculate 100 / 8"), "100 / 8 = 12.500000");
    }

    #[test]
    fn determinism() {
        let e = MicroEval::default();
        let prompt = "The quick brown fox jumps over the lazy dog. Foxes are clever animals. What do foxes do?";
        let a = e.generate_text(prompt);
        let b = e.generate_text(prompt);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn extractive_ranking() {
        let e = MicroEval::default();
        let prompt = "Vulkan is a graphics API. The RX 580 supports Vulkan well. Banana split is a dessert. Which GPU API does the RX 580 support?";
        let out = e.generate_text(prompt);
        assert!(out.contains("RX 580 supports Vulkan"), "got: {out}");
    }

    #[test]
    fn token_generation_bounded() {
        let e = MicroEval::default();
        let tokens = e.generate("hello world this is a test prompt", 3);
        assert_eq!(tokens.len(), 3);
        assert!(e.ingest("a b c") == 3);
    }

    #[test]
    fn think_stripping() {
        assert_eq!(
            strip_think("<think>secret reasoning</think>Final answer."),
            "Final answer."
        );
        assert_eq!(strip_think("A<think>x</think>B<think>y</think>C"), "ABC");
        assert_eq!(strip_think("Before<think>unclosed"), "Before");
        assert_eq!(strip_think("no think here"), "no think here");
    }

    #[test]
    fn keywords_sane() {
        let kw = keywords("Rust rust RUST memory safety and the borrow checker", 3);
        assert_eq!(kw[0], "rust");
        assert!(!kw.contains(&"the".to_string()));
    }
}
