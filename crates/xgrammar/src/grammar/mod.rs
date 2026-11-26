// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar AST, builder, EBNF lexer/parser, normalization functors, and printer
// (pure-Rust port of mlc-ai/xgrammar's grammar subsystem).

pub mod builder;
pub mod data;
pub mod expr;
pub mod functor;
pub mod lexer;
pub mod parser;
pub mod printer;

pub use builder::GrammarBuilder;
pub use data::{GrammarData, Rule, TagDispatch};
pub use expr::{GrammarExpr, GrammarExprType};
pub use functor::{
    GrammarConcat, GrammarFsmBuilder, GrammarFsmHasher, GrammarNormalizer, GrammarOptimizer,
    GrammarUnion,
};
pub use parser::{ParseError, parse_ebnf, parse_ebnf_default};
pub use printer::{GrammarPrinter, print_grammar};
