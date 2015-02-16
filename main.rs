use std::char::CharExt;
#[derive(Debug, Clone, PartialEq)]
enum Token {
	Ident(String),
	Number(i32),
	Operator(String),
	StringLiteral(String),
	Semicolon,
	OpenBrace,
	CloseBrace,
	OpenParen,
	CloseParen,
	FunctionKeyword,
	LetKeyword,
	AsKeyword,
	StructKeyword,
	VoidKeyword,
	IfKeyword,
	ElseKeyword,
	MatchKeyword,
	ForKeyword,
	WhileKeyword,
	LoopKeyword,
	Empty,
	EndOfFile
}

macro_rules! with(
	($e:expr => $d:pat) => (match $e {
		$d => (),
		_ => panic!("{} expected.", stringify!($d)),
	});
	($e:expr => $d:pat => $ret:expr) => (match $e {
		$d => $ret,
		_ => panic!("{} expected.", stringify!($d)),
	});
	($e:expr => $d:pat = if $ret:expr) => (match $e {
		$d if $ret => (),
		_ => panic!("{} expected.", stringify!($d)),
	})
);
fn get_tokens(data: &str) -> Vec<Token> {
	let mut tokens = vec![Token::Empty];
	for c in data.chars() {
		let current = tokens.pop().unwrap();
		match current  {
			Token::Empty => tokens.push(match c {
				c if c.is_alphabetic() => Token::Ident(c.to_string()),
				c if c.is_whitespace() => Token::Empty,
				c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
				c if c == '"' => Token::StringLiteral("".to_string()),
				c => Token::Operator(c.to_string())
			}),
			Token::StringLiteral(lit) => match c {
					'"' => {
						tokens.push(Token::StringLiteral(lit));
						tokens.push(Token::Empty)
					},
					_ => tokens.push(Token::StringLiteral(lit + c.to_string().as_slice()))
			},
			Token::Ident(id) => match c {
				c if c.is_alphabetic() => tokens.push(Token::Ident(id + c.to_string().as_slice())),
				c => {
					tokens.push(match id.as_slice() {
						"if" => Token::IfKeyword,
						"function" => Token::FunctionKeyword,
						"let" => Token::LetKeyword,
						"as" => Token::AsKeyword,
						"struct" => Token::StructKeyword,
						"void" => Token::VoidKeyword,
						"else" => Token::ElseKeyword,
						"match" => Token::MatchKeyword,
						"for" => Token::ForKeyword,
						"while" => Token::WhileKeyword,
						"loop" => Token::LoopKeyword,
						f => Token::Ident(f.to_string()),
					});
					tokens.push(match c {
						c if c.is_alphabetic() => unreachable!("ERROR: Shouldn't be alphabetic for Ident end"),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number('c'.to_digit(10).unwrap() as i32), 
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					}); 
				} 
			},
			Token::Number(x) => match c {
				c if c.is_numeric() => tokens.push(Token::Number(c.to_digit(10).unwrap() as i32 + 10 * x)),
				c => {
					tokens.push(Token::Number(x));
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => unreachable!("ERROR: Shouldn't be numeric for Number end."), 
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					}); 	
				}
			},
			Token::Operator(id) => match id.as_slice() {
				"(" => {
					tokens.push(Token::OpenParen);
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					})
				},
				"{" => {
					tokens.push(Token::OpenBrace);
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					})
				},
				")" => {
					tokens.push(Token::CloseParen);
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					})
				},
				"}" => {
					tokens.push(Token::CloseBrace);
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					})
				}
				";" => {
					tokens.push(Token::Semicolon);
					tokens.push(match c {
						c if c.is_alphabetic() => Token::Ident(c.to_string()),
						c if c.is_whitespace() => Token::Empty,
						c if c.is_numeric() => Token::Number(c.to_digit(10).unwrap() as i32),
						c if c == '"' => Token::StringLiteral("".to_string()),
						c => Token::Operator(c.to_string())
					})
				},
				_ => match c {
					c if c.is_alphabetic() => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::Ident(c.to_string()))
					},
					c if c.is_whitespace() => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::Empty)
					}
					c if c.is_numeric() => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::Number(c.to_digit(10).unwrap() as i32))
					},
					c if c == '"' => {
						tokens.push(Token::Operator(id));
						Token::StringLiteral("".to_string());
					},
					'(' => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::OpenParen);
						tokens.push(Token::Empty);
					},
					'{' => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::OpenBrace);
						tokens.push(Token::Empty);
					},
					')' => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::CloseParen);
						tokens.push(Token::Empty);
					},
					'}' => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::CloseBrace);
						tokens.push(Token::Empty);
					},
					';' => {
						tokens.push(Token::Operator(id));
						tokens.push(Token::Semicolon);
						tokens.push(Token::Empty);
					},
					c => tokens.push(Token::Operator(id + c.to_string().as_slice())),
				}
			},
			_ => ()
		}
	}
	tokens.push(Token::EndOfFile);
	let new_toks = tokens.into_iter().map(|tok| {
		let tok_copy = tok.clone();
		match tok_copy {
			Token::Operator(op_code) => {
				match op_code.as_slice() {
					"(" => Token::OpenParen,
					"{" => Token::OpenBrace,
					")" => Token::CloseParen,
					"}" => Token::CloseBrace,
					";" => Token::Semicolon,
					_ => Token::Operator(op_code)
				}
			},
			_ => tok
		}
	}).collect::<Vec<Token>>();
	new_toks
}
#[derive(Debug)]
enum Value {
    Variable(String),
    Int(i32),
    StringLiteral(String),
    Qualified(String, Box<Value>),
    FunctionCall(String, Vec<Expr>)
}
fn parse_value(tokens: &mut Vec<Token>) -> Value {
	let current = tokens.remove(0);
	match current {
		Token::StringLiteral(id) => Value::StringLiteral(id),
		Token::Number(i) => Value::Int(i),
		Token::Ident(id) => {
			let next = tokens[0].clone();
			match next {
				Token::Operator(ref period) if period.as_slice() == "." => {
					tokens.remove(0); // consume the period
					Value::Qualified(id, Box::new(parse_value(tokens)))
				},
				Token::OpenParen => {
					parse_function_call(id, tokens)
				}
				_ => Value::Variable(id)
			}
		},
		_ => panic!("Expected number literal or ident when parsing value.")


	}
}
fn parse_function_call(name: String, tokens: &mut Vec<Token>) -> Value {
	let mut vals: Vec<Expr> = Vec::new();
	with!(tokens.remove(0) => Token::OpenParen);	
	while tokens[0] != Token::CloseParen {
		vals.push(parse_expr(tokens));
		if tokens[0] == Token::Operator(",".to_string()) {
			tokens.remove(0); // consume comma
		}

	}
	tokens.remove(0); // consume close paren
	Value::FunctionCall(name, vals)
}
#[derive(Debug)]
enum Expr {
    Value(Value),
    Binary(Box<Expr>, String, Box<Expr>),
    Unary(String, Box<Expr>),
    If(Box<Expr>, Box<Statement>, Option<Statement>)


}
#[derive(Debug)]
struct Operator {
    op: String,
    precedence: i32,
}
#[derive(Debug)]
enum Shunt {
   Value(Value),
   Expr(Expr),
   Op(String), 
}
fn parse_expr(tokens: &mut Vec<Token>) -> Expr {
	let current = tokens[0].clone();
	match current {
		Token::OpenParen => {
			with!(tokens.remove(0) => Token::OpenParen);
			let ret = parse_expr(tokens);
			with!(tokens.remove(0) => Token::CloseParen);
			ret
		},
		Token::Operator(op_code) => {
			with!(tokens.remove(0) => Token::Operator(_));
			Expr::Unary(op_code, Box::new(parse_expr(tokens)))
		},
		Token::IfKeyword => {
			with!(tokens.remove(0) => Token::IfKeyword);
			let cond = parse_expr(tokens);
			with!(tokens.remove(0) => Token::OpenBrace);
			let if_true = parse_statement(tokens);
			with!(tokens.remove(0) => Token::CloseBrace);
			let if_false = match tokens[0].clone() {
				Token::ElseKeyword => Some(parse_statement(tokens)),
				_ => None
			};
			Expr::If(Box::new(cond), Box::new(if_true), if_false)

		}
		_ => {
			let first_value = parse_value(tokens);
			let next = tokens[0].clone();
			match next {
				Token::Operator(ref op_code) if op_code.as_slice() != "," => {
					let mut op_stack: Vec<Operator> = Vec::new();
					let mut rpn_stack: Vec<Shunt> = Vec::new();
					rpn_stack.push(Shunt::Value(first_value));
					'shunt: loop {
						match tokens[0].clone() {
							Token::Operator(new_op_code) => {
								let op = Operator {op: new_op_code.clone(), precedence: match new_op_code.as_slice()  {
									"+" | "-" => 5,
									"/" | "*" => 10, 
									_ => 15
								}};
								while op_stack.len() > 0 && op.precedence < op_stack[op_stack.len() - 1].precedence {
									rpn_stack.push(Shunt::Op(op_stack.pop().unwrap().op));
								}
								op_stack.push(op);
							},
							_ => break 'shunt
						}
						tokens.remove(0);
						rpn_stack.push(match (tokens[0].clone())
						{
							Token::OpenParen => Shunt::Expr(parse_expr(tokens)),
							_ => Shunt::Value(parse_value(tokens))
						})

					}
					while op_stack.len() > 0 {
						rpn_stack.push(Shunt::Op(op_stack.pop().unwrap().op));
					}
					let mut first_expr = parse_rpn(&mut rpn_stack); 
					
					first_expr
					//Expr::Binary(Box::new(Expr::Value(first_value)), op_code.clone(), Box::new(parse_expr(tokens)))	
				},
				_ => Expr::Value(first_value)
			}
		}
	}
}
fn parse_rpn(rpn_stack: &mut Vec<Shunt>) -> Expr {
	let top = rpn_stack.pop().unwrap();
	match top {
		Shunt::Expr(expr) => expr,
		Shunt::Value(val) => Expr::Value(val),
		Shunt::Op(op) => {
			let rhs = Box::new(parse_rpn(rpn_stack));
			let lhs = Box::new(parse_rpn(rpn_stack));
			Expr::Binary(lhs, op, rhs)
		}
	}
}
#[derive(Debug)]
enum Statement {
    Block(Vec<Statement>),
    Expr(Box<Expr>),
    Let(Variable, Box<Expr>)
}
fn parse_statement(tokens: &mut Vec<Token>) -> Statement
{
	match tokens[0].clone() {
		Token::OpenBrace => {
			with!(tokens.remove(0) => Token::OpenBrace);
			let mut statements: Vec<Statement> = Vec::new();
			while tokens[0] != Token::CloseBrace {
				statements.push(parse_statement(tokens));
			}
			Statement::Block(statements)
		},
		Token::LetKeyword => {
			with!(tokens.remove(0) => Token::LetKeyword);
			let var = parse_var_declare(tokens);	
			with!(tokens.remove(0) => Token::Operator(ref x) = if x.as_slice() == "=");
			let var_value = parse_expr(tokens);
			with!(tokens.remove(0) => Token::Semicolon);
			Statement::Let(var, Box::new(var_value))
		},
		_ => {
			let expr = parse_expr(tokens);
			with!(tokens.remove(0) => Token::Semicolon);
			Statement::Expr(Box::new(expr))
		}
	}
}
#[derive(Debug)]
enum Type {
	Named(String),
	Optional(Box<Type>),
}
#[derive(Debug)]
struct Variable(String, Type);
#[derive(Debug)]
struct Function(String, Vec<Variable>, Statement, Type);
fn parse_function_definition(tokens: &mut Vec<Token>) -> Function {
	with!(tokens.remove(0) => Token::FunctionKeyword);
	let name = with!(tokens.remove(0) => Token::Ident(x) => x);
	with!(tokens.remove(0) => Token::OpenParen);
	let mut params: Vec<Variable> = Vec::new();
	'param_list: while tokens[0].clone() != Token::CloseParen {
		params.push(parse_var_declare(tokens));
		match tokens[0].clone() {
			Token::Operator(ref x) if x.as_slice() == "," => { tokens.remove(0); },
			_ => break 'param_list
		}
	}
	with!(tokens.remove(0) => Token::CloseParen);
	with!(tokens.remove(0) => Token::Operator(ref x) = if x.as_slice() == "->");
	let func_return_type = parse_type(tokens);
	with!(tokens[0].clone() => Token::OpenBrace);
	let code = parse_statement(tokens);


	Function(name, params, code, func_return_type)

}

fn parse_var_declare(tokens: &mut Vec<Token>) -> Variable
{
	let var_name = with!(tokens.remove(0) => Token::Ident(x) => x);
	with!(tokens.remove(0) => Token::AsKeyword);
	let var_type = parse_type(tokens);

	Variable(var_name, var_type)
}
fn parse_type(tokens: &mut Vec<Token>) -> Type
{
	let type_name = with!(tokens.remove(0) => Token::Ident(x) => x);
	match tokens[0].clone() {
		Token::Operator(ref x) if x.as_slice() == "?" => {
			with!(tokens.remove(0) => Token::Operator(ref x) = if x.as_slice() == "?");
			Type::Optional(Box::new(Type::Named(type_name)))
		},
		_ => Type::Named(type_name)
	}
}

fn main() {
	let mut toks = get_tokens(r"
	function makeMoney(target as person, amount as int, donater as person?) -> int {
		let newBank as int = target.bank + amount;
		let oldBank as int = donater.bank - amount;
		newBank;
	}
	");
	for tok in toks.clone() {
		println!("{:?}", tok);
	}
	//let mut toks = get_tokens("let x as int? = 2 + y;");//"1 + y * f.x(3 + \"stringlit\") - 4;");
	let test_val = parse_function_definition(&mut toks);
	println!("{:?}", test_val);

}
