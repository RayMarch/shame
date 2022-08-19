use quote::ToTokens;
use syn::fold::{self, Fold};
use syn::{parse_quote, Expr, Local, Pat, PatIdent, Stmt};

//quick and dirty thrown together modification of syn example code
//TODO: clean up more
//https://github.com/dtolnay/syn/tree/master/examples/trace-var

pub struct State;

fn extract_ident_from_pat(p: &Pat) -> Option<&PatIdent> {
    let inner = match p {
        Pat::Ident(ident) => return Some(ident),
        Pat::Slice(_) => return None,
        Pat::Struct(_) => return None,
        Pat::Tuple(_) => return None,
        Pat::TupleStruct(_) => return None,
        Pat::Type(pat_type) => &*pat_type.pat,
        _ => return None,
    };
    extract_ident_from_pat(inner)
}

impl State {
    fn should_print_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(p) => match (
                p.path.leading_colon,
                p.path.segments.len(),
                p.path.segments.first(),
            ) {
                (None, 1, Some(first)) if first.arguments.is_empty() => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn assign_and_print(&mut self, left: Expr, op: &dyn ToTokens, right: Expr) -> Expr {
        let right = fold::fold_expr(self, right);
        parse_quote!({
            #left #op #right;
            //println!(concat!(stringify!(#left), " = {:?}"), #left);
        })
    }

    fn let_and_print(&mut self, local: Local) -> Stmt {
        let Local { pat, init, .. } = local;
        let init = self.fold_expr(*init.unwrap().1);
        let ident = match extract_ident_from_pat(&pat) {
            Some(p) => &p.ident,
            _ => unreachable!(),
        };
        parse_quote! {
            let #pat = {
                #[allow(unused_mut)]
                let #pat = #init;
                shame_reexports::shame::keep_idents::TryKeepIdent(&#ident).store_ident(stringify!(#ident));
                //println!(concat!(stringify!(#ident), " = {:?}"), #ident);
                #ident
            };
        }
    }
}

impl Fold for State {
    fn fold_expr(&mut self, e: Expr) -> Expr {
        match e {
            Expr::Assign(e) => {
                if self.should_print_expr(&e.left) {
                    self.assign_and_print(*e.left, &e.eq_token, *e.right)
                } else {
                    Expr::Assign(fold::fold_expr_assign(self, e))
                }
            }
            Expr::AssignOp(e) => {
                if self.should_print_expr(&e.left) {
                    self.assign_and_print(*e.left, &e.op, *e.right)
                } else {
                    Expr::AssignOp(fold::fold_expr_assign_op(self, e))
                }
            }
            _ => fold::fold_expr(self, e),
        }
    }

    fn fold_stmt(&mut self, s: Stmt) -> Stmt {
        match s {
            Stmt::Local(s) => {
                if s.init.is_some() && extract_ident_from_pat(&s.pat).is_some() {
                    self.let_and_print(s)
                } else {
                    Stmt::Local(fold::fold_local(self, s))
                }
            }
            _ => fold::fold_stmt(self, s),
        }
    }
}
