#[doc(hidden)]
#[macro_export]
macro_rules! impl_ops {
    (
        $(
            <$($G: ident: $B: path),*>
            $Add: ident $(! $op: ident:)? $add: ident ($lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty)
            -> $Out: ty: $implem: tt;
        )*
    ) => {
        $(
            impl<$($G: $B),*>
            $Add<$Rhs> for $Lhs {
                type Output = $Out;
                #[track_caller]
                fn $add(self: $Lhs, $rhs: $Rhs) -> $Out {
                    $(let $op = $Add::$add;)?
                    let $lhs = self;
                    $implem
                }
            }
        )*
    };

    // added &mut for op_assign impls
    (
        $(
            <$($G: ident: $B: path),*>
            $AddAssign: ident $(! $op: ident:)? $add_assign: ident($lhs: ident: &mut $Lhs: ty, $rhs: ident: $Rhs: ty)
            : $implem: tt;
        )*
    ) => {
        $(
            impl<$($G: $B),*>
            $AddAssign<$Rhs> for $Lhs {
                #[track_caller]
                fn $add_assign(self: &mut $Lhs, $rhs: $Rhs) {
                    $(let $op = $AddAssign::$add_assign;)?
                    let $lhs = self;
                    $implem
                }
            }
        )*
    };

    (
        + also implements lhs <-> rhs swapped
        $(
            <$($G: ident: $B: path),*>
            $Add: ident $(! $op: ident:)? $add: ident($lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty)
            -> $Out: ty: $imp: tt;
        )*
    ) => {
        //implements the lhs x rhs, and the rhs x lhs version
        impl_ops!{$(<$($G: $B),*> $Add $(! $op:)? $add($lhs: $Lhs, $rhs: $Rhs) -> $Out: $imp;)*}
        impl_ops!{$(<$($G: $B),*> $Add $(! $op:)? $add($lhs: $Rhs, $rhs: $Lhs) -> $Out: $imp;)*}
    };

    (
        $(
            <$($G: ident: $B: path),*>
            $Not: ident $(! $op: ident:)? $not: ident($lhs: ident: $Lhs: ty)
            -> $Out: ty: $implem: tt;
        )*
    ) => {
        $(
            impl<$($G: $B),*>
            $Not for $Lhs {
                type Output = $Out;
                #[track_caller]
                fn $not(self: $Lhs) -> $Out {
                    $(let $op = $Not::$not;)?
                    let $lhs = self;
                    $implem
                }
            }
        )*
    };
}
