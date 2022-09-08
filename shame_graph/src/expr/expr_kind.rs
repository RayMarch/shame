
use std::fmt::Display;

use super::*;
use enum_properties::*;

#[derive(Debug, Clone)]
pub enum ExprKind {
    Copy{comment: &'static str}, //useful for getting rid of lvalues, behaves like a "copy constructor"
    GlobalInterface(Ty),
    Literal(Literal),
    Constructor(Constructor),
    Swizzle(Swizzle),
    FieldSelect(IdentSlot),
    Operator(Operator),
    BuiltinFn(BuiltinFn),
    BuiltinVar(BuiltinVar),
}

impl ExprKind {
    pub fn is_mutating_arg_with_index(&self, index: usize) -> bool {
        match index {
            0 => matches!(self, ExprKind::Operator(op) if op.lhs_lvalue),
            _ => false,
        }
    }

    pub fn is_mutating_any_arg(&self) -> bool {
        matches!(self, ExprKind::Operator(op) if op.lhs_lvalue)
    }
}

pub struct LiteralProps {
    pub dtype: DType,
}

enum_properties! {
    #[derive(Debug, Clone, Copy)]
    pub enum Literal: LiteralProps {
        Bool {dtype: DType::Bool} (bool), //true, false
        F32  {dtype: DType::F32 } (f32 ), //1.0
        F64  {dtype: DType::F64 } (f64 ), //1.0lf
        I32  {dtype: DType::I32 } (i32 ), //1
        U32  {dtype: DType::U32 } (u32 ), //1u
    }
}

#[derive(Debug, Clone)]
pub enum Constructor {
    Tensor(Tensor),
    Struct(Struct),
    Array(Array),
    TextureCombinedSampler(TexDtypeDimensionality),
}

#[derive(Debug, Clone, Copy)]
pub enum Swizzle {
    GetVec4  ([u8; 4]), // GetVec4([3, 0, 2, 2]) means vector.wxzz
    GetVec3  ([u8; 3]),
    GetVec2  ([u8; 2]),
    GetScalar([u8; 1]),
}

impl Swizzle {
    pub fn inner_slice(&self) -> &[u8] {
        use Swizzle::*;
        match self {
            GetVec4  (x) => x,
            GetVec3  (x) => x,
            GetVec2  (x) => x,
            GetScalar(x) => x,
        }
    }
}

pub struct OperatorProps {
    pub argc: usize,
    pub glsl_prec: u32,
    pub glsl_assoc: Associativity,
    pub glsl_str: &'static str,
    pub lhs_lvalue: bool, //whether lhs must be an lvalue
}

#[derive(PartialEq, Eq)]
pub enum Associativity {
    LeftToRight,
    RightToLeft
}
use Associativity::*;

enum_properties! {

    /// see https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.pdf 
    /// chapter 5.1 operators
    #[derive(Debug, Clone)]
    pub enum Operator: OperatorProps {
        //1 parenthetical grouping (implicitly generated)
        //2
        Subscript       {argc: 2, glsl_prec:  2, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "[]"},
        PostfixInc      {argc: 1, glsl_prec:  2, glsl_assoc: LeftToRight, lhs_lvalue: true, glsl_str: "++"},
        PostfixDec      {argc: 1, glsl_prec:  2, glsl_assoc: LeftToRight, lhs_lvalue: true, glsl_str: "--"},
        //3
        PrefixInc       {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "++"},
        PrefixDec       {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "--"},
        Positive        {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: false, glsl_str: "+"},
        Negative        {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: false, glsl_str: "-"},
        BitNot          {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: false, glsl_str: "~"},
        Not             {argc: 1, glsl_prec:  3, glsl_assoc: RightToLeft, lhs_lvalue: false, glsl_str: "!"},
        //4
        Mul             {argc: 2, glsl_prec:  4, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "*"},
        Div             {argc: 2, glsl_prec:  4, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "/"},
        Rem             {argc: 2, glsl_prec:  4, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "%"},
        //5
        Add             {argc: 2, glsl_prec:  5, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "+"},
        Sub             {argc: 2, glsl_prec:  5, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "-"},
        //6
        ShiftL          {argc: 2, glsl_prec:  6, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "<<"},
        ShiftR          {argc: 2, glsl_prec:  6, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: ">>"},
        //7
        Less            {argc: 2, glsl_prec:  7, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "<"},
        Greater         {argc: 2, glsl_prec:  7, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: ">"},
        LessEqual       {argc: 2, glsl_prec:  7, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "<="},
        GreaterEqual    {argc: 2, glsl_prec:  7, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: ">="},
        //8
        Equal           {argc: 2, glsl_prec:  8, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "=="},
        NotEqual        {argc: 2, glsl_prec:  8, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "!="},
        //9
        BitAnd          {argc: 2, glsl_prec:  9, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "&"},
        //10
        BitXor          {argc: 2, glsl_prec: 10, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "^"},
        //11
        BitOr           {argc: 2, glsl_prec: 11, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "|"},
        //12
        And             {argc: 2, glsl_prec: 12, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "&&"},
        //13
        LogicalXor      {argc: 2, glsl_prec: 13, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "^^"},
        //14
        Or              {argc: 2, glsl_prec: 14, glsl_assoc: LeftToRight, lhs_lvalue: false, glsl_str: "||"},
        //15
        TernaryIf       {argc: 3, glsl_prec: 15, glsl_assoc: RightToLeft, lhs_lvalue: false, glsl_str: "?"},
        //16
        Assign          {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "="},
        AddAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "+="},
        SubAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "-="},
        MulAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "*="},
        DivAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "/="},
        RemAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "%="},
        ShiftLAssign    {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "<<="},
        ShiftRAssign    {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: ">>="},
        AndAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "&="},
        XorAssign       {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "^="},
        OrAssign        {argc: 2, glsl_prec: 16, glsl_assoc: RightToLeft, lhs_lvalue: true, glsl_str: "|="},
    }
}

pub struct FuncProps {
    pub glsl_str: &'static str,
}

enum_properties! {
    #[derive(Debug, Clone)]
    pub enum BuiltinFn: FuncProps {
        Equal            {glsl_str: "equal"},
        NotEqual         {glsl_str: "notEqual"},
        LessThan         {glsl_str: "lessThan"},
        LessThanEqual    {glsl_str: "lessThanEqual"},
        GreaterThan      {glsl_str: "greaterThan"},
        GreaterThanEqual {glsl_str: "greaterThanEqual"},
        Dfdx        {glsl_str: "dFdx"},
        Dfdy        {glsl_str: "dFdy"},
        DfdxCoarse  {glsl_str: "dFdxCoarse"},
        DfdyCoarse  {glsl_str: "dFdyCoarse"},      
        DfdxFine    {glsl_str: "dFdxFine"},
        DfdyFine    {glsl_str: "dFdyFine"},  
        Round       {glsl_str: "round"},
        Ceil        {glsl_str: "ceil"},
        Floor       {glsl_str: "floor"},
        All         {glsl_str: "all"},
        Any         {glsl_str: "any"},
        Not         {glsl_str: "not"},
        Atan        {glsl_str: "atan"},
        Sign        {glsl_str: "sign"},
        Pow         {glsl_str: "pow"},
        Sqrt        {glsl_str: "sqrt"},
        Fract       {glsl_str: "fract"},
        Length      {glsl_str: "length"},
        Sin         {glsl_str: "sin"},
        Cos         {glsl_str: "cos"},
        Dot         {glsl_str: "dot"},
        Cross       {glsl_str: "cross"},
        Min         {glsl_str: "min"},
        Max         {glsl_str: "max"},
        Clamp       {glsl_str: "clamp"}, //x, min, max
        Smoothstep  {glsl_str: "smoothstep"}, //x, edge0, edge1
        Mix         {glsl_str: "mix"}, //x, y, alpha
        Abs         {glsl_str: "abs"},
        Normalize   {glsl_str: "normalize"},
        Texture     {glsl_str: "texture"},
    }
}

impl Display for ExprKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExprKind::*;
        match self {
            Copy{comment} => f.write_fmt(format_args!("copy ({})", comment)),
            GlobalInterface(x) => f.write_fmt(format_args!("global interface {}", x)),
            Literal(x) => x.fmt(f),
            Constructor(x) => x.fmt(f),
            Swizzle(x) => x.fmt(f),
            FieldSelect(_) => f.write_fmt(format_args!("field_select")),
            Operator(x) => x.fmt(f),
            BuiltinFn(x) => x.fmt(f),
            BuiltinVar(x) => x.fmt(f),
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} literal", self.dtype))
    }
}

impl Display for BuiltinFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.glsl_str)
    }
}

impl Display for Constructor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constructor::Tensor(x) => f.write_fmt(format_args!("{x} constructor")),
            Constructor::Struct(x) => f.write_fmt(format_args!("struct {x} constructor")), //TODO: write more detailed info
            Constructor::Array(x) => f.write_fmt(format_args!("{x} constructor")),
            Constructor::TextureCombinedSampler(x) => f.write_fmt(format_args!("sampler{x} constructor"))
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("operator '{}'", self.glsl_str))
    }
}

impl Display for Swizzle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("member access/swizzle")?;
        f.write_str("[")?;
        for (i, &c) in self.inner_slice().iter().enumerate() {
            if i > 0 {f.write_str(", ")?;}
            match c {
                0 => f.write_str("x")?,
                1 => f.write_str("y")?,
                2 => f.write_str("z")?,
                3 => f.write_str("w")?,
                x => f.write_fmt(format_args!("{}", x))?,
            };
        }
        f.write_str("]")
    }
}

impl Display for BuiltinVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BuiltinVar::VertexVar  (x) => x.glsl_str(),
            BuiltinVar::FragmentVar(x) => x.glsl_str(),
            BuiltinVar::ComputeVar (x) => x.glsl_str(),
        })
    }
}