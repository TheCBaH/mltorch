(* The main module: same name as the library, so dune generates no alias module
   and this file IS [Expr]. See expr.mli for the public contract, and
   .ai/native_expr_refactoring_design.md for the language itself.

   ---- why so much lives in one file ----

   The value/boolean/reduction types are mutually recursive, and three smaller
   layouts were tried and rejected before this one:

     - separate bool.mli / value.mli units are a dune CYCLE
       (Bool.intf -> Value.intf -> Bool.intf);
     - a [module rec] whose members ALIAS a private AST module's recursive types
       fails ascription: mid-check OCaml has no [Bool.t = ..bool_expr] equation
       yet, and reordering only moves the error;
     - sequential facades over such a module hit [Unbound module Value], since
       whichever is written first names the other's type.

   So the public modules own their types directly, in one [module rec] group.
   Dune then forbids a wrapped library's subordinate modules from naming its
   main module, which is why [Index], [Intrinsic], [Builder], [Rewrite], [Fold],
   [Check], [Eval] and [Pp] are sections here rather than files: they all depend
   on those types. That also keeps the raw constructors reachable for [Rewrite],
   which must rebuild structure WITHOUT going through the smart constructors --
   those fold, and a structure-preserving rewrite must not silently change
   syntax.

   Only the genuinely acyclic leaves stay separate compilation units: [Axis],
   [Role], [Coord], [Source] (and later [Max_op], and private [Checked]).

   ---- declaration order is load-bearing ----

   OCaml resolves modules in source order even for types outside the cycle.
   [Bool.Index_eq], [Reduction.lo]/[hi], [Value.Value_of_index]/[Load] all carry
   an [Index.t] and [Value.Intrinsic] carries an [Intrinsic.t], so those modules
   must precede the recursive group or it fails with [Unbound module Index]. *)

(* ---- leaf re-exports ------------------------------------------------------

   An explicit main module means dune synthesises NO aliases, so without these
   lines [Expr.Axis] simply does not exist. They must appear in expr.mli too.
   [Max_op] joins them when the unit itself moves here; [Checked] never does --
   it stays private. *)

module Axis = Axis
module Role = Role
module Coord = Coord
module Source = Source
module Max_op = Max_op

(* ---- reducer variables ---------------------------------------------------- *)

module Reduce_var = struct
  type t = int

  let compare = Int.compare
  let equal = Int.equal
  let hash n = n

  (* Diagnostics only. The printed name a reader sees comes from [Pp], which
     assigns r0, r1, ... in LEXICAL order, so output never depends on which
     internal ordinals a supply happened to hand out. *)
  let pp fmt n = Fmt.pf fmt "#%d" n

  module Ord = struct
    type nonrec t = t

    let compare = compare
  end

  module Map = Map.Make (Ord)
  module Set = Set.Make (Ord)
end

(* ---- index expressions ---------------------------------------------------- *)

module Index = struct
  (* Concrete here, [private] in expr.mli: consumers pattern-match but cannot
     construct, so a non-positive divisor or a delta smuggled into a position
     cannot enter the AST except through the smart constructors below. *)
  type _ t =
    | Output : Axis.t -> Role.Position.t t
    | Reduce : Reduce_var.t -> Role.Position.t t
    | Zero : Role.Position.t t
    | Const : int -> Role.Delta.t t
    | Of_position : Role.Position.t t -> Role.Delta.t t
    | Add : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
    | Scale : int * Role.Delta.t t -> Role.Delta.t t
    | Floor_div_pos : Role.Delta.t t * int -> Role.Delta.t t
    | Ceil_div_pos : Role.Delta.t t * int -> Role.Delta.t t
    | Min : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
    | Max : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
    | Clamp_low : Role.Delta.t t -> Role.Position.t t
      (* establishes >= 0 by its own semantics *)
    | Assume_position : Role.Delta.t t -> Role.Position.t t
  (* records a CLAIM, proves nothing; [Fold] can locate every site *)

  type error = [ `Non_positive_divisor of int ]

  let pp_error fmt : [< error ] -> unit = function
    | `Non_positive_divisor d -> Fmt.pf fmt "divisor must be > 0, got %d" d

  let output a = Output a
  let reduce v = Reduce v
  let zero = Zero
  let const n = Const n
  let of_position i = Of_position i
  let add a b = Add (a, b)
  let scale k a = Scale (k, a)
  let min a b = Min (a, b)
  let max a b = Max (a, b)
  let clamp_low a = Clamp_low a
  let assume_position a = Assume_position a

  (* Result-returning because the divisor arrives as a raw [int]: a zero or
     negative one must not reach the AST at all. The [d = 1] fold is the one
     identity kept from the start -- [Symbolic] already performs it, so dropping
     it would move goldens. Every other permitted fold is deferred, to keep the
     migration's golden diff attributable to a single cause. *)
  let floor_div_pos a d =
    if d <= 0 then Core.fail (`Non_positive_divisor d)
    else if d = 1 then Core.return a
    else Core.return (Floor_div_pos (a, d))

  let ceil_div_pos a d =
    if d <= 0 then Core.fail (`Non_positive_divisor d)
    else if d = 1 then Core.return a
    else Core.return (Ceil_div_pos (a, d))
end

(* ---- intrinsics ----------------------------------------------------------- *)

module Intrinsic = struct
  module Window = struct
    (* Half-open, already clipped to the input extents. Its own module with the
       type named [t], per the repository record convention. *)
    type t = { hlo : int; hhi : int; wlo : int; whi : int }
  end

  module Max_pool = struct
    type result = Value | Index

    (* Named rather than inlined at the printer, alongside [Value.unary_name]
       and [Value.binary_sym]: the spelling is part of the rendered form expect
       tests pin, so it belongs with the type it describes. *)
    let result_name = function Value -> "value" | Index -> "index"

    (* Stores only expression-library values: a [Source.t] and plain ints, never
       a tensor signature or an [Op_config] type. [in_h]/[in_w] are carried
       explicitly because the old form recovered them from the embedded
       [Tensor_sig.t], and an opaque source cannot. They come from
       [SEMANTICS.max_pool2d]'s [~x_shape], which the current [Symbolic] ignores
       precisely because it stashed the signature instead. *)
    type t = {
      source : Source.t;
      in_h : int;
      in_w : int;
      kernel_h : int;
      kernel_w : int;
      stride_h : int;
      stride_w : int;
      pad_h : int;
      pad_w : int;
      out : Role.Position.t Index.t Coord.t;
      result : result;
    }
  end

  type t = Max_pool of Max_pool.t
  type error = [ Checked.error | `Bad_geometry of string * int ]

  let pp_error fmt : [< error ] -> unit = function
    | #Checked.error as e -> Checked.pp_error fmt e
    | `Bad_geometry (what, n) -> Fmt.pf fmt "%s must be valid, got %d" what n

  (* Result-returning, and the ONLY way to build a descriptor: [Check] does not
     revalidate this later, because a smart constructor makes the invalid state
     unconstructable through the public API and a rule no test can turn red is
     not worth carrying. The native adapter's safety therefore rests on its own
     typed inputs ([Op_config.Pos.t], [Nonneg.t], [Dim.extent Dim.t]) crossing
     here already validated. *)
  let max_pool ~source ~in_h ~in_w ~kernel_h ~kernel_w ~stride_h ~stride_w
      ~pad_h ~pad_w ~out ~result =
    let positive what n =
      if n >= 1 then Core.return n else Core.fail (`Bad_geometry (what, n))
    in
    let nonneg what n =
      if n >= 0 then Core.return n else Core.fail (`Bad_geometry (what, n))
    in
    let open Core.Syntax in
    let* in_h = positive "in_h" in_h in
    let* in_w = positive "in_w" in_w in
    let* kernel_h = positive "kernel_h" kernel_h in
    let* kernel_w = positive "kernel_w" kernel_w in
    let* stride_h = positive "stride_h" stride_h in
    let* stride_w = positive "stride_w" stride_w in
    let* pad_h = nonneg "pad_h" pad_h in
    let+ pad_w = nonneg "pad_w" pad_w in
    Max_pool
      {
        Max_pool.source;
        in_h;
        in_w;
        kernel_h;
        kernel_w;
        stride_h;
        stride_w;
        pad_h;
        pad_w;
        out;
        result;
      }

  (* The window an output position reads, half-open and already clipped to the
     input extents.

     Public and result-returning because there are TWO interpreters. Native's
     [Ground_eval] expands the same stencil independently, and as a consumer it
     cannot reach the library-private [Checked] -- so if only [Eval] went through
     checked arithmetic here, the verifier's copy would stay unchecked and the
     two would drift. That is the same failure mode [Max_op] exists to prevent.

     Validating the descriptor's fields is not enough on its own: [out_h * sh]
     is a product of individually in-range factors, and an aggregate needs its
     own bound. *)
  let window (Max_pool d) ~out_h ~out_w =
    let open Core.Syntax in
    let axis out stride pad kernel extent =
      let* base = Checked.mul out stride in
      let* base = Checked.sub base pad in
      let+ top = Checked.add base kernel in
      (Stdlib.max 0 base, Stdlib.min extent top)
    in
    let* hlo, hhi =
      axis out_h d.Max_pool.stride_h d.Max_pool.pad_h d.Max_pool.kernel_h
        d.Max_pool.in_h
    in
    let+ wlo, whi =
      axis out_w d.Max_pool.stride_w d.Max_pool.pad_w d.Max_pool.kernel_w
        d.Max_pool.in_w
    in
    { Window.hlo; hhi; wlo; whi }

  (* The flattened input position a max-pool index result reports. Checked for
     the same reason: [ih * in_w] is an aggregate. *)
  let flat_index (Max_pool d) ~ih ~iw =
    let open Core.Syntax in
    let* row = Checked.mul ih d.Max_pool.in_w in
    Checked.add row iw
end

(* ---- the recursive core --------------------------------------------------- *)

module rec Bool : sig
  (* Not a general logic: it exists to guard [Value.select], which is why there
     is no [Bool] value constructor. Add a predicate only when an operation
     needs one. *)
  type t =
    | Value_lt of Value.t * Value.t
    | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t

  val value_lt : Value.t -> Value.t -> t
  val index_eq : Role.Delta.t Index.t -> Role.Delta.t Index.t -> t
end = struct
  type t =
    | Value_lt of Value.t * Value.t
    | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t

  let value_lt a b = Value_lt (a, b)
  let index_eq a b = Index_eq (a, b)
end

and Reduction : sig
  type kind = Sum | Max

  (* [var] is in scope only in [body]. Bounds may mention enclosing reducers but
     not [var] itself. The denotation is an ORDERED half-open left fold, so
     rewriting must not treat it as an unordered collection. *)
  type t = {
    kind : kind;
    var : Reduce_var.t;
    lo : Role.Position.t Index.t;
    hi : Role.Delta.t Index.t;
    body : Value.t;
  }

  val kind_name : kind -> string
end = struct
  type kind = Sum | Max

  type t = {
    kind : kind;
    var : Reduce_var.t;
    lo : Role.Position.t Index.t;
    hi : Role.Delta.t Index.t;
    body : Value.t;
  }

  (* [max_reduce], not [max]: it is the generic ordered reduction, distinct from
     the max-pool intrinsic, and the two must stay distinguishable in printed
     output. Carries the name the representation being replaced used, where this
     was [reduction_kind_name]. *)
  let kind_name = function Sum -> "sum" | Max -> "max_reduce"
end

and Value : sig
  type binary_op = Add | Sub | Mul | Div
  type unary_op = Exp | Sqrt

  type t =
    | Const of float
    | Binary of binary_op * t * t
    | Unary of unary_op * t
    | Select of Bool.t * t * t
    | Value_of_index of Role.Delta.t Index.t
    | Load of Source.t * Role.Position.t Index.t Coord.t
    | Round_f32 of t
    | Reduce of Reduction.t
    | Intrinsic of Intrinsic.t

  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val exp : t -> t
  val sqrt : t -> t
  val select : Bool.t -> t -> t -> t
  val value_of_index : Role.Delta.t Index.t -> t
  val load : Source.t -> Role.Position.t Index.t Coord.t -> t
  val round_f32 : t -> t
  val intrinsic : Intrinsic.t -> t

  val reduce : Reduction.t -> t
  (** Prefer [Builder.reduction], which allocates the variable and scopes the
      body; this is the raw wrapper it is built from. *)

  val apply_binary : binary_op -> float -> float -> float
  val apply_unary : unary_op -> float -> float
  val binary_sym : binary_op -> string
  val unary_name : unary_op -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val hash : t -> int
end = struct
  type binary_op = Add | Sub | Mul | Div
  type unary_op = Exp | Sqrt

  type t =
    | Const of float
    | Binary of binary_op * t * t
    | Unary of unary_op * t
    | Select of Bool.t * t * t
    | Value_of_index of Role.Delta.t Index.t
    | Load of Source.t * Role.Position.t Index.t Coord.t
    | Round_f32 of t
    | Reduce of Reduction.t
    | Intrinsic of Intrinsic.t

  let const x = Const x
  let add a b = Binary (Add, a, b)
  let sub a b = Binary (Sub, a, b)
  let mul a b = Binary (Mul, a, b)
  let div a b = Binary (Div, a, b)
  let exp a = Unary (Exp, a)
  let sqrt a = Unary (Sqrt, a)
  let select c a b = Select (c, a, b)
  let value_of_index i = Value_of_index i
  let load s c = Load (s, c)
  let round_f32 a = Round_f32 a
  let intrinsic i = Intrinsic i
  let reduce r = Reduce r

  let apply_binary = function
    | Add -> ( +. )
    | Sub -> ( -. )
    | Mul -> ( *. )
    | Div -> ( /. )

  let apply_unary = function Exp -> Stdlib.exp | Sqrt -> Stdlib.sqrt
  let binary_sym = function Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/"
  let unary_name = function Exp -> "exp" | Sqrt -> "sqrt"

  (* ---- structural identity, up to alpha-equivalence ----

     Reducer identities are opaque, so two expressions that differ only in which
     ordinals their supplies handed out denote the same thing and must compare
     and hash equally. Each side therefore carries a binder->LEVEL map built as
     the traversal descends, and reducers are compared by level rather than by
     identity. A free reducer keeps its identity, so it can still separate two
     otherwise-equal terms.

     Expressions that differ in nesting, bounds, kind, operand order or body do
     NOT compare equal: the reduction is an ordered fold, not a set.

     Constants compare by [Int64.bits_of_float], never by [( = )] or
     [Float.compare]. Both of those equate -0. with 0. and every NaN with every
     other, and this is a claim about structural identity -- the same rule
     [Ground_expr.compare] already states for the ground language. *)

  let index_tag : type r. r Index.t -> int = function
    | Index.Output _ -> 0
    | Index.Reduce _ -> 1
    | Index.Zero -> 2
    | Index.Const _ -> 3
    | Index.Of_position _ -> 4
    | Index.Add _ -> 5
    | Index.Scale _ -> 6
    | Index.Floor_div_pos _ -> 7
    | Index.Ceil_div_pos _ -> 8
    | Index.Min _ -> 9
    | Index.Max _ -> 10
    | Index.Clamp_low _ -> 11
    | Index.Assume_position _ -> 12

  let level env v =
    match Reduce_var.Map.find_opt v env with
    | Some l -> l
    (* Free: fall back to identity, offset so it can never alias a level. *)
    | None -> -1 - Reduce_var.hash v

  let ( <?> ) c f = if c <> 0 then c else f ()

  let rec cmp_index : type r s.
      int Reduce_var.Map.t ->
      int Reduce_var.Map.t ->
      r Index.t ->
      s Index.t ->
      int =
   fun ea eb a b ->
    Int.compare (index_tag a) (index_tag b) <?> fun () ->
    match (a, b) with
    | Index.Output x, Index.Output y -> Axis.compare x y
    | Index.Reduce x, Index.Reduce y -> Int.compare (level ea x) (level eb y)
    | Index.Zero, Index.Zero -> 0
    | Index.Const x, Index.Const y -> Int.compare x y
    | Index.Of_position x, Index.Of_position y -> cmp_index ea eb x y
    | Index.Clamp_low x, Index.Clamp_low y -> cmp_index ea eb x y
    | Index.Assume_position x, Index.Assume_position y -> cmp_index ea eb x y
    | Index.Scale (k, x), Index.Scale (l, y) ->
        Int.compare k l <?> fun () -> cmp_index ea eb x y
    | Index.Floor_div_pos (x, k), Index.Floor_div_pos (y, l)
    | Index.Ceil_div_pos (x, k), Index.Ceil_div_pos (y, l) ->
        Int.compare k l <?> fun () -> cmp_index ea eb x y
    | Index.Add (x1, x2), Index.Add (y1, y2)
    | Index.Min (x1, x2), Index.Min (y1, y2)
    | Index.Max (x1, x2), Index.Max (y1, y2) ->
        cmp_index ea eb x1 y1 <?> fun () -> cmp_index ea eb x2 y2
    (* Unreachable: the tags already agree. *)
    | _ -> 0

  let tag = function
    | Const _ -> 0
    | Binary _ -> 1
    | Unary _ -> 2
    | Select _ -> 3
    | Value_of_index _ -> 4
    | Load _ -> 5
    | Round_f32 _ -> 6
    | Reduce _ -> 7
    | Intrinsic _ -> 8

  let cmp_intrinsic ea eb (Intrinsic.Max_pool x) (Intrinsic.Max_pool y) =
    let open Intrinsic.Max_pool in
    let fld f = Int.compare (f x) (f y) in
    Source.compare x.source y.source <?> fun () ->
    fld (fun d -> d.in_h) <?> fun () ->
    fld (fun d -> d.in_w) <?> fun () ->
    fld (fun d -> d.kernel_h) <?> fun () ->
    fld (fun d -> d.kernel_w) <?> fun () ->
    fld (fun d -> d.stride_h) <?> fun () ->
    fld (fun d -> d.stride_w) <?> fun () ->
    fld (fun d -> d.pad_h) <?> fun () ->
    fld (fun d -> d.pad_w) <?> fun () ->
    Stdlib.compare x.result y.result <?> fun () ->
    List.fold_left2
      (fun acc a b -> acc <?> fun () -> cmp_index ea eb a b)
      0 (Coord.to_list x.out) (Coord.to_list y.out)

  let compare a b =
    let rec go ea eb n a b =
      Int.compare (tag a) (tag b) <?> fun () ->
      match (a, b) with
      | Const x, Const y ->
          Int64.compare (Int64.bits_of_float x) (Int64.bits_of_float y)
      | Binary (o, x1, x2), Binary (p, y1, y2) ->
          Stdlib.compare o p <?> fun () ->
          go ea eb n x1 y1 <?> fun () -> go ea eb n x2 y2
      | Unary (o, x), Unary (p, y) ->
          Stdlib.compare o p <?> fun () -> go ea eb n x y
      | Round_f32 x, Round_f32 y -> go ea eb n x y
      | Select (c, x1, x2), Select (d, y1, y2) ->
          (match (c, d) with
            | Bool.Value_lt (p, q), Bool.Value_lt (r, s) ->
                go ea eb n p r <?> fun () -> go ea eb n q s
            | Bool.Index_eq (p, q), Bool.Index_eq (r, s) ->
                cmp_index ea eb p r <?> fun () -> cmp_index ea eb q s
            | Bool.Value_lt _, Bool.Index_eq _ -> -1
            | Bool.Index_eq _, Bool.Value_lt _ -> 1)
          <?> fun () ->
          go ea eb n x1 y1 <?> fun () -> go ea eb n x2 y2
      | Value_of_index x, Value_of_index y -> cmp_index ea eb x y
      | Load (s, x), Load (t, y) ->
          Source.compare s t <?> fun () ->
          List.fold_left2
            (fun acc a b -> acc <?> fun () -> cmp_index ea eb a b)
            0 (Coord.to_list x) (Coord.to_list y)
      | Reduce r, Reduce s ->
          Stdlib.compare r.Reduction.kind s.Reduction.kind <?> fun () ->
          cmp_index ea eb r.Reduction.lo s.Reduction.lo <?> fun () ->
          cmp_index ea eb r.Reduction.hi s.Reduction.hi <?> fun () ->
          go
            (Reduce_var.Map.add r.Reduction.var n ea)
            (Reduce_var.Map.add s.Reduction.var n eb)
            (n + 1) r.Reduction.body s.Reduction.body
      | Intrinsic x, Intrinsic y -> cmp_intrinsic ea eb x y
      (* Unreachable: the tags already agree. *)
      | _ -> 0
    in
    go Reduce_var.Map.empty Reduce_var.Map.empty 0 a b

  let equal a b = compare a b = 0

  (* Agrees with [compare] by construction: it mixes the same information in the
     same order, and reducers by level rather than identity. An optimisation
     only -- structural equality remains the authority. *)
  let hash e =
    let mix h x = (h * 31) + x in
    let rec idx : type r. int Reduce_var.Map.t -> int -> r Index.t -> int =
     fun env h i ->
      let h = mix h (index_tag i) in
      match i with
      | Index.Output a -> mix h (Axis.to_int a)
      | Index.Reduce v -> mix h (level env v)
      | Index.Zero -> h
      | Index.Const n -> mix h n
      | Index.Of_position a -> idx env h a
      | Index.Clamp_low a | Index.Assume_position a -> idx env h a
      | Index.Scale (k, a) -> idx env (mix h k) a
      | Index.Floor_div_pos (a, d) | Index.Ceil_div_pos (a, d) ->
          idx env (mix h d) a
      | Index.Add (a, b) | Index.Min (a, b) | Index.Max (a, b) ->
          idx env (idx env h a) b
    in
    let rec go env n h (e : t) =
      let h = mix h (tag e) in
      match e with
      | Const x ->
          (* Both halves: [Int64.to_int] drops the top bit on a 63-bit int and
             the top 32 under js_of_ocaml, so hashing the raw conversion would
             collide -0. with 0. -- permitted by the contract, but avoidable,
             and that is exactly the pair this comparison exists to separate. *)
          let b = Int64.bits_of_float x in
          mix
            (mix h (Int64.to_int (Int64.logand b 0xFFFFFFFFL)))
            (Int64.to_int (Int64.shift_right_logical b 32))
      | Binary (o, a, b) -> go env n (go env n (mix h (Hashtbl.hash o)) a) b
      | Unary (o, a) -> go env n (mix h (Hashtbl.hash o)) a
      | Round_f32 a -> go env n h a
      | Select (c, a, b) ->
          let h =
            match c with
            | Bool.Value_lt (x, y) -> go env n (go env n h x) y
            | Bool.Index_eq (x, y) -> idx env (idx env h x) y
          in
          go env n (go env n h a) b
      | Value_of_index i -> idx env h i
      | Load (s, c) ->
          Coord.fold (fun h i -> idx env h i) (mix h (Source.hash s)) c
      | Reduce r ->
          let h = mix h (Hashtbl.hash r.Reduction.kind) in
          let h = idx env (idx env h r.Reduction.lo) r.Reduction.hi in
          go
            (Reduce_var.Map.add r.Reduction.var n env)
            (n + 1) h r.Reduction.body
      | Intrinsic (Intrinsic.Max_pool d) ->
          let open Intrinsic.Max_pool in
          let h = mix h (Source.hash d.source) in
          let h =
            List.fold_left mix h
              [
                d.in_h;
                d.in_w;
                d.kernel_h;
                d.kernel_w;
                d.stride_h;
                d.stride_w;
                d.pad_h;
                d.pad_w;
                Hashtbl.hash d.result;
              ]
          in
          Coord.fold (fun h i -> idx env h i) h d.out
    in
    go Reduce_var.Map.empty 0 17 e
end

(* ---- construction supply -------------------------------------------------- *)

module Builder = struct
  (* Reducer identities must be fresh during construction AND rewriting. The
     supply is threaded, not global: [run_from] is what lets an adapter that
     cannot itself mint a [Reduce_var.t] (because [SEMANTICS.sum] is a plain
     function, not a monadic one) still advance it. With only [run], every
     reduction would restart at ordinal 0 and nested reductions would capture. *)
  type state = int
  type 'a t = state -> 'a * state

  exception Supply_exhausted
  (** Raised by [fresh_reduce] when the ordinal would wrap. A wrapped ordinal
      would re-mint a live identity and break scope, alpha-equivalence,
      comparison and hashing at once, so this is checked rather than assumed --
      a trusted-capacity boundary, named so it is visible. *)

  let initial : state = 0
  let return x s = (x, s)

  let bind m f s =
    let x, s = m s in
    f x s

  let map f m s =
    let x, s = m s in
    (f x, s)

  let run_from s m = m s
  let run m = fst (m initial)

  let fresh_reduce s =
    if s = Stdlib.max_int then raise Supply_exhausted else (s, s + 1)

  module Syntax = struct
    let ( let* ) = bind
    let ( let+ ) m f = map f m
  end

  (* Allocates the variable, hands its index expression to the body, and threads
     the updated supply through. Going through here rather than building a
     [Reduction.t] by hand is what makes scope correct by construction: the
     callback cannot name a variable that is not this reduction's, and the
     variable cannot escape into a sibling. *)
  let reduction ~kind ~lo ~hi body s =
    let v, s = fresh_reduce s in
    let b, s = body (Index.reduce v) s in
    (Value.reduce { Reduction.kind; var = v; lo; hi; body = b }, s)

  (* Assembles a reduction around an ALREADY-MINTED variable. Prefer
     [reduction], which is correct by construction.

     This exists for an adapter conforming to a non-monadic interface --
     [SEMANTICS.sum] is a plain function -- where the supply is held in a ref.
     Such an adapter must mint the variable and PUBLISH the advanced supply
     before it evaluates the body, because the body may itself contain a nested
     reduction that reads the same ref; doing it the other way round hands both
     the same ordinal. Splitting minting from assembly is what makes that
     ordering explicit rather than accidental.

     The caller owes that [var] came from its own supply and is not already in
     scope. [Check.value] verifies it. *)
  let reduction_of ~kind ~var ~lo ~hi ~body =
    Value.reduce { Reduction.kind; var; lo; hi; body }
end

(* ---- traversals ----------------------------------------------------------- *)

(* ---- read-only traversals -------------------------------------------------- *)

module Fold = struct
  (* Concrete, reviewed queries rather than a generic visitor: binder behaviour
     stays visible in each signature, and a new constructor breaks the ones that
     must handle it instead of silently falling through a default. *)

  let rec index_reducers : type r. Reduce_var.Set.t -> r Index.t -> _ =
   fun acc -> function
    | Index.Output _ | Index.Zero | Index.Const _ -> acc
    | Index.Reduce v -> Reduce_var.Set.add v acc
    | Index.Of_position a -> index_reducers acc a
    | Index.Scale (_, a)
    | Index.Floor_div_pos (a, _)
    | Index.Ceil_div_pos (a, _)
    | Index.Clamp_low a
    | Index.Assume_position a ->
        index_reducers acc a
    | Index.Add (a, b) | Index.Min (a, b) | Index.Max (a, b) ->
        index_reducers (index_reducers acc a) b

  let rec index_axes : type r. Axis.t list -> r Index.t -> Axis.t list =
   fun acc -> function
    | Index.Output a -> if List.mem a acc then acc else a :: acc
    | Index.Reduce _ | Index.Zero | Index.Const _ -> acc
    | Index.Of_position a -> index_axes acc a
    | Index.Scale (_, a)
    | Index.Floor_div_pos (a, _)
    | Index.Ceil_div_pos (a, _)
    | Index.Clamp_low a
    | Index.Assume_position a ->
        index_axes acc a
    | Index.Add (a, b) | Index.Min (a, b) | Index.Max (a, b) ->
        index_axes (index_axes acc a) b

  let rec index_assume_sites : type r. int -> r Index.t -> int =
   fun acc -> function
    | Index.Output _ | Index.Reduce _ | Index.Zero | Index.Const _ -> acc
    | Index.Assume_position a -> index_assume_sites (acc + 1) a
    | Index.Of_position a -> index_assume_sites acc a
    | Index.Scale (_, a)
    | Index.Floor_div_pos (a, _)
    | Index.Ceil_div_pos (a, _)
    | Index.Clamp_low a ->
        index_assume_sites acc a
    | Index.Add (a, b) | Index.Min (a, b) | Index.Max (a, b) ->
        index_assume_sites (index_assume_sites acc a) b

  (* The index callback has to be RANK-2: a [Load]'s coordinate components are
     [Role.Position.t Index.t] while a reduction's upper bound is
     [Role.Delta.t Index.t], and an ordinary function argument would be fixed at
     whichever the inference engine saw first. Hence the record with an
     explicitly quantified field. *)
  type 'acc idx_fn = { idx : 'r. 'acc -> 'r Index.t -> 'acc }

  (* Generic bottom-up walk over a value, threading an accumulator. Every
     traversal below is written in terms of it, so a new [Value] constructor is
     handled in exactly one place. Reduction bounds are visited as indices and
     the body as a value; the BINDER is not interpreted here -- callers that care
     about scope (free variables, [Check]) handle it themselves. *)
  let rec walk ~value ~index ~intrinsic acc (e : Value.t) =
    let acc = value acc e in
    let recur = walk ~value ~index ~intrinsic in
    match e with
    | Value.Const _ -> acc
    | Value.Binary (_, a, b) -> recur (recur acc a) b
    | Value.Unary (_, a) | Value.Round_f32 a -> recur acc a
    | Value.Select (c, a, b) ->
        let acc =
          match c with
          | Bool.Value_lt (x, y) -> recur (recur acc x) y
          | Bool.Index_eq (x, y) -> index.idx (index.idx acc x) y
        in
        recur (recur acc a) b
    | Value.Value_of_index i -> index.idx acc i
    | Value.Load (_, c) -> Coord.fold (fun acc i -> index.idx acc i) acc c
    | Value.Reduce r ->
        let acc = index.idx (index.idx acc r.Reduction.lo) r.Reduction.hi in
        recur acc r.Reduction.body
    | Value.Intrinsic i ->
        let acc = intrinsic acc i in
        let (Intrinsic.Max_pool d) = i in
        Coord.fold (fun acc x -> index.idx acc x) acc d.Intrinsic.Max_pool.out

  let nothing acc _ = acc
  let no_index = { idx = (fun acc _ -> acc) }

  (* ONE metered traversal, computing both measures and carrying both budgets.
     [Check]'s limits exist to reject an oversized tree, so measuring first and
     comparing after would exhaust the stack on exactly the input the limit is
     there to refuse. But a walk per limit is not enough either: whichever runs
     first still descends the full input whenever its OWN bound is loose, so a
     loose size limit defeats a tight depth limit, and swapping the order
     defeats the dual case. Carrying both on one walk bounds the recursion by
     the TIGHTER of the two.

     Index trees are metered too. They are where a load's addressing lives, so a
     limit that treated them as leaves would bound nothing useful: a single
     [Value_of_index] can carry an arbitrarily deep affine expression.

     [size] and [depth] are this same walk with both budgets at [max_int], which
     never trip — so there is exactly one description of what counts as a node
     and what counts as a level. *)
  exception Over of [ `Size | `Depth ]

  let measure ~max_size ~max_depth e =
    let left = ref max_size in
    (* Charged once per node, before descending: that is what keeps the
       recursion inside the budget rather than merely reporting on it. *)
    let node budget =
      if budget <= 0 then raise_notrace (Over `Depth);
      if !left <= 0 then raise_notrace (Over `Size);
      decr left
    in
    let rec index : type r. int -> r Index.t -> int =
     fun budget i ->
      node budget;
      let sub = budget - 1 in
      match i with
      | Index.Output _ | Index.Reduce _ | Index.Zero | Index.Const _ -> 1
      (* Separate arm: [Of_position]'s operand is a position, the rest are
         deltas, and an or-pattern cannot bind [a] at both roles. *)
      | Index.Of_position a -> 1 + index sub a
      | Index.Scale (_, a)
      | Index.Floor_div_pos (a, _)
      | Index.Ceil_div_pos (a, _)
      | Index.Clamp_low a
      | Index.Assume_position a ->
          1 + index sub a
      | Index.Add (a, b) | Index.Min (a, b) | Index.Max (a, b) ->
          1 + Stdlib.max (index sub a) (index sub b)
    in
    let coord budget c =
      Coord.fold (fun m i -> Stdlib.max m (index budget i)) 0 c
    in
    let rec value budget (e : Value.t) =
      node budget;
      let sub = budget - 1 in
      match e with
      | Value.Const _ -> 1
      | Value.Value_of_index i -> 1 + index sub i
      | Value.Load (_, c) -> 1 + coord sub c
      | Value.Unary (_, a) | Value.Round_f32 a -> 1 + value sub a
      | Value.Binary (_, a, b) -> 1 + Stdlib.max (value sub a) (value sub b)
      | Value.Select (c, a, b) ->
          let g =
            match c with
            | Bool.Value_lt (x, y) -> Stdlib.max (value sub x) (value sub y)
            | Bool.Index_eq (x, y) -> Stdlib.max (index sub x) (index sub y)
          in
          1 + Stdlib.max g (Stdlib.max (value sub a) (value sub b))
      | Value.Reduce r ->
          1
          + Stdlib.max
              (Stdlib.max (index sub r.Reduction.lo) (index sub r.Reduction.hi))
              (value sub r.Reduction.body)
      | Value.Intrinsic (Intrinsic.Max_pool d) ->
          1 + coord sub d.Intrinsic.Max_pool.out
    in
    let d = value max_depth e in
    (max_size - !left, d)

  let unmetered e = measure ~max_size:Stdlib.max_int ~max_depth:Stdlib.max_int e
  let size e = fst (unmetered e)
  let depth e = snd (unmetered e)

  (* Which limit was passed, without measuring the rest. Both are enforced
     together for the reason above, so an absent limit is [max_int] rather than
     a skipped budget. *)
  let exceeds ~max_size ~max_depth e =
    match measure ~max_size ~max_depth e with
    | _ -> None
    | exception Over w -> Some w

  let sources e =
    walk
      ~value:(fun acc -> function
        | Value.Load (s, _) -> Source.Set.add s acc
        | Value.Intrinsic (Intrinsic.Max_pool d) ->
            Source.Set.add d.Intrinsic.Max_pool.source acc
        | _ -> acc)
      ~index:no_index ~intrinsic:nothing Source.Set.empty e

  let output_axes e =
    walk ~value:nothing ~index:{ idx = index_axes } ~intrinsic:nothing [] e
    |> List.sort Axis.compare

  let assume_sites e =
    walk ~value:nothing
      ~index:{ idx = index_assume_sites }
      ~intrinsic:nothing 0 e

  let intrinsics e =
    walk ~value:nothing ~index:no_index ~intrinsic:(fun n _ -> n + 1) 0 e

  (* Scope-aware, unlike the queries above: a reducer mentioned under its own
     binder is bound, not free. A well-formed top-level expression has none. *)
  let free_reducers e =
    let rec go bound acc (e : Value.t) =
      let idx acc i =
        Reduce_var.Set.diff (index_reducers Reduce_var.Set.empty i) bound
        |> Reduce_var.Set.union acc
      in
      match e with
      | Value.Const _ -> acc
      | Value.Binary (_, a, b) -> go bound (go bound acc a) b
      | Value.Unary (_, a) | Value.Round_f32 a -> go bound acc a
      | Value.Select (c, a, b) ->
          let acc =
            match c with
            | Bool.Value_lt (x, y) -> go bound (go bound acc x) y
            | Bool.Index_eq (x, y) -> idx (idx acc x) y
          in
          go bound (go bound acc a) b
      | Value.Value_of_index i -> idx acc i
      | Value.Load (_, c) -> Coord.fold idx acc c
      | Value.Reduce r ->
          (* The bounds are OUTSIDE the binder: they may mention enclosing
             reducers but not this one. *)
          let acc = idx (idx acc r.Reduction.lo) r.Reduction.hi in
          go (Reduce_var.Set.add r.Reduction.var bound) acc r.Reduction.body
      | Value.Intrinsic (Intrinsic.Max_pool d) ->
          Coord.fold idx acc d.Intrinsic.Max_pool.out
    in
    go Reduce_var.Set.empty Reduce_var.Set.empty e

  (* Binders in lexical (pre-)order, with repeats: an identity bound in two
     sibling scopes appears twice, which is what makes this usable for counting
     binders as distinct from counting identities. Inspection only -- [Pp] and
     the structural comparison each carry their own SCOPED environment, because
     a list keyed by identity cannot distinguish those siblings. *)
  let binders e =
    let rec go acc (e : Value.t) =
      match e with
      | Value.Const _ | Value.Value_of_index _ | Value.Load _
      | Value.Intrinsic _ ->
          acc
      | Value.Binary (_, a, b) -> go (go acc a) b
      | Value.Unary (_, a) | Value.Round_f32 a -> go acc a
      | Value.Select (c, a, b) ->
          let acc =
            match c with Bool.Value_lt (x, y) -> go (go acc x) y | _ -> acc
          in
          go (go acc a) b
      | Value.Reduce r -> go (r.Reduction.var :: acc) r.Reduction.body
    in
    List.rev (go [] e)
end

(* ---- scope-aware rewriting ------------------------------------------------ *)

module Rewrite = struct
  (* Everything here rebuilds with the RAW constructors, never the smart ones.
     Smart constructors fold -- [floor_div_pos] by 1 collapses today, and stage 8
     adds more -- and a structure-preserving rewrite that silently changed syntax
     would make the migration's golden diff unattributable. The raw constructors
     are reachable because these are sections of [Expr] rather than separate
     units. *)

  let rec map_index_reducers : type r.
      (Reduce_var.t -> Reduce_var.t) -> r Index.t -> r Index.t =
   fun f i ->
    match i with
    | Index.Output _ | Index.Zero | Index.Const _ -> i
    | Index.Reduce v -> Index.Reduce (f v)
    | Index.Of_position a -> Index.Of_position (map_index_reducers f a)
    | Index.Clamp_low a -> Index.Clamp_low (map_index_reducers f a)
    | Index.Assume_position a -> Index.Assume_position (map_index_reducers f a)
    | Index.Scale (k, a) -> Index.Scale (k, map_index_reducers f a)
    | Index.Floor_div_pos (a, d) ->
        Index.Floor_div_pos (map_index_reducers f a, d)
    | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (map_index_reducers f a, d)
    | Index.Add (a, b) ->
        Index.Add (map_index_reducers f a, map_index_reducers f b)
    | Index.Min (a, b) ->
        Index.Min (map_index_reducers f a, map_index_reducers f b)
    | Index.Max (a, b) ->
        Index.Max (map_index_reducers f a, map_index_reducers f b)

  let rec subst_index : type r.
      Role.Position.t Index.t Coord.t -> r Index.t -> r Index.t =
   fun c i ->
    match i with
    (* The one substituted form. Role-preserving: an output variable is a
       position and so is its replacement. *)
    | Index.Output a -> Coord.get c a
    (* Deliberately NOT substituted -- a reducer is bound by its reduction, and
       replacing one here is precisely the capture this module prevents. *)
    | Index.Reduce _ | Index.Zero | Index.Const _ -> i
    | Index.Of_position a -> Index.Of_position (subst_index c a)
    | Index.Clamp_low a -> Index.Clamp_low (subst_index c a)
    | Index.Assume_position a -> Index.Assume_position (subst_index c a)
    | Index.Scale (k, a) -> Index.Scale (k, subst_index c a)
    | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (subst_index c a, d)
    | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (subst_index c a, d)
    | Index.Add (a, b) -> Index.Add (subst_index c a, subst_index c b)
    | Index.Min (a, b) -> Index.Min (subst_index c a, subst_index c b)
    | Index.Max (a, b) -> Index.Max (subst_index c a, subst_index c b)

  (* Rank-2, for the same reason as [Fold.walk]: a [Load]'s components are
     [Role.Position.t Index.t] while a reduction's upper bound is
     [Role.Delta.t Index.t], and a plain function argument would be fixed at
     whichever inference saw first. *)
  type 'env idx_fn = { on_index : 'r. 'env -> 'r Index.t -> 'r Index.t }

  let keep_indices = { on_index = (fun _ i -> i) }

  (* A structural map that rebuilds every node, parameterised by what to do at
     the leaves an individual rewrite cares about. [on_reduce] sees each binder
     and returns its replacement plus the environment its body is rewritten
     under, which is what keeps scope handling in ONE place instead of repeated
     per rewrite. *)
  let rec rebuild ~idx ~src ~on_reduce env (e : Value.t) : Value.t =
    let go = rebuild ~idx ~src ~on_reduce env in
    let idxe i = idx.on_index env i in
    match e with
    | Value.Const _ -> e
    | Value.Binary (op, a, b) -> Value.Binary (op, go a, go b)
    | Value.Unary (op, a) -> Value.Unary (op, go a)
    | Value.Round_f32 a -> Value.Round_f32 (go a)
    | Value.Select (c, a, b) ->
        let c =
          match c with
          | Bool.Value_lt (x, y) -> Bool.Value_lt (go x, go y)
          | Bool.Index_eq (x, y) -> Bool.Index_eq (idxe x, idxe y)
        in
        Value.Select (c, go a, go b)
    | Value.Value_of_index i -> Value.Value_of_index (idxe i)
    | Value.Load (s, c) -> Value.Load (src s, Coord.map idxe c)
    | Value.Reduce r ->
        let var, env' = on_reduce env r.Reduction.var in
        Value.Reduce
          {
            Reduction.kind = r.Reduction.kind;
            var;
            (* Bounds sit OUTSIDE the binder: rewritten under the enclosing
               environment, not the body's. *)
            lo = idxe r.Reduction.lo;
            hi = idxe r.Reduction.hi;
            body = rebuild ~idx ~src ~on_reduce env' r.Reduction.body;
          }
    | Value.Intrinsic (Intrinsic.Max_pool d) ->
        Value.Intrinsic
          (Intrinsic.Max_pool
             {
               d with
               Intrinsic.Max_pool.source = src d.Intrinsic.Max_pool.source;
               out = Coord.map idxe d.Intrinsic.Max_pool.out;
             })

  let subst_env env v =
    match Reduce_var.Map.find_opt v env with Some w -> w | None -> v

  (* Replaces every BOUND reducer identity consistently and leaves free ones
     alone. Composing two independently built fragments is exactly when this is
     needed: both supplies start at [initial], so both mint ordinal 0, and the
     inner binder would otherwise capture references meant for the outer one.

     Freshen the fragment being INSERTED, before composing it. Freshening the
     combined tree afterwards cannot repair anything -- once a nominal collision
     has captured a reference there is no record of which binder it meant. *)
  let freshen e s =
    (* Replacements must SKIP identities that occur free in [e]. A free reducer
       is a reference to a binder outside this expression, and minting one of
       them here binds it -- silently turning an ill-scoped term into a
       well-scoped-looking one with a different denotation, which [Check] would
       then report as [Ok].

       [alpha_normalize] makes that maximally likely, since it always starts
       from [initial]: any free ordinal near zero is directly in the way. *)
    let free = Fold.free_reducers e in
    let supply = ref s in
    let rec mint () =
      let w, s' = Builder.run_from !supply Builder.fresh_reduce in
      supply := s';
      if Reduce_var.Set.mem w free then mint () else w
    in
    let on_reduce env v =
      let w = mint () in
      (w, Reduce_var.Map.add v w env)
    in
    let out =
      rebuild
        ~idx:{ on_index = (fun env i -> map_index_reducers (subst_env env) i) }
        ~src:Fun.id ~on_reduce Reduce_var.Map.empty e
    in
    (out, !supply)

  (* Deterministic renaming by lexical traversal: running [freshen] from a fixed
     initial supply gives binders the LOWEST ORDINALS NOT FREE in the
     expression, in the order they are met -- 0, 1, ... for a closed expression,
     and skipping the free ones otherwise, because [freshen] must not bind them.
     Canonical either way: alpha-equivalent expressions share a free set, so
     they skip the same ordinals. Does not reorder operations or reductions. *)
  let alpha_normalize e = fst (freshen e Builder.initial)

  (* Replaces only output-axis variables, never reducers. If the result is
     placed beneath another reduction, the CALLER freshens it first -- this
     function cannot know that context. *)
  let substitute_output c e =
    rebuild
      ~idx:{ on_index = (fun _ i -> subst_index c i) }
      ~src:Fun.id
      ~on_reduce:(fun () v -> (v, ()))
      () e

  let map_sources f e =
    rebuild ~idx:keep_indices ~src:f ~on_reduce:(fun () v -> (v, ())) () e
end

(* ---- structural validation ------------------------------------------------ *)

module Check = struct
  (* Deliberately narrow. Division parameters, load roles and intrinsic
     dimensions are NOT rechecked here: the smart constructors and the [Index]
     GADT make each of those unconstructable through the public API, so a rule
     for them could never be turned red, and CLAUDE.md is explicit that a check
     which has never failed is not evidence. What remains is exactly what
     COMPOSITION can still violate. *)
  type error =
    [ `Free_reducer of Reduce_var.t
    | `Duplicate_binder of Reduce_var.t
    | `Too_large of int
    | `Too_deep of int ]

  (* The payload is the LIMIT, not the measure. Reporting the actual size would
     mean measuring the whole tree, which is the thing the limit exists to
     avoid. *)
  let pp_error fmt : [< error ] -> unit = function
    | `Free_reducer v -> Fmt.pf fmt "free reducer %a" Reduce_var.pp v
    | `Duplicate_binder v ->
        Fmt.pf fmt "reducer %a is bound twice on one path" Reduce_var.pp v
    | `Too_large limit -> Fmt.pf fmt "size exceeds limit %d" limit
    | `Too_deep limit -> Fmt.pf fmt "depth exceeds limit %d" limit

  (* A variable bound again inside its own scope. Two independently built
     fragments composed without freshening is how this arises in practice, and
     it is a real defect rather than shadowing: the inner binder captures
     references meant for the outer one, so evaluation silently changes. *)
  let duplicate_binder e =
    let rec go bound (e : Value.t) =
      match e with
      | Value.Const _ | Value.Value_of_index _ | Value.Load _
      | Value.Intrinsic _ ->
          None
      | Value.Binary (_, a, b) -> (
          match go bound a with None -> go bound b | some -> some)
      | Value.Unary (_, a) | Value.Round_f32 a -> go bound a
      | Value.Select (c, a, b) -> (
          let guard =
            match c with
            | Bool.Value_lt (x, y) -> (
                match go bound x with None -> go bound y | some -> some)
            | Bool.Index_eq _ -> None
          in
          match guard with
          | Some _ -> guard
          | None -> ( match go bound a with None -> go bound b | some -> some))
      | Value.Reduce r ->
          if Reduce_var.Set.mem r.Reduction.var bound then Some r.Reduction.var
          else go (Reduce_var.Set.add r.Reduction.var bound) r.Reduction.body
    in
    go Reduce_var.Set.empty e

  (* The limits come FIRST, and are metered by one traversal carrying both. Both
     scope traversals recurse over the whole tree, so running them ahead of the
     limits would exhaust the stack on precisely the oversized input the limits
     exist to reject -- a resource guard that only works on trees that did not
     need one. So an expression past a configured limit is reported as
     [`Too_large]/[`Too_deep] even when it is also ill-scoped.

     An absent limit becomes [max_int], which cannot trip: the two budgets have
     to ride the same walk (see [Fold.measure]), so leaving one out must mean an
     unreachable bound rather than a separate pass. With both absent there is
     nothing to bound, and the walk is skipped. *)
  let value ?max_size ?max_depth e =
    let open Core.Syntax in
    let or_unbounded = function Some l -> l | None -> Stdlib.max_int in
    let* () =
      match (max_size, max_depth) with
      | None, None -> Core.return ()
      | _ -> (
          let max_size = or_unbounded max_size
          and max_depth = or_unbounded max_depth in
          match Fold.exceeds ~max_size ~max_depth e with
          | Some `Size -> Core.fail (`Too_large max_size)
          | Some `Depth -> Core.fail (`Too_deep max_depth)
          | None -> Core.return ())
    in
    let* () =
      match Reduce_var.Set.min_elt_opt (Fold.free_reducers e) with
      | Some v -> Core.fail (`Free_reducer v)
      | None -> Core.return ()
    in
    match duplicate_binder e with
    | Some v -> Core.fail (`Duplicate_binder v)
    | None -> Core.return ()
end

(* ---- interpretation ------------------------------------------------------- *)

module Eval = struct
  (* Only what index evaluation can actually raise. [Eval.error] widens this in
     stage 4, once intrinsics and loads exist -- a stage cannot publish a type
     equation naming a module that arrives in the next one. *)
  type index_error =
    [ Checked.error
    | `Unbound_reducer of Reduce_var.t
    | `Index_not_exact_in_float of int ]

  let pp_index_error fmt : [< index_error ] -> unit = function
    | #Checked.error as e -> Checked.pp_error fmt e
    | `Unbound_reducer v -> Fmt.pf fmt "unbound reducer %a" Reduce_var.pp v
    | `Index_not_exact_in_float n ->
        Fmt.pf fmt "index %d is not exactly representable as a float" n

  (* The recursion raises and the public entry point converts once, rather than
     allocating an [Ok] per AST node per output pixel. The design permits this
     explicitly -- no state is exposed and the result is the defined value -- and
     it matters: this runs inside the grounding loop, which the transform
     verifier drives for every random-walk config.

     The exception carries an already-built [Core.Error.t], so the backtrace is
     the one captured where the failure was DETECTED, not where it was caught. *)
  exception Fail of index_error Core.Error.t

  let fail_with (k : index_error) = raise (Fail (Core.Error.make k))

  let chk : ('a, [< index_error ]) Core.result -> 'a = function
    | Ok v -> v
    | Error e -> raise (Fail (e :> index_error Core.Error.t))

  let eval_index (type r) ~(output : int Coord.t)
      ~(reducers : Reduce_var.t -> int option) (e : r Index.t) : int =
    let rec go : type r. r Index.t -> int = function
      | Index.Output a -> Coord.get output a
      | Index.Reduce v -> (
          match reducers v with
          | Some i -> i
          | None -> fail_with (`Unbound_reducer v))
      | Index.Zero -> 0
      | Index.Const n -> n
      | Index.Of_position i -> go i
      | Index.Add (a, b) -> chk (Checked.add (go a) (go b))
      | Index.Scale (k, a) -> chk (Checked.mul k (go a))
      | Index.Floor_div_pos (a, d) -> chk (Checked.floor_div_pos (go a) d)
      | Index.Ceil_div_pos (a, d) -> chk (Checked.ceil_div_pos (go a) d)
      | Index.Min (a, b) -> Stdlib.min (go a) (go b)
      | Index.Max (a, b) -> Stdlib.max (go a) (go b)
      | Index.Clamp_low a -> Stdlib.max 0 (go a)
      | Index.Assume_position a -> go a
    in
    go e

  let index ~output ~reducers e =
    try Ok (eval_index ~output ~reducers e) with Fail e -> Error e

  (* Exactness is a ROUND TRIP, not a magnitude threshold. binary64 stops
     representing CONSECUTIVE integers above 2^53, but plenty of larger ones are
     exact -- 2^53+2, 2^54 and [min_int] all are, while 2^53+1, 2^54+2 and
     [max_int] are not. A "reject above 2^53" rule would be wrong in both
     directions.

     The [Sys.int_size] guard is not a micro-optimisation: every representable
     js_of_ocaml [int] is already exact in binary64, so without it this would put
     two emulated [Int64] conversions per [Value_of_index] back on the JS
     per-pixel path -- the exact allocation cost the [int] index domain was
     chosen to avoid -- in exchange for a check that can never fire there. *)
  let float_of_index i =
    let f = Stdlib.float_of_int i in
    if Sys.int_size <= 53 then Core.return f
    else if Int64.equal (Int64.of_float f) (Int64.of_int i) then Core.return f
    else Core.fail (`Index_not_exact_in_float i)

  (* Everything a value can fail on. [`Unknown_source] and [`Coord_out_of_range]
     are raised by the host's [load], not here -- the language knows nothing
     about what a source is. *)
  type error =
    [ index_error
    | Intrinsic.error
    | `Unknown_source of Source.t
    | `Coord_out_of_range of Source.t * Axis.t * int * int Coord.t ]

  let pp_error fmt : [< error ] -> unit = function
    | #index_error as e -> pp_index_error fmt e
    | #Intrinsic.error as e -> Intrinsic.pp_error fmt e
    | `Unknown_source s -> Fmt.pf fmt "unknown source %a" Source.pp s
    | `Coord_out_of_range (s, a, v, c) ->
        Fmt.pf fmt "%a[%a] out of range on axis %a: %d" Source.pp s
          (Coord.pp Fmt.int) c Axis.pp a v

  module Env = struct
    (* The whole boundary between the language and its host. [Expr] supplies
       evaluated coordinates and consumes a working float; everything about
       storage format, quantization and tensor ownership lives on the other
       side. This is what keeps the library independent of [native]. *)
    type t = { load : Source.t -> int Coord.t -> (float, error) Core.result }
  end

  exception Fail_value of error Core.Error.t

  (* Every value-level failure arrives as a [Core.result] from somewhere else --
     the host's [load], the intrinsic geometry, the index evaluator -- so there
     is no direct-raise helper here to go with [vchk]. *)
  let vchk : ('a, [< error ]) Core.result -> 'a = function
    | Ok v -> v
    | Error e -> raise (Fail_value (e :> error Core.Error.t))

  let value (env : Env.t) ~output e =
    let idx reducers i =
      try eval_index ~output ~reducers i
      with Fail e -> raise (Fail_value (e :> error Core.Error.t))
    in
    let rec go reducers (e : Value.t) : float =
      match e with
      | Value.Const x -> x
      | Value.Binary (op, a, b) ->
          Value.apply_binary op (go reducers a) (go reducers b)
      | Value.Unary (op, a) -> Value.apply_unary op (go reducers a)
      (* Only the SELECTED branch is evaluated -- the other may divide by zero
         or read out of bounds, and guarding is what the caller built it for. *)
      | Value.Select (c, a, b) ->
          if guard reducers c then go reducers a else go reducers b
      | Value.Value_of_index i -> vchk (float_of_index (idx reducers i))
      | Value.Load (s, c) -> vchk (env.Env.load s (Coord.map (idx reducers) c))
      | Value.Round_f32 a ->
          (* Convert to binary32 and widen back. The one value expression that
             changes a value without being arithmetic. *)
          Int32.float_of_bits (Int32.bits_of_float (go reducers a))
      | Value.Reduce r ->
          let lo = idx reducers r.Reduction.lo
          and hi = idx reducers r.Reduction.hi in
          let combine, init =
            match r.Reduction.kind with
            | Reduction.Sum -> (( +. ), 0.)
            | Reduction.Max ->
                (Max_op.apply Max_op.Float_max, Float.neg_infinity)
          in
          (* The ordered half-open left fold the denotation specifies. Same seed
             and same association as the engine's own reduction -- a rewrite that
             reassociated this would change the answer, not just its shape. *)
          let rec fold i acc =
            if i >= hi then acc
            else
              let bound v =
                if Reduce_var.equal v r.Reduction.var then Some i
                else reducers v
              in
              fold (i + 1) (combine acc (go bound r.Reduction.body))
          in
          fold lo init
      | Value.Intrinsic i -> intrinsic reducers i
    and guard reducers = function
      | Bool.Value_lt (a, b) -> go reducers a < go reducers b
      | Bool.Index_eq (a, b) -> Int.equal (idx reducers a) (idx reducers b)
    and intrinsic reducers (Intrinsic.Max_pool d as i) =
      let open Intrinsic.Max_pool in
      let at a = idx reducers (Coord.get d.out a) in
      let w = vchk (Intrinsic.window i ~out_h:(at Axis.H) ~out_w:(at Axis.W)) in
      let read ih iw =
        vchk
          (env.Env.load d.source
             (Coord.of_fn (fun a ->
                  if a = Axis.H then ih else if a = Axis.W then iw else at a)))
      in
      (* Value and index advance TOGETHER under one predicate. Updating them
         separately is how they fell out of step originally, which is why
         [Max_op.pool_better] is shared rather than open-coded. An ordinary tie
         keeps the incumbent; a NaN re-triggers, so the LAST NaN wins. *)
      let rec rows ih best best_ix =
        if ih >= w.Intrinsic.Window.hhi then (best, best_ix)
        else cols ih w.Intrinsic.Window.wlo best best_ix
      and cols ih iw best best_ix =
        if iw >= w.Intrinsic.Window.whi then rows (ih + 1) best best_ix
        else
          let v = read ih iw in
          let best, best_ix =
            if Max_op.pool_better ~best ~value:v then
              (v, vchk (Intrinsic.flat_index i ~ih ~iw))
            else (best, best_ix)
          in
          cols ih (iw + 1) best best_ix
      in
      let best, best_ix = rows w.Intrinsic.Window.hlo Float.neg_infinity 0 in
      match d.result with
      | Value -> best
      | Index -> vchk (float_of_index best_ix)
    in
    try Ok (go (fun _ -> None) e) with Fail_value e -> Error e
end

(* ---- printing ------------------------------------------------------------- *)

module Pp = struct
  (* Reducer display names are supplied by the caller rather than derived from
     the opaque identity, so output never depends on which ordinals a supply
     handed out. [Pp.value] threads the lexical r0, r1, ... assignment in stage
     4; here the naming function is explicit, with no default to fall back on.

     Every form below renders exactly as the representation it replaces, which
     is what keeps the migration's golden diff attributable to a single cause:
     [Zero] as [0] (it was [Index_const 0]), [Of_position] and [Assume_position]
     transparently (both were the identity), and [Clamp_low] as [max(0,_)] (it
     was [Index_max (Index_const 0, _)]). *)
  let index ~names fmt e =
    let rec go : type r. Format.formatter -> r Index.t -> unit =
     fun fmt -> function
       | Index.Output a -> Axis.pp fmt a
       | Index.Reduce v -> Fmt.string fmt (names v)
       | Index.Zero -> Fmt.int fmt 0
       | Index.Const n -> Fmt.int fmt n
       | Index.Of_position i -> go fmt i
       | Index.Add (a, b) -> Fmt.pf fmt "%a+%a" go a go b
       | Index.Scale (k, a) -> Fmt.pf fmt "%d*%a" k go a
       | Index.Floor_div_pos (a, d) -> Fmt.pf fmt "floor_div(%a,%d)" go a d
       | Index.Ceil_div_pos (a, d) -> Fmt.pf fmt "ceil_div(%a,%d)" go a d
       | Index.Min (a, b) -> Fmt.pf fmt "min(%a,%a)" go a go b
       | Index.Max (a, b) -> Fmt.pf fmt "max(%a,%a)" go a go b
       | Index.Clamp_low a -> Fmt.pf fmt "max(0,%a)" go a
       | Index.Assume_position a -> go fmt a
    in
    go fmt e

  (* Display names are assigned r1, r2, ... in LEXICAL order, so output is
     stable across expressions that differ only in which ordinals their supply
     handed out -- two structurally identical formulas built by independent
     builders print identically. *)
  (* Numbered from 1, not 0. The representation being replaced numbered reducers
     from an allocation counter that incremented BEFORE building the body, so its
     order was already lexical and only the base differed. Matching it means the
     migration moves no goldens at all, which turns "did the cutover change
     behaviour?" into a question the diff answers by itself. Renumbering from 0
     is a separate, labelled change if it is ever wanted. *)

  (* The naming environment is SCOPED, not a global identity-to-name map. Two
     independently built fragments can bind the same ordinal in sibling scopes
     -- well-scoped, and [Check] accepts it, since neither shadows the other --
     and one map keyed by identity would then give both binders whichever name
     came last, and would render a free reference in one sibling as bound
     because its twin binds that identity. So the environment descends with the
     tree and the counter runs across it, which is what makes the names lexical
     rather than merely distinct. *)
  let names_in env v =
    match Reduce_var.Map.find_opt v env with
    | Some n -> n
    | None -> Fmt.str "?%a" Reduce_var.pp v

  (* Renders exactly as the representation being replaced, so the migration's
     golden diff stays attributable to a single cause. [Round_f32] is the one
     genuinely new form and has no old spelling to preserve. *)
  let value fmt e =
    let next = ref 1 in
    let idx env fmt i = index ~names:(names_in env) fmt i in
    let rec at env fmt (e : Value.t) =
      let go fmt e = at env fmt e in
      let idx fmt i = idx env fmt i in
      let guard fmt c = guard_at env fmt c in
      match e with
      | Value.Const x -> Fmt.float fmt x
      | Value.Binary (op, a, b) ->
          Fmt.pf fmt "(%a %s %a)" go a (Value.binary_sym op) go b
      | Value.Unary (op, a) -> Fmt.pf fmt "%s(%a)" (Value.unary_name op) go a
      | Value.Select (c, a, b) ->
          Fmt.pf fmt "select(%a, %a, %a)" guard c go a go b
      | Value.Value_of_index i -> Fmt.pf fmt "value_of_index(%a)" idx i
      | Value.Load (s, c) -> Fmt.pf fmt "%a[%a]" Source.pp s (Coord.pp idx) c
      | Value.Round_f32 a -> Fmt.pf fmt "f32(%a)" go a
      | Value.Reduce r ->
          (* Named before descending, so the counter follows lexical order --
             and the bounds print under the OUTER environment, since they are
             evaluated outside the binder and cannot mention it. *)
          let name = Fmt.str "r%d" !next in
          incr next;
          let inner = Reduce_var.Map.add r.Reduction.var name env in
          Fmt.pf fmt "%s(%s=%a..%a: %a)"
            (Reduction.kind_name r.Reduction.kind)
            name idx r.Reduction.lo idx r.Reduction.hi (at inner)
            r.Reduction.body
      | Value.Intrinsic (Intrinsic.Max_pool d) ->
          Fmt.pf fmt "max_pool2d_%s(%a; k=%dx%d s=%dx%d p=%dx%d; out=[%a])"
            (Intrinsic.Max_pool.result_name d.Intrinsic.Max_pool.result)
            Source.pp d.Intrinsic.Max_pool.source d.Intrinsic.Max_pool.kernel_h
            d.Intrinsic.Max_pool.kernel_w d.Intrinsic.Max_pool.stride_h
            d.Intrinsic.Max_pool.stride_w d.Intrinsic.Max_pool.pad_h
            d.Intrinsic.Max_pool.pad_w (Coord.pp idx) d.Intrinsic.Max_pool.out
    and guard_at env fmt = function
      | Bool.Value_lt (a, b) -> Fmt.pf fmt "(%a < %a)" (at env) a (at env) b
      | Bool.Index_eq (a, b) -> Fmt.pf fmt "(%a = %a)" (idx env) a (idx env) b
    in
    at Reduce_var.Map.empty fmt e
end
