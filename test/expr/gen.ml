(* A generator of well-formed expressions, for the property tests.

   The Builder supply is threaded monadically through the ENTIRE tree -- a
   single [Builder.run] per sample -- which is what actually makes the output
   well-scoped. Using only the public API does not buy that: two [Builder.run]
   calls both start at [initial], so their results compose through public
   constructors with duplicate reducer identities. That collision is precisely
   why freshening and [Check] exist, so a generator producing it by accident
   would report Expr defects that were really malformed inputs. [Prop] asserts
   [Check.value] on every sample rather than trusting this.

   The PCG lives in a ref for the duration of one sample. It is seeded and
   deterministic, and it is an implementation convenience inside a generator --
   the SUPPLY, which is what scope depends on, stays threaded. Deliberately
   overlapping supplies belong in [colliding_pair], which is the capture case
   and is kept separate on purpose. *)

module Pcg = Walk_core.Pcg
open Expr.Builder.Syntax

type t = { mutable rng : Pcg.t }

let create ~seed = { rng = Pcg.seed ~seed ~seq:1L }

(* [Pcg.next] yields an int64 in [0, 2^32); every span here is tiny, so the
   narrowing to [int] is bounded by construction and safe at 32 bits. *)
let int_in st ~lo ~hi =
  let n, rng = Pcg.next st.rng in
  st.rng <- rng;
  lo + Int64.to_int (Int64.rem n (Int64.of_int (hi - lo + 1)))

let pick st xs = List.nth xs (int_in st ~lo:0 ~hi:(List.length xs - 1))
let sources = [ Expr.Source.create 0; Expr.Source.create 1 ]

(* Indices reference only reducers in [in_scope], so a generated term never has
   a free one. The property suite checks that rather than assuming it. *)
let rec index st ~in_scope ~depth =
  if depth <= 0 then leaf st ~in_scope
  else
    match int_in st ~lo:0 ~hi:5 with
    | 0 | 1 -> leaf st ~in_scope
    | 2 ->
        let a = index st ~in_scope ~depth:(depth - 1) in
        Expr.Index.add a (index st ~in_scope ~depth:(depth - 1))
    | 3 ->
        let k = int_in st ~lo:(-3) ~hi:3 in
        Expr.Index.scale k (index st ~in_scope ~depth:(depth - 1))
    | 4 ->
        let a = index st ~in_scope ~depth:(depth - 1) in
        Expr.Index.min a (index st ~in_scope ~depth:(depth - 1))
    | _ -> (
        let a = index st ~in_scope ~depth:(depth - 1) in
        let d = int_in st ~lo:1 ~hi:4 in
        match Expr.Index.floor_div_pos a d with Ok i -> i | Error _ -> a)

and leaf st ~in_scope =
  match int_in st ~lo:0 ~hi:2 with
  | 0 -> Expr.Index.const (int_in st ~lo:(-4) ~hi:4)
  | 1 -> Expr.Index.of_position (Expr.Index.output (pick st Expr.Axis.all))
  | _ -> (
      match in_scope with
      | [] -> Expr.Index.const (int_in st ~lo:0 ~hi:4)
      | _ -> Expr.Index.of_position (pick st in_scope))

(* Back to a position soundly, so [load] accepts it. *)
let position st ~in_scope ~depth =
  Expr.Index.clamp_low (index st ~in_scope ~depth)

let coord st ~in_scope ~depth =
  (* [Coord.of_fn] applies its argument once per axis in a fixed order, so the
     draws stay reproducible from the seed. *)
  Expr.Coord.of_fn (fun _ -> position st ~in_scope ~depth)

let rec value st ~in_scope ~depth : Expr.Value.t Expr.Builder.t =
  let const () =
    Expr.Builder.return
      (Expr.Value.const (float_of_int (int_in st ~lo:(-3) ~hi:3)))
  in
  if depth <= 0 then const ()
  else
    let sub () = value st ~in_scope ~depth:(depth - 1) in
    match int_in st ~lo:0 ~hi:7 with
    | 0 -> const ()
    | 1 ->
        let* a = sub () in
        let+ b = sub () in
        Expr.Value.add a b
    | 2 ->
        let* a = sub () in
        let+ b = sub () in
        Expr.Value.mul a b
    | 3 ->
        let s = pick st sources in
        Expr.Builder.return (Expr.Value.load s (coord st ~in_scope ~depth:1))
    | 4 ->
        let* a = sub () in
        let* b = sub () in
        let* x = sub () in
        let+ y = sub () in
        Expr.Value.select (Expr.Bool.value_lt a b) x y
    | 5 ->
        Expr.Builder.return
          (Expr.Value.value_of_index (index st ~in_scope ~depth:1))
    | 6 ->
        let+ a = sub () in
        Expr.Value.round_f32 a
    | _ ->
        (* [Builder.reduction] is the only way to bind a variable: it allocates,
           scopes the body, and threads the supply. *)
        let kind = pick st [ Expr.Reduction.Sum; Expr.Reduction.Max ] in
        let hi = int_in st ~lo:1 ~hi:3 in
        Expr.Builder.reduction ~kind ~lo:Expr.Index.zero
          ~hi:(Expr.Index.const hi) (fun r ->
            value st ~in_scope:(r :: in_scope) ~depth:(depth - 1))

let expr st ~depth = Expr.Builder.run (value st ~in_scope:[] ~depth)

(* Two well-formed fragments from supplies that OVERLAP: both start at
   [initial], so both mint ordinal 0. Neither references the other, so composing
   them shadows rather than captures -- which is the ordinary case, and the one
   freshening before composition repairs. *)
let colliding_pair () =
  let frag () =
    Expr.Builder.run
      (Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
         ~hi:(Expr.Index.const 3) (fun r ->
           Expr.Builder.return
             (Expr.Value.value_of_index (Expr.Index.of_position r))))
  in
  (frag (), frag ())

(* A fragment built from an INDEPENDENT supply that both binds a variable of its
   own and references [outer], which belongs to whoever will insert it.

   That combination is what makes capture observable. A fragment that merely
   shadows an unreferenced binder is a structural hazard [Check] flags, but it
   evaluates identically -- nothing looks the outer variable up. Capture bites
   only when a free reference meant for the destination's binder is swallowed by
   a colliding one inside the fragment. Since both supplies start at [initial],
   the ordinals collide by construction.

   The only place overlapping supplies are used on purpose. *)
let capturing_fragment outer =
  Expr.Builder.run
    (Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
       ~hi:(Expr.Index.const 3) (fun own ->
         Expr.Builder.return
           (Expr.Value.add
              (Expr.Value.value_of_index (Expr.Index.of_position outer))
              (Expr.Value.value_of_index (Expr.Index.of_position own)))))
