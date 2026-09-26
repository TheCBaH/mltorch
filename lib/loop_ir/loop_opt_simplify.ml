(* Index simplification against the enclosing loops' ranges: what
   [Loop_opt_fold] cannot do without them.

   - An index whose range is one point is that constant: [floor (i / 64)] for
     [i] in [0, 64).
   - [Max]/[Min]/[Clamp_low] whose operands' ranges are ordered pick the
     operand that always wins: [max (0, 2 * i)] is [2 * i] and
     [max (0, -2 * i)] is [0] for [i >= 0].
   - A division splits off the multiples of its divisor exactly,
     [floor ((d * q + r) / d) = q + floor (r / d)]: [floor (8 * i / 4)] is
     [2 * i].
   - An [Add]/[Scale] tree is rebuilt as a sum of scaled atoms when that is
     cheaper, which cancels [floor (i / 64) - floor (i / 64)].

   The rewrites are exact on the integers, so the one thing each has to keep
   is the overflow report (invariant 3): a node is rewritten only when
   [Loop_range.proven] holds for it, so no [Add]/[Scale] it removes could have
   overflowed, and for its replacement, so none it introduces can. The
   children are rewritten first, and a child is only ever replaced under the
   same proof, so a proof over the rewritten node covers the original's top
   operation too (it computes the same value). A result is kept only when it
   is cheaper ([cost]), so an index that is already simple is left alone. *)

(* A division or a comparison costs a call in the emitted JavaScript; an
   [Add]/[Scale] is one operator; a leaf is free. *)
let rec cost : Loop_index.t -> int = function
  | Loop_index.Add (a, b) -> 1 + cost a + cost b
  | Loop_index.Ceil_div_pos (a, _) | Loop_index.Floor_div_pos (a, _) ->
      3 + cost a
  | Loop_index.Clamp_low a -> 2 + cost a
  | Loop_index.Max (a, b) | Loop_index.Min (a, b) -> 2 + cost a + cost b
  | Loop_index.Scale (_, a) -> 1 + cost a
  | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> 0

let ( let* ) = Option.bind

(* [c = d * q + r] with [r] in [0, d), in [int64]: [d * q] of a constant near
   [-2^31] leaves a 32-bit [int]. Both results are in the domain, so they
   narrow. *)
let split_const c d =
  let c = Int64.of_int c and d = Int64.of_int d in
  let q = Int64.div c d and r = Int64.rem c d in
  let q, r =
    if Int64.compare r 0L < 0 then (Int64.pred q, Int64.add r d) else (q, r)
  in
  (Int64.to_int q, Int64.to_int r)

(* [div (d * q + r)] is [q + div r] for either rounding: the multiples of [d]
   leave the division exactly. [q] and [r] are split from the dividend's
   linear form, the constant into [d * floor (c / d)] and a remainder in
   [0, d). *)
let split_division round (a : Loop_index.t) d =
  let* f = Loop_linear.of_index a in
  let divisible, rest =
    List.partition (fun (_, c) -> c mod d = 0) f.Loop_linear.terms
  in
  let q0, r0 = split_const f.Loop_linear.const d in
  let quotient =
    {
      Loop_linear.terms = List.map (fun (x, c) -> (x, c / d)) divisible;
      const = q0;
    }
  in
  let remainder = { Loop_linear.terms = rest; const = r0 } in
  let div =
    match (remainder.Loop_linear.terms, round) with
    | [], `Floor -> Loop_linear.const 0
    | [], `Ceil -> Loop_linear.const (if remainder.const > 0 then 1 else 0)
    | _, `Floor ->
        Loop_linear.atom
          (Loop_index.Floor_div_pos (Loop_linear.to_index remainder, d))
    | _, `Ceil ->
        Loop_linear.atom
          (Loop_index.Ceil_div_pos (Loop_linear.to_index remainder, d))
  in
  let* sum = Loop_linear.add quotient div in
  Some (Loop_linear.to_index sum)

let point env (idx : Loop_index.t) =
  let r = Loop_range.of_index env idx in
  if
    Int64.equal r.Loop_range.lo r.Loop_range.hi
    && Loop_range.within ~inner:r ~outer:Loop_range.domain
  then Some (Loop_index.Const (Int64.to_int r.Loop_range.lo))
  else None

let candidate env (idx : Loop_index.t) : Loop_index.t option =
  let range = Loop_range.of_index env in
  match point env idx with
  | Some c -> Some c
  | None -> (
      match idx with
      | Loop_index.Max (a, b) ->
          let ra = range a and rb = range b in
          if Int64.compare ra.lo rb.hi >= 0 then Some a
          else if Int64.compare rb.lo ra.hi >= 0 then Some b
          else None
      | Loop_index.Min (a, b) ->
          let ra = range a and rb = range b in
          if Int64.compare ra.hi rb.lo <= 0 then Some a
          else if Int64.compare rb.hi ra.lo <= 0 then Some b
          else None
      | Loop_index.Clamp_low a ->
          let ra = range a in
          if Int64.compare ra.lo 0L >= 0 then Some a
          else if Int64.compare ra.hi 0L <= 0 then Some (Loop_index.Const 0)
          else None
      | Loop_index.Floor_div_pos (a, d) -> split_division `Floor a d
      | Loop_index.Ceil_div_pos (a, d) -> split_division `Ceil a d
      | Loop_index.Add _ | Loop_index.Scale _ ->
          let* f = Loop_linear.of_index idx in
          Some (Loop_linear.to_index f)
      | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> None)

let step ~proven env (idx : Loop_index.t) : Loop_index.t =
  if not (proven env idx) then idx
  else
    match candidate env idx with
    | Some r when cost r < cost idx && proven env r -> r
    | _ -> idx

let rec rewrite ~proven ~single env (stmts : Loop_stmt.t list) :
    Loop_stmt.t list =
  List.map
    (fun (stmt : Loop_stmt.t) ->
      let f = step ~proven env in
      let index = Loop_index_map.index ~f in
      match stmt with
      | Loop_stmt.For { var; lo; hi; body } ->
          let lo = index lo and hi = index hi in
          let env' =
            Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
          in
          Loop_stmt.For
            { var; lo; hi; body = rewrite ~proven ~single env' body }
      | Loop_stmt.If (p, a, b) ->
          let p = Loop_index_map.pred ~f p in
          Loop_stmt.If
            (p, rewrite ~proven ~single env a, rewrite ~proven ~single env b)
      | Loop_stmt.Assign_index (t, idx) ->
          let idx = index idx in
          if single t then
            Loop_range.Env.set_temp t (Loop_range.of_index env idx) env;
          Loop_stmt.Assign_index (t, idx)
      | s -> Loop_index_map.stmt ~f s)
    stmts

let with_proof proven program =
  let single = Loop_opt_scope.single_assignment program.Loop_program.body in
  {
    program with
    Loop_program.body =
      rewrite ~proven ~single (Loop_range.Env.create ())
        program.Loop_program.body;
  }

let run = with_proof Loop_range.proven
