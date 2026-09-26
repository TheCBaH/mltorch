(* ---- relational bounds (clamped windows) --------------------------------

   A window loop is [for k in [max (0, a), min (K, b))] and the access inside
   it is at [base + d * k]; its bounds check follows from the loop's bounds
   only relationally ([k >= -base] makes [base + k >= 0]), which an interval
   over [k] and [base] separately cannot see. So a variable's bounds are kept
   as facts [d * k >= a] / [d * k <= a], and a bound on a coordinate's linear
   form is proven by replacing [c * k] with the matching fact's side and
   checking what is left by interval. The loop's own bounds are the proof: no
   claim an op's author made ([Expr.Index.Assume_position]) is trusted.

   A fact may mention only loop variables. They are immutable in the loop's
   body, so the bound's value at the loop's entry is its value at the guard;
   an index temporary could have been reassigned in between. *)

type fact = { d : int; bound : Loop_linear.t }

let rec mentions_temp : Loop_index.t -> bool = function
  | Loop_index.Temp _ -> true
  | Loop_index.Var _ | Loop_index.Const _ -> false
  | Loop_index.Add (a, b) | Loop_index.Max (a, b) | Loop_index.Min (a, b) ->
      mentions_temp a || mentions_temp b
  | Loop_index.Ceil_div_pos (a, _)
  | Loop_index.Clamp_low a
  | Loop_index.Floor_div_pos (a, _)
  | Loop_index.Scale (_, a) ->
      mentions_temp a

let ( let* ) = Option.bind

(* A bound [e + c], [e] the one atom of the form at coefficient 1, split out
   so a division atom can become a fact on [d * k]. *)
let single_atom f =
  match f.Loop_linear.terms with
  | [ (a, 1) ] -> Some (a, f.Loop_linear.const)
  | _ -> None

(* [k >= lo]: [max]/[clamp_low] give one fact per operand, and
   [k >= ceil (x / d) + c] gives [d * k >= x + d * c]. *)
let rec lower_facts (lo : Loop_index.t) : fact list =
  match lo with
  | Loop_index.Clamp_low a ->
      { d = 1; bound = Loop_linear.const 0 } :: lower_facts a
  | Loop_index.Max (a, b) -> lower_facts a @ lower_facts b
  | _ -> (
      match Loop_linear.of_index lo with
      | None -> []
      | Some f -> (
          let plain = [ { d = 1; bound = f } ] in
          match single_atom f with
          | Some (Loop_index.Ceil_div_pos (x, d), c) -> (
              match
                let* x = Loop_linear.of_index x in
                let* dc = Loop_linear.scale d (Loop_linear.const c) in
                Loop_linear.add x dc
              with
              | Some bound -> { d; bound } :: plain
              | None -> plain)
          | _ -> plain))

(* [k < hi], so [k <= hi - 1]: [min] gives one fact per operand, and
   [k <= floor (x / d) + c - 1] gives [d * k <= x + d * (c - 1)]. [slack] is
   the [1] of [hi - 1]; the mutation test widens it. *)
let rec upper_facts ~slack (hi : Loop_index.t) : fact list =
  match hi with
  | Loop_index.Min (a, b) -> upper_facts ~slack a @ upper_facts ~slack b
  | _ -> (
      match Loop_linear.of_index hi with
      | None -> []
      | Some f -> (
          let plain =
            match Loop_linear.add f (Loop_linear.const (-slack)) with
            | Some bound -> [ { d = 1; bound } ]
            | None -> []
          in
          match single_atom f with
          | Some (Loop_index.Floor_div_pos (x, d), c) -> (
              match
                let* x = Loop_linear.of_index x in
                let* dc = Loop_linear.scale d (Loop_linear.const (c - slack)) in
                Loop_linear.add x dc
              with
              | Some bound -> { d; bound } :: plain
              | None -> plain)
          | _ -> plain))

type bounds = { lower : fact list; upper : fact list }

let bounds_of ~slack lo hi =
  let keep =
    List.filter (fun f ->
        not
          (List.exists
             (fun (a, _) -> mentions_temp a)
             f.bound.Loop_linear.terms))
  in
  if mentions_temp lo || mentions_temp hi then { lower = []; upper = [] }
  else { lower = keep (lower_facts lo); upper = keep (upper_facts ~slack hi) }

(* [f >= 0] (for [`Lower]) or [f <= 0] (for [`Upper]). With [c * k] in [f]
   and a fact [d * k >= a] (or [<= a]) where [c = m * d], [c * k] is bounded
   by [m * a] on the side the sign of [m] picks, so it is replaced by that. *)
let rec provable ~depth env vars side (f : Loop_linear.t) =
  let r = Loop_linear.range env f in
  (match side with
    | `Lower -> Int64.compare r.Loop_range.lo 0L >= 0
    | `Upper -> Int64.compare r.Loop_range.hi 0L <= 0)
  || depth > 0
     && List.exists
          (fun (atom, c) ->
            match atom with
            | Loop_index.Var v -> (
                match Loop_var.Map.find_opt v vars with
                | None -> false
                | Some b ->
                    let facts =
                      match (side, c > 0) with
                      | `Lower, true | `Upper, false -> b.lower
                      | `Lower, false | `Upper, true -> b.upper
                    in
                    List.exists
                      (fun { d; bound } ->
                        c mod d = 0
                        &&
                        match
                          let* without =
                            Loop_linear.sub f
                              { Loop_linear.terms = [ (atom, c) ]; const = 0 }
                          in
                          let* by = Loop_linear.scale (c / d) bound in
                          Loop_linear.add without by
                        with
                        | Some f' ->
                            provable ~depth:(depth - 1) env vars side f'
                        | None -> false)
                      facts)
            | _ -> false)
          f.Loop_linear.terms

let in_extent env vars idx extent =
  match Loop_linear.of_index idx with
  | None -> false
  | Some f -> (
      provable ~depth:2 env vars `Lower f
      &&
      match Loop_linear.add f (Loop_linear.const (1 - extent)) with
      | Some g -> provable ~depth:2 env vars `Upper g
      | None -> false)

(* [Loop_lower_index.load_guard] is the only producer of a multi-axis bounds
   check, and it builds one as a left fold of [Or] over [Out_of_range] --
   never mixing in an [Index_overflows] or anything else -- so recursing
   through [Or] alone (never [Not]) is a complete match for what the
   lowering emits, not an arbitrary generalization. *)
let rec never_fires env vars (pred : Loop_expr.pred) =
  match pred with
  | Loop_bool.Index_overflows idx -> Loop_range.proven env idx
  | Loop_bool.Out_of_range (idx, extent) ->
      Loop_range.within
        ~inner:(Loop_range.of_index env idx)
        ~outer:{ Loop_range.lo = 0L; hi = Int64.of_int (extent - 1) }
      || in_extent env vars idx extent
  | Loop_bool.Or (a, b) -> never_fires env vars a && never_fires env vars b
  | _ -> false

let rec rewrite ~slack ~single env vars (stmts : Loop_stmt.t list) :
    Loop_stmt.t list =
  let continue rest = rewrite ~slack ~single env vars rest in
  match stmts with
  | [] -> []
  | stmt :: rest -> (
      match stmt with
      | Loop_stmt.Fail_if (pred, _) when never_fires env vars pred ->
          continue rest
      | Loop_stmt.For { var; lo; hi; body } ->
          let env' =
            Loop_range.Env.add_var var (Loop_opt_scope.var_range env lo hi) env
          in
          let vars' = Loop_var.Map.add var (bounds_of ~slack lo hi) vars in
          let body = rewrite ~slack ~single env' vars' body in
          Loop_stmt.For { var; lo; hi; body } :: continue rest
      | Loop_stmt.If (p, a, b) ->
          let a = rewrite ~slack ~single env vars a in
          let b = rewrite ~slack ~single env vars b in
          Loop_stmt.If (p, a, b) :: continue rest
      | Loop_stmt.Assign_index (t, idx) ->
          if single t then
            Loop_range.Env.set_temp t (Loop_range.of_index env idx) env;
          stmt :: continue rest
      | s -> s :: continue rest)

let with_upper_slack slack program =
  let single = Loop_opt_scope.single_assignment program.Loop_program.body in
  {
    program with
    Loop_program.body =
      rewrite ~slack ~single (Loop_range.Env.create ()) Loop_var.Map.empty
        program.Loop_program.body;
  }

let run = with_upper_slack 1
