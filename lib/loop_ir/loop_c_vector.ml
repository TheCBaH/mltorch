open Loop_c_base

(* ---- vector loops --------------------------------------------------------- *)

module V = Loop_vector

(* A vector expression as C text of type [v4df] (a mask as [v4di]). Vector
   temporaries are named by their id; a splatted scalar by its position among the
   loop's splats. *)
type vstate = { mutable splats : (float Loop_expr.t * string) list }

let lane_offset (a : V.Access.t) k : Loop_index.t =
  if k = 0 then a.V.Access.offset
  else
    Loop_index.Add (a.V.Access.offset, Loop_index.Const (k * a.V.Access.stride))

let vtemp t = "vt" ^ string_of_int (V.Temp.to_int t)

(* The element kind of a vector loop: four binary64 lanes ([v4df]) or
   [R.f32_lanes] binary32 ones ([v8sf]), with the helper prefix that goes with
   them. *)
module Vkind = struct
  type t = { ty : string; p : string; n : int }
end

let vkind nm : Vkind.t =
  if nm.f32 then
    {
      Vkind.ty = Printf.sprintf "v%dsf" R.f32_lanes;
      p = "vs_";
      n = R.f32_lanes;
    }
  else { Vkind.ty = "v4df"; p = "vf_"; n = 4 }

let vload nm (a : V.Access.t) =
  let b = a.V.Access.buffer in
  let k = vkind nm in
  let scalar i = load_cell nm b (Flat (lane_offset a i)) in
  let lanes () = String.concat ", " (List.init k.n scalar) in
  if a.V.Access.stride = 0 then k.p ^ "splat(" ^ scalar 0 ^ ")"
  else if a.V.Access.stride <> 1 then "((" ^ k.ty ^ "){" ^ lanes () ^ "})"
  else
    let at = "(" ^ buffer nm b ^ " + " ^ index nm a.V.Access.offset ^ ")" in
    match (fmt_of b, nm.f32) with
    | "f32", true -> "vs_load(" ^ at ^ ")"
    | "f32", false -> "vf_load_f32(" ^ at ^ ")"
    | "f64", false -> "vf_load_f64(" ^ at ^ ")"
    | "i32", false -> "vf_load_i32(" ^ at ^ ")"
    | _ -> "((" ^ k.ty ^ "){" ^ lanes () ^ "})"

let rec vexpr nm vs (e : V.t) : string =
  match e with
  | V.Const x -> (vkind nm).p ^ "splat(" ^ lit nm x ^ ")"
  | V.Splat s -> (
      match List.assq_opt s vs.splats with
      | Some name -> name
      | None -> (vkind nm).p ^ "splat(" ^ num nm s ^ ")")
  | V.Binary (op, a, b) ->
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      "(" ^ a ^ " " ^ binary_sym op ^ " " ^ b ^ ")"
  | V.Fma (a, b, c) ->
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      let c = vexpr nm vs c in
      use nm R.Name.Vector_prelude_fma_f32;
      (vkind nm).p ^ "fma(" ^ a ^ ", " ^ b ^ ", " ^ c ^ ")"
  | V.Float_max (a, b) ->
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      (vkind nm).p ^ "max(" ^ a ^ ", " ^ b ^ ")"
  | V.Round_f32 a ->
      (* the identity in binary32: the lanes are already single precision *)
      if nm.f32 then vexpr nm vs a else "vf_round_f32(" ^ vexpr nm vs a ^ ")"
  | V.Unary (op, a) ->
      let a = vexpr nm vs a in
      (match op with
      | Expr.Value.Erf -> use nm (if nm.f32 then R.Name.Erf_f32 else R.Name.Erf)
      | _ -> ());
      let name =
        (vkind nm).p
        ^
        match op with
        | Expr.Value.Cos -> "cos"
        | Expr.Value.Erf -> "erf"
        | Expr.Value.Exp -> "exp"
        | Expr.Value.Log -> "log"
        | Expr.Value.Sin -> "sin"
        | Expr.Value.Sqrt -> "sqrt"
        | Expr.Value.Trunc -> "trunc"
      in
      name ^ "(" ^ a ^ ")"
  | V.Select (m, a, b) ->
      let m = vmask nm vs m in
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      (vkind nm).p ^ "sel(" ^ m ^ ", " ^ a ^ ", " ^ b ^ ")"
  | V.Temp t -> vtemp t
  | V.Index_value { base; step } ->
      let k = vkind nm in
      "((" ^ k.ty ^ "){"
      ^ String.concat ", "
          (List.init k.n (fun i ->
               "(" ^ float_type nm ^ ")"
               ^ index nm (Loop_index.Add (base, Loop_index.Const (i * step)))))
      ^ "})"
  | V.Load a -> vload nm a

and vmask nm vs (m : V.mask) : string =
  match m with
  | V.Not m -> "(~" ^ vmask nm vs m ^ ")"
  | V.Or (a, b) -> "(" ^ vmask nm vs a ^ " | " ^ vmask nm vs b ^ ")"
  | V.Value_eq (a, b) -> "(" ^ vexpr nm vs a ^ " == " ^ vexpr nm vs b ^ ")"
  | V.Value_lt (a, b) -> "(" ^ vexpr nm vs a ^ " < " ^ vexpr nm vs b ^ ")"
  | V.Pool_better (best, value) ->
      (* The candidate wins on strict greater-than or on NaN. Both operands are
         pure, so naming the value twice is exact. *)
      let best = vexpr nm vs best in
      let value = vexpr nm vs value in
      "((" ^ value ^ " > " ^ best ^ ") | (" ^ value ^ " != " ^ value ^ "))"

let vstore nm vs ~ind (a : V.Access.t) (value : V.stored) =
  let b = a.V.Access.buffer in
  let e = match value with V.F32 e | V.Bool e -> e in
  let v = vexpr nm vs e in
  let k = vkind nm in
  match value with
  | V.F32 _ when a.V.Access.stride = 1 ->
      [
        Printf.sprintf "%s%s(%s + %s, %s);" ind
          (if nm.f32 then "vs_store" else "vf_store_f32")
          (buffer nm b)
          (index nm a.V.Access.offset)
          v;
      ]
  | _ ->
      let lane i =
        let cell = buffer nm b ^ "[" ^ index nm (lane_offset a i) ^ "]" in
        match value with
        | V.F32 _ -> Printf.sprintf "%s%s = (float)vs[%d];" ind cell i
        | V.Bool _ ->
            Printf.sprintf "%s%s = vs[%d] != %s ? 1 : 0;" ind cell i (lit nm 0.)
      in
      [ Printf.sprintf "%s{ const %s vs = %s;" ind k.ty v ]
      @ List.init k.n lane
      @ [ ind ^ "}" ]

module VF = Loop_vector_facts

(* A splat computed once, before the loop, is one that no enclosing inner loop's
   variable reaches; any other is evaluated where it is used. *)
let hoistable inner e =
  not (List.exists (fun v -> VF.expr_depends v Loop_temp.Set.empty e) inner)

let rec collect_splats ~inner acc (e : V.t) =
  match e with
  | V.Splat s ->
      if List.memq s acc || not (hoistable inner s) then acc else acc @ [ s ]
  | V.Binary (_, a, b) | V.Float_max (a, b) ->
      collect_splats ~inner (collect_splats ~inner acc a) b
  | V.Fma (a, b, c) ->
      collect_splats ~inner
        (collect_splats ~inner (collect_splats ~inner acc a) b)
        c
  | V.Round_f32 a | V.Unary (_, a) -> collect_splats ~inner acc a
  | V.Select (m, a, b) ->
      collect_splats ~inner
        (collect_splats ~inner (mask_splats ~inner acc m) a)
        b
  | V.Const _ | V.Index_value _ | V.Load _ | V.Temp _ -> acc

and mask_splats ~inner acc (m : V.mask) =
  match m with
  | V.Not m -> mask_splats ~inner acc m
  | V.Or (a, b) -> mask_splats ~inner (mask_splats ~inner acc a) b
  | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
      collect_splats ~inner (collect_splats ~inner acc a) b

let rec stmt_splats ~inner acc (s : V.stmt) =
  match s with
  | V.Assign (_, e) -> collect_splats ~inner acc e
  | V.Store { value = V.F32 e | V.Bool e; _ } -> collect_splats ~inner acc e
  | V.Index_assign _ | V.Mark _ -> acc
  | V.Inner { var; body; _ } ->
      List.fold_left (stmt_splats ~inner:(var :: inner)) acc body

let rec assigned_vtemps acc (s : V.stmt) =
  match s with
  | V.Assign (t, _) -> if List.mem t acc then acc else acc @ [ t ]
  | V.Inner { body; _ } -> List.fold_left assigned_vtemps acc body
  | V.Index_assign _ | V.Mark _ | V.Store _ -> acc

let rec vstmt nm vs ~ind (s : V.stmt) : string list =
  match s with
  | V.Assign (t, e) ->
      [ Printf.sprintf "%s%s = %s;" ind (vtemp t) (vexpr nm vs e) ]
  | V.Store { access; value } -> vstore nm vs ~ind access value
  | V.Index_assign (t, i) ->
      [ Printf.sprintf "%s%s = %s;" ind (index_temp nm t) (index nm i) ]
  | V.Mark _ -> []
  | V.Inner { var = v; lo; hi; body } ->
      let name = var nm v in
      [
        Printf.sprintf "%sfor (int64_t %s = %s; %s < %s; %s++) {" ind name
          (index nm lo) name (index nm hi) name;
      ]
      @ List.concat_map (vstmt nm vs ~ind:(ind ^ "  ")) body
      @ [ ind ^ "}" ]

let vloop nm ~depth ~scalar_stmt (l : V.loop) : string list =
  let ind = indent depth in
  let k = vkind nm in
  if l.V.lanes <> k.n then
    invalid_arg "Loop_c: a vector loop whose lanes the element kind cannot hold";
  if nm.f32 then (
    use nm R.Name.Vector_prelude_f32;
    use nm R.Name.Float_max_f32;
    use nm R.Name.Erf_f32)
  else (
    use nm R.Name.Vector_prelude;
    use nm R.Name.Float_max;
    use nm R.Name.Erf);
  let lo, hi =
    match (l.V.lo, l.V.hi) with
    | Loop_index.Const lo, Loop_index.Const hi -> (lo, hi)
    | _ -> invalid_arg "Loop_c: a vector loop without constant bounds"
  in
  let stop = lo + (max 0 (hi - lo) / l.V.lanes * l.V.lanes) in
  let vs = { splats = [] } in
  let splat_exprs = List.fold_left (stmt_splats ~inner:[]) [] l.V.body in
  let prelude =
    List.mapi
      (fun i e ->
        let name = Printf.sprintf "vs%d" i in
        vs.splats <- vs.splats @ [ (e, name) ];
        Printf.sprintf "%s  const %s %s = %ssplat(%s);" ind k.ty name k.p
          (num nm e))
      splat_exprs
  in
  let iv = var nm l.V.var in
  let decls =
    List.map
      (fun t ->
        Printf.sprintf "%s    %s %s = %ssplat(%s); (void)%s;" ind k.ty (vtemp t)
          k.p (lit nm 0.) (vtemp t))
      (List.fold_left assigned_vtemps [] l.V.body)
  in
  let body = List.concat_map (vstmt nm vs ~ind:(ind ^ "    ")) l.V.body in
  let remainder =
    match l.V.scalar with
    | Loop_stmt.For f ->
        scalar_stmt (Loop_stmt.For { f with lo = Loop_index.Const stop })
    | s -> scalar_stmt s
  in
  [ ind ^ "{" ]
  @ prelude
  @ [
      Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s += %d) {" ind iv
        (int_lit lo) iv (int_lit stop) iv l.V.lanes;
    ]
  @ decls @ body
  @ [ ind ^ "  }" ]
  @ remainder
  @ [ ind ^ "}" ]

(* A scheduled sum ({!Loop_vector.Reduction}): [parts] vector accumulators over the
   main rounds, the leftover vectors, the adjacent-pair trees (accumulators
   first, then lanes), the sequential tail and the seed, in exactly the order the
   definition gives. The sum's variable is a C local the rounds step. *)
let vreduction nm ~depth (r : V.Reduction.t) : string list =
  let ind = indent depth in
  let k = vkind nm in
  if r.V.Reduction.lanes <> k.n then
    invalid_arg "Loop_c: a reduction whose lanes the element kind cannot hold";
  if nm.f32 then (
    use nm R.Name.Vector_prelude_f32;
    use nm R.Name.Float_max_f32;
    use nm R.Name.Erf_f32)
  else (
    use nm R.Name.Vector_prelude;
    use nm R.Name.Float_max;
    use nm R.Name.Erf);
  let lanes = r.V.Reduction.lanes and parts = r.V.Reduction.parts in
  let lo = r.V.Reduction.lo in
  let n = r.V.Reduction.hi - lo in
  let full = n / lanes in
  let main = full / parts and extra = full mod parts in
  let tail = n - (full * lanes) in
  let vs = { splats = [] } in
  let iv = var nm r.V.Reduction.var in
  let acc j = Printf.sprintf "va%d" j in
  (* One accumulate on a vector accumulator [a]: [a + term], or a fused
     multiply-add of the term's factors when the sum is fused. *)
  let accumulate a delta =
    let at e = vexpr nm vs (V.shift r.V.Reduction.var delta e) in
    match (r.V.Reduction.fused, r.V.Reduction.term) with
    | true, V.Binary (Expr.Value.Mul, x, y) ->
        use nm R.Name.Vector_prelude_fma_f32;
        Printf.sprintf "%sfma(%s, %s, %s)" k.p (at x) (at y) a
    | true, _ -> invalid_arg "Loop_c: a fused reduction without a product"
    | false, _ -> Printf.sprintf "%s + %s" a (at r.V.Reduction.term)
  in
  (* The same on a scalar accumulator, for the sequential tail. *)
  let scalar_accumulate a =
    let lane e =
      num nm
        (Loop_vector_expand.lane_expr ~var:r.V.Reduction.var
           ~base:(Loop_index.Var r.V.Reduction.var)
           ~temp:(fun _ _ ->
             invalid_arg "Loop_c: a reduction term has no temporaries")
           e 0)
    in
    match (r.V.Reduction.fused, r.V.Reduction.term) with
    | true, V.Binary (Expr.Value.Mul, x, y) ->
        Printf.sprintf "fmaf(%s, %s, %s)" (lane x) (lane y) a
    | true, _ -> invalid_arg "Loop_c: a fused reduction without a product"
    | false, _ -> Printf.sprintf "%s + %s" a (lane r.V.Reduction.term)
  in
  let rec tree = function
    | [] -> invalid_arg "Loop_c.tree"
    | [ x ] -> x
    | xs ->
        let rec pairs = function
          | a :: b :: rest -> ("(" ^ a ^ " + " ^ b ^ ")") :: pairs rest
          | rest -> rest
        in
        tree (pairs xs)
  in
  let decls =
    List.init parts (fun j ->
        Printf.sprintf "%s  %s %s = %ssplat(%s);" ind k.ty (acc j) k.p
          (lit nm 0.))
  in
  let rounds =
    if main = 0 then []
    else
      [
        Printf.sprintf "%s  for (%s = %s; %s < %s; %s += %d) {" ind iv
          (int_lit lo) iv
          (int_lit (lo + (main * parts * lanes)))
          iv (parts * lanes);
      ]
      @ List.init parts (fun j ->
          Printf.sprintf "%s    %s = %s;" ind (acc j)
            (accumulate (acc j) (j * lanes)))
      @ [ ind ^ "  }" ]
  in
  let leftover =
    List.concat
      (List.init extra (fun e ->
           [
             Printf.sprintf "%s  %s = %s;" ind iv
               (int_lit (lo + (main * parts * lanes) + (e * lanes)));
             Printf.sprintf "%s  %s = %s;" ind (acc e) (accumulate (acc e) 0);
           ]))
  in
  let combined =
    Printf.sprintf "%s  const %s vb = %s;" ind k.ty (tree (List.init parts acc))
  in
  let horizontal =
    Printf.sprintf "%s  %s vh = %s;" ind (float_type nm)
      (tree (List.init lanes (fun i -> Printf.sprintf "vb[%d]" i)))
  in
  let tail_stmts =
    Printf.sprintf "%s  %s vt = %s;" ind (float_type nm) (lit nm 0.)
    :: List.concat
         (List.init tail (fun u ->
              [
                Printf.sprintf "%s  %s = %s;" ind iv
                  (int_lit (lo + (full * lanes) + u));
                Printf.sprintf "%s  vt = %s;" ind (scalar_accumulate "vt");
              ]))
  in
  [ ind ^ "{"; Printf.sprintf "%s  int64_t %s;" ind iv ]
  @ decls @ rounds @ leftover @ [ combined; horizontal ] @ tail_stmts
  @ [
      Printf.sprintf "%s  %s = %s + (vh + vt);" ind
        (temp nm r.V.Reduction.acc)
        (lit nm r.V.Reduction.seed);
      ind ^ "}";
    ]
