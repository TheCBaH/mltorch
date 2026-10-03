module V = Loop_vector

module Reason = struct
  type t =
    | Bad_lanes of int
    | Bad_parts of int
    | Fused_term_not_a_product
    | Index_assignment_depends_on_loop_variable
    | Index_value_step_mismatch of { step : int; coefficient : int }
    | Inner_bounds_depend_on_loop_variable
    | Non_constant_bounds
    | Offset_not_affine
    | Reduction_too_short of { terms : int; lanes : int }
    | Splat_depends_on_loop_variable
    | Splat_loads_stored_buffer of Loop_buffer.t
    | Store_through_broadcast
    | Store_loaded_elsewhere of Loop_buffer.t
    | Stores_overlap of Loop_buffer.t
    | Stride_mismatch of { stride : int; coefficient : int }
    | Temp_read_before_assigned of Loop_vector.Temp.t
    | Unsupported_load_format of Loop_buffer.t

  let pp ppf = function
    | Bad_lanes n -> Fmt.pf ppf "%d lanes (at least two are needed)" n
    | Bad_parts n -> Fmt.pf ppf "%d accumulators (at least one is needed)" n
    | Fused_term_not_a_product ->
        Fmt.string ppf "a fused reduction whose term is not a product"
    | Index_assignment_depends_on_loop_variable ->
        Fmt.string ppf "an index temporary is assigned from the loop variable"
    | Inner_bounds_depend_on_loop_variable ->
        Fmt.string ppf "an inner loop's bounds depend on the loop variable"
    | Index_value_step_mismatch { step; coefficient } ->
        Fmt.pf ppf
          "index value steps by %d, the loop variable's coefficient is %d" step
          coefficient
    | Non_constant_bounds -> Fmt.string ppf "the loop bounds are not constants"
    | Offset_not_affine ->
        Fmt.string ppf "an access offset is not affine in the loop variable"
    | Reduction_too_short { terms; lanes } ->
        Fmt.pf ppf "a reduction of %d terms holds no full vector of %d lanes"
          terms lanes
    | Splat_depends_on_loop_variable ->
        Fmt.string ppf "a splat reads the loop variable or a loop temporary"
    | Splat_loads_stored_buffer b ->
        Fmt.pf ppf "a splat loads from buffer t%d, which the loop stores"
          (Tensor_id.to_int b.Loop_buffer.id)
    | Store_loaded_elsewhere b ->
        Fmt.pf ppf "buffer t%d is stored and loaded at different accesses"
          (Tensor_id.to_int b.Loop_buffer.id)
    | Stores_overlap b ->
        Fmt.pf ppf "buffer t%d is stored at two different accesses"
          (Tensor_id.to_int b.Loop_buffer.id)
    | Store_through_broadcast -> Fmt.string ppf "a store with stride zero"
    | Stride_mismatch { stride; coefficient } ->
        Fmt.pf ppf "stride %d, but the loop variable's coefficient is %d" stride
          coefficient
    | Temp_read_before_assigned t ->
        Fmt.pf ppf "vector temporary %a is read before it is assigned" V.Temp.pp
          t
    | Unsupported_load_format b ->
        Fmt.pf ppf "buffer t%d has a format a vector load does not decode"
          (Tensor_id.to_int b.Loop_buffer.id)
end

type error = [ `Vector_invalid of Reason.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Vector_invalid r -> Fmt.pf ppf "invalid vector program: %a" Reason.pp r

let ( let* ) = Result.bind

module F = Loop_vector_facts

let mentions = F.mentions
let assigned_temps = F.assigned_temps
let expr_depends = F.expr_depends
let format_ok = F.format_ok

let coefficient var offset =
  match F.coefficient var offset with
  | Ok c -> Ok c
  | Error `Not_affine -> Error Reason.Offset_not_affine

(* Whether two accesses to one buffer touch no common cell over the iterations
   of the loop: the same non-zero stride [c] and offsets that differ by a
   constant [d], so the cells meet only if [c] divides [d] and the quotient is
   within the iteration count. An offset that mentions an inner loop's variable
   spans more than the loop variable's iterations and is not proved. *)
let disjoint ~inner_vars ~iterations (a : V.Access.t) (b : V.Access.t) =
  let c = a.V.Access.stride in
  c <> 0 && c = b.V.Access.stride
  && Tensor_id.equal a.V.Access.buffer.Loop_buffer.id
       b.V.Access.buffer.Loop_buffer.id
  && (not
        (List.exists
           (fun v ->
             F.mentions v a.V.Access.offset || F.mentions v b.V.Access.offset)
           inner_vars))
  &&
  match
    ( Loop_linear.of_index a.V.Access.offset,
      Loop_linear.of_index b.V.Access.offset )
  with
  | Some la, Some lb -> (
      match Loop_linear.sub la lb with
      | Some { Loop_linear.terms = []; const = d } ->
          d mod c <> 0 || abs (d / c) >= iterations
      | _ -> false)
  | _ -> false

let loop (l : V.loop) =
  let fail r = Error (`Vector_invalid r) in
  let var = l.V.var in
  let lanes = l.V.lanes in
  if lanes < 2 then fail (Reason.Bad_lanes lanes)
  else
    match (l.V.lo, l.V.hi) with
    | Loop_index.Const _, Loop_index.Const _ ->
        let scalar_body =
          match l.V.scalar with Loop_stmt.For { body; _ } -> body | _ -> []
        in
        let loop_temps =
          List.fold_left assigned_temps Loop_temp.Set.empty scalar_body
        in
        let splat_buffers = ref [] in
        let stores = ref [] and reads = ref [] in
        let access (a : V.Access.t) =
          match coefficient var a.V.Access.offset with
          | Error r -> fail r
          | Ok c ->
              if c <> a.V.Access.stride then
                fail
                  (Reason.Stride_mismatch
                     { stride = a.V.Access.stride; coefficient = c })
              else Ok ()
        in
        let rec vexpr assigned (e : V.t) =
          match e with
          | V.Binary (_, a, b) | V.Float_max (a, b) ->
              let* () = vexpr assigned a in
              vexpr assigned b
          | V.Fma (a, b, c) ->
              let* () = vexpr assigned a in
              let* () = vexpr assigned b in
              vexpr assigned c
          | V.Const _ -> Ok ()
          | V.Index_value { base; step } -> (
              match coefficient var base with
              | Error r -> fail r
              | Ok c ->
                  if c <> step then
                    fail
                      (Reason.Index_value_step_mismatch
                         { step; coefficient = c })
                  else Ok ())
          | V.Load a ->
              if not (format_ok a.V.Access.buffer) then
                fail (Reason.Unsupported_load_format a.V.Access.buffer)
              else (
                reads := a :: !reads;
                access a)
          | V.Round_f32 a | V.Unary (_, a) -> vexpr assigned a
          | V.Select (m, a, b) ->
              let* () = mask assigned m in
              let* () = vexpr assigned a in
              vexpr assigned b
          | V.Splat s ->
              if expr_depends var loop_temps s then
                fail Reason.Splat_depends_on_loop_variable
              else (
                splat_buffers := F.expr_buffers s @ !splat_buffers;
                Ok ())
          | V.Temp t ->
              if V.Temp.Set.mem t assigned then Ok ()
              else fail (Reason.Temp_read_before_assigned t)
        and mask assigned (m : V.mask) =
          match m with
          | V.Not m -> mask assigned m
          | V.Or (a, b) ->
              let* () = mask assigned a in
              mask assigned b
          | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
              let* () = vexpr assigned a in
              vexpr assigned b
        in
        (* [assigned] is what is definitely assigned on entry: an inner loop's
           own assignments do not count after it, which may run no iterations. *)
        let rec body assigned stmts =
          List.fold_left
            (fun acc (s : V.stmt) ->
              let* assigned = acc in
              match s with
              | V.Assign (t, e) ->
                  let* () = vexpr assigned e in
                  Ok (V.Temp.Set.add t assigned)
              | V.Index_assign (_, i) ->
                  if mentions var i then
                    fail Reason.Index_assignment_depends_on_loop_variable
                  else Ok assigned
              | V.Mark _ -> Ok assigned
              | V.Inner { lo; hi; body = inner; _ } ->
                  if mentions var lo || mentions var hi then
                    fail Reason.Inner_bounds_depend_on_loop_variable
                  else
                    let* _ = body assigned inner in
                    Ok assigned
              | V.Store { access = a; value } ->
                  let e = match value with V.Bool e | V.F32 e -> e in
                  let* () = vexpr assigned e in
                  let* () = access a in
                  if a.V.Access.stride = 0 then
                    fail Reason.Store_through_broadcast
                  else (
                    stores := a :: !stores;
                    Ok assigned))
            (Ok assigned) stmts
        in
        let* _ = body V.Temp.Set.empty l.V.body in
        (* A buffer stored and loaded must be touched at one identical access,
           or at accesses whose cells provably never meet; two stores to a
           buffer likewise, so statement-major lane order cannot reorder
           overlapping writes. *)
        let lo_n, hi_n =
          match (l.V.lo, l.V.hi) with
          | Loop_index.Const a, Loop_index.Const b -> (a, b)
          | _ -> (0, 0)
        in
        let inner_vars =
          let rec go acc = function
            | V.Inner { var; body; _ } -> List.fold_left go (var :: acc) body
            | V.Assign _ | V.Index_assign _ | V.Mark _ | V.Store _ -> acc
          in
          List.fold_left go [] l.V.body
        in
        let same (a : V.Access.t) (b : V.Access.t) =
          Tensor_id.equal a.V.Access.buffer.Loop_buffer.id
            b.V.Access.buffer.Loop_buffer.id
          && (a.V.Access.offset = b.V.Access.offset
              && a.V.Access.stride = b.V.Access.stride
             || disjoint ~inner_vars ~iterations:(hi_n - lo_n) a b)
        in
        let rec stores_ok = function
          | [] -> Ok ()
          | (s : V.Access.t) :: rest ->
              let id = s.V.Access.buffer.Loop_buffer.id in
              let* () =
                if
                  List.exists
                    (fun (r : V.Access.t) ->
                      Tensor_id.equal r.V.Access.buffer.Loop_buffer.id id
                      && not (same r s))
                    !reads
                then fail (Reason.Store_loaded_elsewhere s.V.Access.buffer)
                else Ok ()
              in
              let* () =
                if
                  List.exists
                    (fun (r : V.Access.t) ->
                      Tensor_id.equal r.V.Access.buffer.Loop_buffer.id id
                      && not (same r s))
                    rest
                then fail (Reason.Stores_overlap s.V.Access.buffer)
                else Ok ()
              in
              stores_ok rest
        in
        let* () =
          List.fold_left
            (fun acc (s : V.Access.t) ->
              let* () = acc in
              match
                List.find_opt
                  (fun (b : Loop_buffer.t) ->
                    Tensor_id.equal b.Loop_buffer.id
                      s.V.Access.buffer.Loop_buffer.id)
                  !splat_buffers
              with
              | Some b -> fail (Reason.Splat_loads_stored_buffer b)
              | None -> Ok ())
            (Ok ()) !stores
        in
        stores_ok !stores
    | _ -> fail Reason.Non_constant_bounds

(* A scheduled sum is a vector loop with no stores: its term is checked as a loop
   body would be (affine accesses, splats independent of the sum's variable), so
   every load a lane makes is a load the scalar loop makes at that term. *)
let reduction (r : V.Reduction.t) =
  let fail r = Error (`Vector_invalid r) in
  if r.V.Reduction.parts < 1 then fail (Reason.Bad_parts r.V.Reduction.parts)
  else if r.V.Reduction.lanes < 2 then
    fail (Reason.Bad_lanes r.V.Reduction.lanes)
  else
    let terms = r.V.Reduction.hi - r.V.Reduction.lo in
    if
      r.V.Reduction.fused
      &&
      match r.V.Reduction.term with
      | V.Binary (Expr.Value.Mul, _, _) -> false
      | _ -> true
    then fail Reason.Fused_term_not_a_product
    else if terms < r.V.Reduction.lanes then
      fail (Reason.Reduction_too_short { terms; lanes = r.V.Reduction.lanes })
    else
      loop
        {
          V.var = r.V.Reduction.var;
          lo = Loop_index.Const r.V.Reduction.lo;
          hi = Loop_index.Const r.V.Reduction.hi;
          lanes = r.V.Reduction.lanes;
          body = [ V.Assign (V.Temp.of_int 0, r.V.Reduction.term) ];
          scalar = r.V.Reduction.scalar;
        }

let rec nodes ns =
  List.fold_left
    (fun acc n ->
      let* () = acc in
      match n with
      | V.Scalar _ -> Ok ()
      | V.Reduction r -> reduction r
      | V.Vector l -> loop l
      | V.If (_, a, b) ->
          let* () = nodes a in
          nodes b
      | V.Loop { body; _ } -> nodes body)
    (Ok ()) ns

let program (p : V.program) = Err.import Fun.id (nodes p.V.body)
