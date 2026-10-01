module V = Loop_vector

module Reason = struct
  type t =
    | Bad_lanes of int
    | Index_value_step_mismatch of { step : int; coefficient : int }
    | Non_constant_bounds
    | Offset_not_affine
    | Splat_depends_on_loop_variable
    | Splat_loads_stored_buffer of Loop_buffer.t
    | Store_through_broadcast
    | Store_loaded_elsewhere of Loop_buffer.t
    | Stores_overlap of Loop_buffer.t
    | Stride_mismatch of { stride : int; coefficient : int }
    | Temp_read_before_assigned of Loop_vector.Temp.t
    | Temp_assigned_twice of Loop_vector.Temp.t
    | Unsupported_load_format of Loop_buffer.t

  let pp ppf = function
    | Bad_lanes n -> Fmt.pf ppf "%d lanes (at least two are needed)" n
    | Index_value_step_mismatch { step; coefficient } ->
        Fmt.pf ppf
          "index value steps by %d, the loop variable's coefficient is %d" step
          coefficient
    | Non_constant_bounds -> Fmt.string ppf "the loop bounds are not constants"
    | Offset_not_affine ->
        Fmt.string ppf "an access offset is not affine in the loop variable"
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
    | Temp_assigned_twice t ->
        Fmt.pf ppf "vector temporary %a is assigned twice" V.Temp.pp t
    | Unsupported_load_format b ->
        Fmt.pf ppf "buffer t%d has a format a vector load does not decode"
          (Tensor_id.to_int b.Loop_buffer.id)
end

type error = [ `Vector_invalid of Reason.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Vector_invalid r -> Fmt.pf ppf "invalid vector program: %a" Reason.pp r

let ( let* ) = Result.bind

module F = Loop_vector_facts

let assigned_temps = F.assigned_temps
let expr_depends = F.expr_depends
let format_ok = F.format_ok

let coefficient var offset =
  match F.coefficient var offset with
  | Ok c -> Ok c
  | Error `Not_affine -> Error Reason.Offset_not_affine

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
              else access a
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
        let rec loads acc (e : V.t) =
          match e with
          | V.Load a -> a :: acc
          | V.Binary (_, a, b) | V.Float_max (a, b) -> loads (loads acc a) b
          | V.Round_f32 a | V.Unary (_, a) -> loads acc a
          | V.Select (m, a, b) -> loads (loads (mask_loads acc m) a) b
          | V.Const _ | V.Index_value _ | V.Splat _ | V.Temp _ -> acc
        and mask_loads acc (m : V.mask) =
          match m with
          | V.Not m -> mask_loads acc m
          | V.Or (a, b) -> mask_loads (mask_loads acc a) b
          | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
              loads (loads acc a) b
        in
        let* assigned, stores, reads =
          List.fold_left
            (fun acc (s : V.stmt) ->
              let* assigned, stores, reads = acc in
              match s with
              | V.Assign (t, e) ->
                  let* () = vexpr assigned e in
                  if V.Temp.Set.mem t assigned then
                    fail (Reason.Temp_assigned_twice t)
                  else Ok (V.Temp.Set.add t assigned, stores, loads reads e)
              | V.Store { access = a; value } ->
                  let e = match value with V.Bool e | V.F32 e -> e in
                  let* () = vexpr assigned e in
                  let* () = access a in
                  if a.V.Access.stride = 0 then
                    fail Reason.Store_through_broadcast
                  else Ok (assigned, a :: stores, loads reads e))
            (Ok (V.Temp.Set.empty, [], []))
            l.V.body
        in
        ignore assigned;
        (* A buffer stored and loaded must be touched at one identical access;
           two stores to a buffer must be identical too, so statement-major
           lane order cannot reorder overlapping writes. *)
        let same (a : V.Access.t) (b : V.Access.t) =
          Tensor_id.equal a.V.Access.buffer.Loop_buffer.id
            b.V.Access.buffer.Loop_buffer.id
          && a.V.Access.offset = b.V.Access.offset
          && a.V.Access.stride = b.V.Access.stride
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
                    reads
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
            (Ok ()) stores
        in
        stores_ok stores
    | _ -> fail Reason.Non_constant_bounds

let rec nodes ns =
  List.fold_left
    (fun acc n ->
      let* () = acc in
      match n with
      | V.Scalar _ -> Ok ()
      | V.Vector l -> loop l
      | V.If (_, a, b) ->
          let* () = nodes a in
          nodes b
      | V.Loop { body; _ } -> nodes body)
    (Ok ()) ns

let program (p : V.program) = Err.import Fun.id (nodes p.V.body)
