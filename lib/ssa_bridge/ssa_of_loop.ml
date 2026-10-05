open Ssa_ir
open Loop_ir
module B = Ssa_builder

type error = [ `Unsupported of Ssa_of_loop_unsupported.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unsupported u -> Ssa_of_loop_unsupported.pp fmt u

module Int_map = Map.Make (Int)

(* What each Loop name denotes at a point of the walk. Temporaries are keyed by
   their id, loop variables by theirs; float and index temporaries share the id
   space of [Loop_temp], so one map holds both. *)
type env = { temps : Ssa_value.t Int_map.t; vars : Ssa_value.t Int_map.t }

type ctx = {
  esc : error Err.Escape.t;
  b : B.t;
  buffers : (Ssa_buffer.t * Loop_buffer.t) Tensor_id.Map.t;
}

let refuse ctx construct =
  Err.Escape.throw ctx.esc
    (`Unsupported { Ssa_of_loop_unsupported.construct } : error)

let temp_key t = Loop_temp.to_int t
let var_key v = Loop_var.to_int v

let lookup ctx map key =
  match Int_map.find_opt key map with
  | Some v -> v
  | None -> refuse ctx Ssa_of_loop_unsupported.Unassigned_temp

let literal ctx n =
  let n = Int64.of_int n in
  if Ssa_const.in_index_domain n then n
  else refuse ctx Ssa_of_loop_unsupported.Index_literal

let rec index ctx env : Loop_index.t -> Ssa_type.index B.value = function
  | Loop_index.Add (a, b) ->
      let a = index ctx env a in
      let b = index ctx env b in
      B.index_add ctx.b a b
  | Loop_index.Const n -> B.index ctx.b (literal ctx n)
  | Loop_index.Scale (k, a) ->
      let a = index ctx env a in
      B.index_scale ctx.b (literal ctx k) a
  | Loop_index.Temp t -> B.as_index (lookup ctx env.temps (temp_key t))
  | Loop_index.Var v -> B.as_index (lookup ctx env.vars (var_key v))
  | Loop_index.Ceil_div_pos _ | Loop_index.Clamp_low _
  | Loop_index.Floor_div_pos _ | Loop_index.Max _ | Loop_index.Min _ ->
      refuse ctx Ssa_of_loop_unsupported.Index_operation

(* The axes in [Expr.Axis.all] order, as an explicit list: the evaluation order
   of a record's fields is not specified. *)
let coord ctx env (c : Loop_index.coord) =
  let components =
    List.map (fun a -> (a, index ctx env (Expr.Coord.get c a))) Expr.Axis.all
  in
  Expr.Coord.of_fn (fun a -> List.assoc a components)

let buffer ctx (b : Loop_buffer.t) =
  match Tensor_id.Map.find_opt b.Loop_buffer.id ctx.buffers with
  | Some (sb, _) -> sb
  | None ->
      let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
      refuse ctx (Ssa_of_loop_unsupported.Load_format (Payload.fmt_name f))

let decode ctx (sb : Ssa_buffer.t) =
  match sb.Ssa_buffer.format with
  | Ssa_format.Bool -> Ssa_op.Decode.Bool_to_f64
  | Ssa_format.F32 -> Ssa_op.Decode.F32_to_f64
  | Ssa_format.I64 -> refuse ctx Ssa_of_loop_unsupported.I64

let rec expr ctx env : float Loop_expr.t -> Ssa_type.f64 B.value = function
  | Loop_expr.Binary (op, a, b) ->
      let a = expr ctx env a in
      let b = expr ctx env b in
      B.f64_binary ctx.b op a b
  | Loop_expr.Const x -> B.f64 ctx.b x
  | Loop_expr.Load (b, c) ->
      let sb = buffer ctx b in
      let decode = decode ctx sb in
      B.load_f64 ctx.b sb.Ssa_buffer.id ~decode (B.Coord (coord ctx env c))
  | Loop_expr.Load_flat (b, o) ->
      let sb = buffer ctx b in
      let decode = decode ctx sb in
      B.load_f64 ctx.b sb.Ssa_buffer.id ~decode (B.Flat (index ctx env o))
  | Loop_expr.Round_f32 a ->
      let a = expr ctx env a in
      B.f32_to_f64 ctx.b (B.f64_to_f32 ctx.b a)
  | Loop_expr.Temp (Loop_carrier.Float, t) ->
      B.as_f64 (lookup ctx env.temps (temp_key t))
  | Loop_expr.Value_of_index i -> B.index_to_f64 ctx.b (index ctx env i)
  | Loop_expr.Array_get _ -> refuse ctx Ssa_of_loop_unsupported.Array
  | Loop_expr.Float_max _ -> refuse ctx Ssa_of_loop_unsupported.Float_max
  | Loop_expr.Fma _ -> refuse ctx Ssa_of_loop_unsupported.Fma
  | Loop_expr.I64_to_float _ -> refuse ctx Ssa_of_loop_unsupported.I64
  | Loop_expr.Select _ -> refuse ctx Ssa_of_loop_unsupported.Select
  | Loop_expr.Unary _ -> refuse ctx Ssa_of_loop_unsupported.Unary

(* Temporaries a statement list assigns, in id order, with nesting included. *)
let rec assigned acc : Loop_stmt.t list -> int list =
 fun stmts ->
  List.fold_left
    (fun acc -> function
      | Loop_stmt.Assign (_, t, _)
      | Loop_stmt.Assign_index (t, _)
      | Loop_stmt.Assign_index_of_i64 (t, _) ->
          temp_key t :: acc
      | Loop_stmt.For { body; _ } -> assigned acc body
      | Loop_stmt.If (_, yes, no) -> assigned (assigned acc yes) no
      | Loop_stmt.Reduce_sum { acc = a; body; _ } ->
          assigned (temp_key a :: acc) body
      | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Charge_scan_update
      | Loop_stmt.Fail_if _ | Loop_stmt.Mark _ | Loop_stmt.Release_scan_state _
      | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter
      | Loop_stmt.Store _ | Loop_stmt.Store_flat _ ->
          acc)
    acc stmts

let assigned_set stmts = List.sort_uniq Int.compare (assigned [] stmts)

let stored ctx env : Loop_stored.t -> Ssa_op.Encode.t * Ssa_type.f64 B.value =
  function
  | Loop_stored.Bool e -> (Ssa_op.Encode.Bool_nonzero, expr ctx env e)
  | Loop_stored.F32 e -> (Ssa_op.Encode.F32_round, expr ctx env e)
  | Loop_stored.I64 _ -> refuse ctx Ssa_of_loop_unsupported.I64

let rec stmts ctx env = List.fold_left (stmt ctx) env

and stmt ctx env : Loop_stmt.t -> env = function
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      let v = expr ctx env e in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Assign (Loop_carrier.Int64, _, _)
  | Loop_stmt.Assign_index_of_i64 _ ->
      refuse ctx Ssa_of_loop_unsupported.I64
  | Loop_stmt.Assign_index (t, i) ->
      let v = index ctx env i in
      { env with temps = Int_map.add (temp_key t) (v :> Ssa_value.t) env.temps }
  | Loop_stmt.Fail_if _ -> refuse ctx Ssa_of_loop_unsupported.Guard
  | Loop_stmt.For { var; lo; hi; body } ->
      let lo = index ctx env lo in
      let hi = index ctx env hi in
      let carried =
        List.filter (fun t -> Int_map.mem t env.temps) (assigned_set body)
      in
      let init = List.map (fun t -> Int_map.find t env.temps) carried in
      let results =
        B.for_dyn ctx.b ~lo ~hi ~init (fun b iv params ->
            let inner =
              {
                temps =
                  List.fold_left2
                    (fun m t p -> Int_map.add t p m)
                    env.temps carried params;
                vars = Int_map.add (var_key var) (iv :> Ssa_value.t) env.vars;
              }
            in
            let after = stmts { ctx with b } inner body in
            List.map (fun t -> Int_map.find t after.temps) carried)
      in
      {
        env with
        temps =
          List.fold_left2
            (fun m t r -> Int_map.add t r m)
            env.temps carried results;
      }
  | Loop_stmt.If _ -> refuse ctx Ssa_of_loop_unsupported.If
  | Loop_stmt.Mark m ->
      B.mark ctx.b
        (match m with
        | Loop_mark.Emitter -> Ssa_mark.Emitter
        | Loop_mark.Key -> Ssa_mark.Key
        | Loop_mark.Local -> Ssa_mark.Local
        | Loop_mark.Reduction -> Ssa_mark.Reduction
        | Loop_mark.Scan -> Ssa_mark.Scan
        | Loop_mark.Scan_update -> Ssa_mark.Scan_update);
      env
  | Loop_stmt.Reduce_sum { var; lo; hi; acc; seed; body; term; at = _ } ->
      let lo = index ctx env lo in
      let hi = index ctx env hi in
      let seed = B.f64 ctx.b seed in
      let sum =
        B.ordered_sum ctx.b ~lo ~hi ~seed (fun b iv ->
            B.mark b Ssa_mark.Reduction;
            let ctx = { ctx with b } in
            let inner =
              {
                env with
                vars = Int_map.add (var_key var) (iv :> Ssa_value.t) env.vars;
              }
            in
            let after = stmts ctx inner body in
            expr ctx after term)
      in
      {
        env with
        temps = Int_map.add (temp_key acc) (sum :> Ssa_value.t) env.temps;
      }
  | Loop_stmt.Store { buffer = b; coord = c; value } ->
      let sb = buffer ctx b in
      let encode, x = stored ctx env value in
      B.store_f64 ctx.b sb.Ssa_buffer.id ~encode (B.Coord (coord ctx env c)) x;
      env
  | Loop_stmt.Store_flat { buffer = b; offset; value } ->
      let sb = buffer ctx b in
      let encode, x = stored ctx env value in
      B.store_f64 ctx.b sb.Ssa_buffer.id ~encode
        (B.Flat (index ctx env offset))
        x;
      env
  | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ ->
      refuse ctx Ssa_of_loop_unsupported.Alloc
  | Loop_stmt.Charge_scan_update ->
      refuse ctx Ssa_of_loop_unsupported.Charge_scan
  | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _ ->
      refuse ctx Ssa_of_loop_unsupported.Scan_state
  | Loop_stmt.Reset_meter -> refuse ctx Ssa_of_loop_unsupported.Meter

let extents (sg : Tensor_sig.t) =
  Expr.Coord.of_fn (fun a ->
      Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape a)))

let format_of (sg : Tensor_sig.t) =
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  match f with
  | Payload.Bool -> Some Ssa_format.Bool
  | Payload.F32 -> Some Ssa_format.F32
  | Payload.I64 -> Some Ssa_format.I64
  | _ -> None

let role : Loop_buffer.role -> Ssa_buffer.role = function
  | Loop_buffer.Input -> Ssa_buffer.Input
  | Loop_buffer.Output -> Ssa_buffer.Output
  | Loop_buffer.Scratch -> Ssa_buffer.Scratch

let convert (p : Loop_program.t) =
  Err.Escape.with_escape @@ fun esc ->
  let declared =
    List.filter_map
      (fun (lb : Loop_buffer.t) ->
        Option.map
          (fun format ->
            ( {
                Ssa_buffer.id =
                  Ssa_id.Buffer.of_int (Tensor_id.to_int lb.Loop_buffer.id);
                extents = extents lb.Loop_buffer.sg;
                format;
                role = role lb.Loop_buffer.role;
              },
              lb ))
          (format_of lb.Loop_buffer.sg))
      p.Loop_program.buffers
  in
  let buffers =
    List.fold_left
      (fun m ((_, lb) as entry) -> Tensor_id.Map.add lb.Loop_buffer.id entry m)
      Tensor_id.Map.empty declared
  in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers:(List.map fst declared) (fun b ->
         let ctx = { esc; b; buffers } in
         ignore
           (stmts ctx
              { temps = Int_map.empty; vars = Int_map.empty }
              p.Loop_program.body)))
