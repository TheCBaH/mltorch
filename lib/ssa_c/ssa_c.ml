open Ssa_ir
module R = Loop_ir.Loop_c_runtime
module F = Loop_ir.Loop_js_failure
open Ssa_c_ctx

type error = Ssa_c_ctx.error

let pp_error = Ssa_c_ctx.pp_error

let meter_op cx depth ~limits (op : Ssa_op.t) =
  match op with
  | Ssa_op.Meter_charge ->
      line cx depth "if (scan_remaining <= 0) %s"
        (Loop_ir.Loop_c_base.meter_failure F.Meter.Updates_exhausted
           (i64_lit (Expr.Scan_limits.max_updates limits)));
      line cx depth "scan_remaining -= 1;"
  | Ssa_op.Meter_release width ->
      line cx depth "scan_live -= %Ld;" (Int64.mul 2L width)
  | Ssa_op.Meter_reserve width ->
      let need = Int64.mul 2L width in
      let max_state = Expr.Scan_limits.max_state limits in
      line cx depth "if (scan_live + %Ld > %s) %s" need (int_lit max_state)
        (Loop_ir.Loop_c_base.meter_failure F.Meter.State_over_limit
           (string_of_int max_state));
      line cx depth "scan_live += %Ld;" need
  | Ssa_op.Meter_reset ->
      line cx depth "scan_remaining = %s;"
        (i64_lit (Expr.Scan_limits.max_updates limits));
      line cx depth "scan_live = 0;"
  | _ -> invalid_arg "Ssa_c.meter_op"

let is_meter (op : Ssa_op.t) =
  match op with
  | Ssa_op.Meter_charge | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _
  | Ssa_op.Meter_reset ->
      true
  | _ -> false

let rec uses_meter (r : Ssa_region.t) =
  List.exists
    (function
      | Ssa_stmt.Instr i -> is_meter i.Ssa_instr.op
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          uses_meter body
      | Ssa_stmt.If { then_; else_; _ } -> uses_meter then_ || uses_meter else_)
    r.Ssa_region.body

(* The buffers an operation names, in declaration order: a declared buffer no
   operation touches is validated by the caller's binding and is no argument
   of the kernel, as in the Loop program's own buffer list. *)
let arguments (p : Ssa_program.t) =
  let seen = ref Ssa_id.Buffer.Set.empty in
  let note id = seen := Ssa_id.Buffer.Set.add id !seen in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr i -> (
        match i.Ssa_instr.op with
        | Ssa_op.Check_access { buffer; _ }
        | Ssa_op.Load { buffer; _ }
        | Ssa_op.Load_in_bounds { buffer; _ }
        | Ssa_op.Store { buffer; _ }
        | Ssa_op.Vec_load { buffer; _ }
        | Ssa_op.Vec_store { buffer; _ } ->
            note buffer
        | _ -> ())
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  List.filter
    (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.Set.mem b.Ssa_buffer.id !seen)
    p.Ssa_program.buffers

let ctype cx (v : Ssa_value.t) =
  match c_type cx v.Ssa_value.ty with
  | Some (t, _) -> t
  | None -> invalid_arg "Ssa_c: an effect has no C type"

(* Every yield is read before any parameter is rebound: through a temporary
   unless a parameter is yielded to itself. *)
let transfer cx depth params yields =
  let moves =
    List.filter
      (fun ((p : Ssa_value.t), (y : Ssa_value.t)) ->
        (not (is_erased p)) && not (Ssa_value.equal p y))
      (List.combine params yields)
  in
  if moves <> [] then (
    line cx depth "{";
    let temps =
      List.map
        (fun ((p : Ssa_value.t), y) ->
          let t = temp cx in
          line cx (depth + 1) "%s %s = %s;" (ctype cx p) t (name cx y);
          (p, t))
        moves
    in
    List.iter
      (fun ((p : Ssa_value.t), t) ->
        line cx (depth + 1) "%s = %s;" (name cx p) t)
      temps;
    line cx depth "}")

let rec region cx depth ~limits (r : Ssa_region.t) =
  List.iter (stmt cx depth ~limits) r.Ssa_region.body

and stmt cx depth ~limits (s : Ssa_region.t Ssa_stmt.t) =
  match s with
  | Ssa_stmt.Instr i ->
      if is_meter i.Ssa_instr.op then meter_op cx depth ~limits i.Ssa_instr.op
      else Ssa_c_op.instr cx depth i
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_c: a loop without an induction value"
      in
      let ivn = define cx iv in
      List.iter (fun p -> ignore (define cx p)) carried;
      let bound = temp cx in
      line cx depth "{";
      line cx (depth + 1) "const int64_t %s = %s;" bound (name cx hi);
      List.iter2
        (fun (p : Ssa_value.t) init ->
          if not (is_erased p) then
            line cx (depth + 1) "%s = %s;" (name cx p) (name cx init))
        carried inits;
      line cx (depth + 1) "for (%s = %s; %s < %s; %s += %Ld) {" ivn (name cx lo)
        ivn bound ivn step;
      region cx (depth + 2) ~limits body;
      transfer cx (depth + 2) carried body.Ssa_region.yields;
      line cx (depth + 1) "}";
      List.iter2
        (fun (r : Ssa_value.t) (p : Ssa_value.t) ->
          if not (is_erased r) then
            let rn = define cx r in
            line cx (depth + 1) "%s = %s;" rn (name cx p))
        results carried;
      line cx depth "}"
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      List.iter (fun r -> ignore (define cx r)) results;
      let arm (r : Ssa_region.t) =
        region cx (depth + 1) ~limits r;
        List.iter2
          (fun (res : Ssa_value.t) y ->
            if not (is_erased res) then
              line cx (depth + 1) "%s = %s;" (name cx res) (name cx y))
          results r.Ssa_region.yields
      in
      line cx depth "if (%s) {" (name cx cond);
      arm then_;
      line cx depth "} else {";
      arm else_;
      line cx depth "}"
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
      let iv =
        match body.Ssa_region.params with
        | iv :: _ -> iv
        | [] -> invalid_arg "Ssa_c: a sum without an induction value"
      in
      let ivn = define cx iv in
      let bound = temp cx and acc = temp cx in
      line cx depth "{";
      line cx (depth + 1) "const int64_t %s = %s;" bound (name cx hi);
      line cx (depth + 1) "%s %s = %s;" (ctype cx seed) acc (name cx seed);
      line cx (depth + 1) "for (%s = %s; %s < %s; %s += 1) {" ivn (name cx lo)
        ivn bound ivn;
      region cx (depth + 2) ~limits body;
      let term = List.hd body.Ssa_region.yields in
      line cx (depth + 2) "%s = %s + %s;" acc acc (name cx term);
      line cx (depth + 1) "}";
      (match results with
      | sum :: _ ->
          let sn = define cx sum in
          line cx (depth + 1) "%s = %s;" sn acc
      | [] -> ());
      line cx depth "}"

let kernel ?buffers ?sites ~name:fname (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_c.kernel: the program does not verify: %a"
           Ssa_verify.pp_error e));
  let buffers = Option.value buffers ~default:(arguments p) in
  let cx = Ssa_c_ctx.create ?site_table:sites buffers in
  match
    let params =
      List.map
        (fun (b : Ssa_buffer.t) ->
          let t = cell_type b.Ssa_buffer.id b.Ssa_buffer.format in
          Printf.sprintf "%s%s *%s"
            (match b.Ssa_buffer.role with
            | Ssa_buffer.Input -> "const "
            | Ssa_buffer.Output | Ssa_buffer.Scratch -> "")
            t
            (buffer_name cx b.Ssa_buffer.id))
        buffers
    in
    let limits = p.Ssa_program.scan_limits in
    let entry = p.Ssa_program.entry in
    List.iter (fun v -> ignore (define cx v)) entry.Ssa_region.params;
    region cx 1 ~limits entry;
    (params, limits)
  with
  | exception Refused e -> Error e
  | params, limits ->
      let body = Buffer.contents cx.out in
      let decls = List.rev cx.decls in
      let meter =
        if uses_meter p.Ssa_program.entry then
          [
            Printf.sprintf "  int64_t scan_remaining = %s;"
              (i64_lit (Expr.Scan_limits.max_updates limits));
            "  int64_t scan_live = 0;";
            "  (void)scan_remaining;";
            "  (void)scan_live;";
          ]
        else []
      in
      let decl_lines =
        List.map
          (fun (n, t, init) -> Printf.sprintf "  %s %s = %s;" t n init)
          decls
        @ List.map (fun (n, _, _) -> Printf.sprintf "  (void)%s;" n) decls
      in
      let voids =
        "  (void)err; (void)local;"
        :: List.map
             (fun (b : Ssa_buffer.t) ->
               Printf.sprintf "  (void)%s;" (buffer_name cx b.Ssa_buffer.id))
             buffers
      in
      let params_text =
        String.concat ", "
          ("struct model_error *err" :: "double *local" :: params)
      in
      let fn =
        String.concat "\n"
          ([ Printf.sprintf "static int %s(%s) {" fname params_text ]
          @ voids @ decl_lines @ meter
          @ [ body ^ "  return 0;"; "}"; "" ])
      in
      (* the emitter's own types and functions, each block guarded so that the
         kernels of one translation unit can all carry theirs *)
      let guarded (name, text) =
        Printf.sprintf "#ifndef SSA_C_%s\n#define SSA_C_%s\n%s\n#endif" name
          name text
      in
      let prelude =
        String.concat "\n"
          (List.map guarded (cx.vector_types @ cx.vector_helpers))
      in
      if cx.f32 then use cx R.Name.F32_prelude;
      let used = cx.used in
      Ok
        ( {
            Loop_ir.Loop_c.source = fn;
            prelude;
            helpers = List.filter (fun n -> List.mem n used) R.Name.all;
            local_doubles = cx.local_doubles;
            precision =
              (if cx.f32 then Loop_ir.Loop_numerics.Precision.F32
               else Loop_ir.Loop_numerics.Precision.F64);
            refusal = None;
            buffer_types =
              List.map
                (fun (b : Ssa_buffer.t) ->
                  cell_type b.Ssa_buffer.id b.Ssa_buffer.format)
                buffers;
          },
          match sites with
          | Some table -> table
          | None -> Array.of_list (List.rev cx.sites) )
