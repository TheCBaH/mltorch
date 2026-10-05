open Ssa_ir
open Ssa_wasm_ctx
open Ssa_wasm_op
open Ssa_wasm_instr

type error = Ssa_wasm_ctx.error

let pp_error = Ssa_wasm_ctx.pp_error

(* ---- control flow --------------------------------------------------------------- *)

let flat_regs st (vs : Ssa_value.t list) =
  List.concat_map
    (fun v -> if is_erased v then [] else Array.to_list (regs st v))
    vs

(* Every yield read before any parameter is rebound: pushed, then popped in
   reverse, so the transfer is simultaneous with no temporary. *)
let transfer st params yields =
  let moves =
    List.filter
      (fun (p, y) -> p <> y)
      (List.combine (flat_regs st params) (flat_regs st yields))
  in
  List.map (fun (_, y) -> get y) moves
  @ List.rev_map (fun (p, _) -> set p) moves

let copy st dst src =
  List.concat
    (List.map2
       (fun d s -> [ get s; set d ])
       (flat_regs st dst) (flat_regs st src))

let rec region st ~limits (r : Ssa_region.t) =
  List.concat_map (stmt st ~limits) r.Ssa_region.body

and stmt st ~limits (s : Ssa_region.t Ssa_stmt.t) : I.t list =
  match s with
  | Ssa_stmt.Instr i -> instr st ~limits i
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_wasm: a loop without an induction value"
      in
      (match Ssa_range.range st.ranges iv with
      | Ssa_range.Empty -> ()
      | Ssa_range.Range r ->
          if Int64.compare (Int64.add r.hi step) Ssa_const.index_max > 0 then
            refuse `Loop_leaves_domain);
      let ivl = (define st iv).(0) in
      List.iter (fun p -> ignore (define st p)) carried;
      let init = copy st carried inits in
      let body_instrs = region st ~limits body in
      let moves = transfer st carried body.Ssa_region.yields in
      List.iter (fun r -> ignore (define st r)) results;
      init @ read st lo
      @ [ set ivl ]
      @ [
          I.Block
            ( None,
              [
                I.Loop
                  ( None,
                    [ get ivl ]
                    @ read st hi
                    @ [ n Wasm_op.I32_ge_s; I.Br_if 1 ]
                    @ body_instrs @ moves
                    @ [
                        get ivl;
                        index_const step;
                        n Wasm_op.I32_add;
                        set ivl;
                        I.Br 0;
                      ] );
              ] );
        ]
      @ copy st results carried
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      List.iter (fun r -> ignore (define st r)) results;
      let arm (r : Ssa_region.t) =
        let body = region st ~limits r in
        body @ copy st results r.Ssa_region.yields
      in
      let yes = arm then_ in
      let no = arm else_ in
      read st cond @ [ I.If (None, yes, no) ]
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
      let iv =
        match body.Ssa_region.params with
        | iv :: _ -> iv
        | [] -> invalid_arg "Ssa_wasm: a sum without an induction value"
      in
      let ivl = (define st iv).(0) in
      let sum = List.hd results in
      let acc = define ~shape:Wide st sum in
      let body_instrs = region st ~limits body in
      let term = List.hd body.Ssa_region.yields in
      let add =
        match sum.Ssa_value.ty with
        | Ssa_type.Vec (e, _) ->
            List.concat
              (List.mapi
                 (fun i a ->
                   [
                     get a;
                     get (regs st term).(i);
                     vn (vbin e Expr.Value.Add);
                     set a;
                   ])
                 (Array.to_list acc))
        | _ ->
            [
              get acc.(0);
              read st term |> List.hd;
              n (float_op ~f32:(is_f32 sum) Expr.Value.Add);
              set acc.(0);
            ]
      in
      List.concat
        (List.mapi
           (fun i a -> [ get (regs st seed).(i); set a ])
           (Array.to_list acc))
      @ read st lo
      @ [ set ivl ]
      @ [
          I.Block
            ( None,
              [
                I.Loop
                  ( None,
                    [ get ivl ]
                    @ read st hi
                    @ [ n Wasm_op.I32_ge_s; I.Br_if 1 ]
                    @ body_instrs @ add
                    @ [ get ivl; i32 1; n Wasm_op.I32_add; set ivl; I.Br 0 ] );
              ] );
        ]

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

let f64_bytes xs =
  let buf = Buffer.create 64 in
  List.iter
    (fun x ->
      let bits = Int64.bits_of_float x in
      for k = 0 to 7 do
        Buffer.add_char buf
          (Char.chr
             (Int64.to_int (Int64.shift_right_logical bits (8 * k)) land 0xFF))
      done)
    xs;
  Buffer.contents buf

let kernel ?buffers ?sites ~relaxed_madd ~table_alloc (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_wasm.kernel: the program does not verify: %a"
           Ssa_verify.pp_error e));
  let buffers = Option.value buffers ~default:(arguments p) in
  let st =
    {
      extra = [];
      n_params = 1 + List.length buffers;
      values = Hashtbl.create 64;
      mask_shapes = Hashtbl.create 8;
      buffers;
      tables = Hashtbl.create 4;
      locals = Hashtbl.create 8;
      local_top = 0L;
      table_alloc;
      used = [];
      sites = [];
      site_count = 0;
      meter = None;
      f32 = false;
      relaxed_madd;
      ranges = Ssa_range.analyze p;
      site_table = sites;
    }
  in
  match
    (* per-channel parameters, once, as constant [f64] arrays *)
    let data =
      List.concat_map
        (fun (b : Ssa_buffer.t) ->
          match b.Ssa_buffer.format with
          | Ssa_format.I8 (Ssa_format.Per_channel { scale; zero_point })
          | Ssa_format.I16 (Ssa_format.Per_channel { scale; zero_point }) ->
              let bytes = 8 * Array.length scale in
              let scales = table_alloc ~bytes in
              let zeros = table_alloc ~bytes in
              Hashtbl.replace st.tables (b.Ssa_buffer.id :> int) (scales, zeros);
              [
                {
                  Wasm.Data.offset = scales;
                  bytes = f64_bytes (Array.to_list scale);
                };
                {
                  Wasm.Data.offset = zeros;
                  bytes =
                    f64_bytes
                      (Array.to_list (Array.map float_of_int zero_point));
                };
              ]
          | _ -> [])
        buffers
    in
    let limits = p.Ssa_program.scan_limits in
    let entry = p.Ssa_program.entry in
    List.iter (fun v -> ignore (define st v)) entry.Ssa_region.params;
    let body = region st ~limits entry in
    (data, limits, body)
  with
  | exception Refused e -> Error e
  | data, limits, body ->
      let prologue =
        match st.meter with
        | None -> []
        | Some (remaining, _) ->
            [ I.I64_const (Expr.Scan_limits.max_updates limits); set remaining ]
      in
      Ok
        {
          Loop_ir.Loop_wasm.func =
            {
              Wasm.Func.type_ =
                {
                  Wasm.Func_type.params =
                    Wasm_type.I32 :: List.map (fun _ -> Wasm_type.I32) buffers;
                  results = [ Wasm_type.I32 ];
                };
              locals = List.rev st.extra;
              body = prologue @ body @ [ i32 0 ];
            };
          callees = Loop_ir.Loop_wasm_link.reached st.used;
          local_bytes = st.local_top;
          data;
          sites =
            (match sites with
            | Some table -> table
            | None -> Array.of_list (List.rev st.sites));
          precision =
            (if st.f32 then Loop_ir.Loop_numerics.Precision.F32
             else Loop_ir.Loop_numerics.Precision.F64);
          refusal = None;
        }

let lower ?(relaxed_madd = false)
    ?(numerics = Loop_ir.Loop_numerics.Reference_f64) (p : Ssa_program.t) =
  Err.payload
    (Loop_ir.Loop_wasm.lower_with ~numerics (fun ~mark_base:_ ~table_alloc ->
         match kernel ~relaxed_madd ~table_alloc p with
         | Ok k -> Err.return k
         | Error e -> Err.fail e))
  |> Result.map_error (fun (e : error) -> e)
