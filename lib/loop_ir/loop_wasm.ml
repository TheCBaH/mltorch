open Loop_wasm_ctx
open Loop_wasm_value
open Loop_wasm_fail

type error = Loop_wasm_ctx.error

let pp_error = Loop_wasm_ctx.pp_error
let function_name = Loop_wasm_ctx.function_name
let error_address = Loop_wasm_ctx.error_address

type t = {
  module_ : Wasm.Module.t;
  local_base : int;
  mark_base : int option;
  heap_base : int;
  sites : Loop_failure.t array;
  precision : Loop_numerics.Precision.t;
}

type kernel = {
  func : Wasm.Func.t;
  callees : R.Callee.t list;
  local_bytes : int64;
  data : Wasm.Data.t list;
  sites : Loop_failure.t array;
  precision : Loop_numerics.Precision.t;
  refusal : Loop_numerics.Refusal.t option;
}

(* The [Fail_if] sites are numbered in the walk [Loop_js_failure.sites] makes,
   and each is checked against that array by physical equality. *)
let next_site st f =
  let k = st.next_site in
  if k >= Array.length st.sites || st.sites.(k) != f then
    invalid_arg "Loop_wasm: a failure site drifted from Loop_js_failure.sites";
  st.next_site <- k + 1;
  k

let meter st =
  match st.meter with
  | Some m -> m
  | None ->
      let remaining = fresh st Wasm_type.I64 in
      let live = fresh st Wasm_type.I64 in
      st.meter <- Some (remaining, live);
      (remaining, live)

let meter_failure st which limit =
  fail st F.Kind.Scan_meter
    [
      ( 0,
        i64_of_int
          (match which with
          | F.Meter.State_over_limit -> 0
          | F.Meter.Updates_exhausted -> 1) );
      (1, [ I.I64_const limit ]);
    ]

let rec stmt st ~limits (s : Loop_stmt.t) : I.t list =
  match s with
  | Loop_stmt.Alloc (a, count) ->
      let len = (count :> int) in
      let off = reserve_local st (Int64.of_int (array_cell_bytes st * len)) in
      Hashtbl.replace st.arrays (Loop_array.to_int a) off;
      if len = 0 then []
      else
        let c = fresh st Wasm_type.I32 in
        [
          i32 0;
          set c;
          I.Block
            ( None,
              [
                I.Loop
                  ( None,
                    [
                      get c;
                      int_const st len;
                      n Wasm_op.I32_ge_s;
                      I.Br_if 1;
                      get 0;
                      i32 off;
                      n Wasm_op.I32_add;
                      get c;
                      i32 (array_log2 st);
                      n Wasm_op.I32_shl;
                      n Wasm_op.I32_add;
                      fconst st 0.;
                      array_store st;
                      get c;
                      i32 1;
                      n Wasm_op.I32_add;
                      set c;
                      I.Br 0;
                    ] );
              ] );
        ]
  | Loop_stmt.Array_set (a, i, e) ->
      let at = array_address st a i in
      let e = num st e in
      at @ e @ [ array_store st ]
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      let e = num st e in
      e @ [ set (ftemp st t) ]
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      let e = big st e in
      e @ [ set (itemp st t) ]
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      let e = big st e in
      e @ [ n Wasm_op.I32_wrap_i64; set (xtemp st t) ]
  | Loop_stmt.Assign_index (t, i) ->
      let i = index st i in
      i @ [ set (xtemp st t) ]
  | Loop_stmt.Fail_if (p, f) -> (
      let site = next_site st f in
      match (p, f) with
      | Loop_bool.Index_overflows i, Loop_failure.Index_overflow { index = j }
        when i = j ->
          List.concat_map
            (fun node ->
              outside_int32 node.value
              @ [
                  I.If
                    ( None,
                      fail st F.Kind.Index_overflow
                        [
                          (0, i64_of_int node.op); (1, node.lhs); (2, node.rhs);
                        ],
                      [] );
                ])
            (overflow_nodes st i)
      | _, Loop_failure.Index_overflow _ ->
          invalid_arg
            "Loop_wasm: an index overflow failure under a foreign predicate"
      | _ ->
          let p = pred st p in
          p @ [ I.If (None, failure st ~site f, []) ])
  | Loop_stmt.For { var = v; lo; hi = hi_ix; body } ->
      for_loop st v lo hi_ix (fun () -> block st ~limits body)
  | Loop_stmt.If (p, yes, no) ->
      let p = pred st p in
      let yes = block st ~limits yes in
      let no = block st ~limits no in
      p @ [ I.If (None, yes, no) ]
  | Loop_stmt.Charge_scan_update ->
      let remaining, _ = meter st in
      let limit = Expr.Scan_limits.max_updates limits in
      [
        get remaining;
        I.I64_const 0L;
        n Wasm_op.I64_le_s;
        I.If (None, meter_failure st F.Meter.Updates_exhausted limit, []);
        get remaining;
        I.I64_const 1L;
        n Wasm_op.I64_sub;
        set remaining;
      ]
  | Loop_stmt.Reduce_sum _ ->
      invalid_arg
        "Loop_wasm: a structured sum is expanded at the entry point \
         (Loop_sum.program)"
  | Loop_stmt.Mark m -> (
      (* A counting build bumps its mark's word; the default build emits
         nothing, so no inference path pays for a mark or calls the host. *)
      match st.mark_base with
      | None -> []
      | Some base ->
          let at = base + (4 * Loop_mark.index m) in
          [
            i32 at;
            i32 at;
            I.Load (Wasm.Load.I32_load, arg 2 0);
            i32 1;
            n Wasm_op.I32_add;
            I.Store (Wasm.Store.I32_store, arg 2 0);
          ])
  | Loop_stmt.Release_scan_state width ->
      let _, live = meter st in
      [
        get live;
        I.I64_const (Int64.of_int (2 * width));
        n Wasm_op.I64_sub;
        set live;
      ]
  | Loop_stmt.Reserve_scan_state width ->
      let _, live = meter st in
      let amount = 2 * width in
      let max_state = Expr.Scan_limits.max_state limits in
      [
        get live;
        I.I64_const (Int64.of_int amount);
        n Wasm_op.I64_add;
        I.I64_const (Int64.of_int max_state);
        n Wasm_op.I64_gt_s;
        I.If
          ( None,
            meter_failure st F.Meter.State_over_limit (Int64.of_int max_state),
            [] );
        get live;
        I.I64_const (Int64.of_int amount);
        n Wasm_op.I64_add;
        set live;
      ]
  | Loop_stmt.Reset_meter ->
      let remaining, live = meter st in
      [
        I.I64_const (Expr.Scan_limits.max_updates limits);
        set remaining;
        I.I64_const 0L;
        set live;
      ]
  | Loop_stmt.Store { buffer = b; coord = c; value } -> store st b (At c) value
  | Loop_stmt.Store_flat { buffer = b; offset = i; value } ->
      store st b (Flat i) value

(* A scalar loop over [\[lo, hi)]. The body is lowered last, by [body]: the
   loop's own locals come first, so a module's local numbering does not depend on
   whether a body allocates any. *)
and for_loop st v lo hi_ix body =
  let name = var st v in
  let lo = index st lo in
  let hi, limit =
    match hi_ix with
    | Loop_index.Const _ | Loop_index.Var _ -> ([], index st hi_ix)
    | _ ->
        let b = bound st v in
        (index st hi_ix @ [ set b ], [ get b ])
  in
  let body = body () in
  lo
  @ [ set name ]
  @ hi
  @ [
      I.Block
        ( None,
          [
            I.Loop
              ( None,
                [ get name ]
                @ limit
                @ [ n Wasm_op.I32_ge_s; I.Br_if 1 ]
                @ body
                @ [ get name; i32 1; n Wasm_op.I32_add; set name; I.Br 0 ] );
          ] );
    ]

(* A vector program's tree: scalar statements as they were, scalar loops and
   branches around vector loops, and the vector loops themselves. *)
and node st ~limits (nd : Loop_vector.node) =
  match nd with
  | Loop_vector.Scalar s -> stmt st ~limits s
  | Loop_vector.Reduction r -> Loop_wasm_vector.reduction st r
  | Loop_vector.If (p, yes, no) ->
      let p = pred st p in
      let yes = nodes st ~limits yes in
      let no = nodes st ~limits no in
      p @ [ I.If (None, yes, no) ]
  | Loop_vector.Loop { var = v; lo; hi; body } ->
      for_loop st v lo hi (fun () -> nodes st ~limits body)
  | Loop_vector.Vector l ->
      Loop_wasm_vector.loop st ~scalar_stmt:(stmt st ~limits) l

and nodes st ~limits ns = List.concat_map (node st ~limits) ns

and store st b addr value =
  let at = cell_address st b addr in
  match value with
  | Loop_stored.Bool e ->
      let e = num st e in
      at @ e
      @ [
          fconst st 0.;
          n (if st.f32 then Wasm_op.F32_ne else Wasm_op.F64_ne);
          I.Store (Wasm.Store.I32_store8, arg 0 0);
        ]
  | Loop_stored.F32 (Loop_expr.Round_f32 e) | Loop_stored.F32 e ->
      (* The store narrows to binary32 itself, so an outermost [Round_f32] is
         the same rounding written twice. *)
      let e = num st e in
      at @ e
      @ (if st.f32 then [] else [ n Wasm_op.F32_demote_f64 ])
      @ [ I.Store (Wasm.Store.F32_store, arg 2 0) ]
  | Loop_stored.I64 e ->
      let e = big st e in
      at @ e @ [ I.Store (Wasm.Store.I64_store, arg 3 0) ]

and block st ~limits body = List.concat_map (stmt st ~limits) body

let channel_tables (b : Loop_buffer.t) =
  match fmt_of b with
  | "i8" | "i16" -> (
      let q = quant_of b in
      match Quant.channel_count q with
      | None -> None
      | Some k -> Some (List.init k (fun c -> Quant.params q ~c:(Dim.index c))))
  | _ -> None

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

let kernel_exact ~vector ~numerics ~precision ~mark_base ~table_alloc
    (p : Loop_program.t) : (kernel, error) Err.t =
  let p = Loop_sum.program p in
  let plan =
    match precision with
    | None -> Ok (Loop_plan.resolve ?target:vector ~numerics p)
    | Some precision ->
        Result.map_error
          (fun r -> `Unsupported_precision r)
          (Loop_plan.force ?target:vector ~precision p)
  in
  match plan with
  | Error e -> Err.fail e
  | Ok plan ->
      Err.Escape.with_escape (fun esc ->
          let buffers = Hashtbl.create 8 in
          List.iteri
            (fun k (b : Loop_buffer.t) ->
              Hashtbl.replace buffers (Tensor_id.to_int b.Loop_buffer.id) k)
            p.Loop_program.buffers;
          let st =
            {
              esc;
              f32 = plan.Loop_plan.precision = Loop_numerics.Precision.F32;
              relaxed_madd =
                (match vector with
                | Some t -> (Loop_target.f32 t).Loop_target.relaxed_madd
                | None -> false);
              extra = [];
              n_params = 1 + List.length p.Loop_program.buffers;
              vars = Hashtbl.create 8;
              bounds = Hashtbl.create 8;
              floats = Hashtbl.create 8;
              int64s = Hashtbl.create 8;
              indices = Hashtbl.create 8;
              buffers;
              tables = Hashtbl.create 4;
              arrays = Hashtbl.create 4;
              local_top = 0L;
              table_alloc;
              mark_base;
              used = [];
              meter = None;
              sites = F.sites p;
              next_site = 0;
            }
          in
          (* Per-channel parameters, once, as constant [f64] arrays. *)
          let data =
            List.concat_map
              (fun (b : Loop_buffer.t) ->
                match channel_tables b with
                | None -> []
                | Some params ->
                    let bytes = 8 * List.length params in
                    let scales = table_alloc ~bytes in
                    let zeros = table_alloc ~bytes in
                    Hashtbl.replace st.tables
                      (Tensor_id.to_int b.Loop_buffer.id)
                      (scales, zeros);
                    [
                      {
                        Wasm.Data.offset = scales;
                        bytes = f64_bytes (List.map fst params);
                      };
                      {
                        Wasm.Data.offset = zeros;
                        bytes =
                          f64_bytes
                            (List.map (fun (_, z) -> float_of_int z) params);
                      };
                    ])
              p.Loop_program.buffers
          in
          let limits = p.Loop_program.scan_limits in
          let body =
            match plan.Loop_plan.vector with
            | Some vp ->
                (* Strict vectorization for the 128-bit Wasm target: the program, with
             its independent loops replaced by vector loops; the verifier runs
             before anything is lowered. *)
                (match Err.payload (Loop_vector_check.program vp) with
                | Ok () -> ()
                | Error e ->
                    invalid_arg
                      (Fmt.str
                         "Loop_wasm: the vectorizer built an invalid program: \
                          %a"
                         Loop_vector_check.pp_error e));
                nodes st ~limits vp.Loop_vector.body
            | None -> block st ~limits p.Loop_program.body
          in
          if st.next_site <> Array.length st.sites then
            invalid_arg "Loop_wasm: a failure site was not written";
          let prologue =
            match st.meter with
            | None -> []
            | Some (remaining, _) ->
                [
                  I.I64_const (Expr.Scan_limits.max_updates limits);
                  set remaining;
                ]
          in
          {
            func =
              {
                Wasm.Func.type_ =
                  {
                    Wasm.Func_type.params =
                      Wasm_type.I32
                      :: List.map
                           (fun _ -> Wasm_type.I32)
                           p.Loop_program.buffers;
                    results = [ Wasm_type.I32 ];
                  };
                locals = List.rev st.extra;
                body = prologue @ body @ [ i32 0 ];
              };
            callees = Loop_wasm_link.reached st.used;
            local_bytes = st.local_top;
            data;
            sites = st.sites;
            precision = plan.Loop_plan.precision;
            refusal = plan.Loop_plan.refusal;
          })

(* Widened for a caller that composes it with other errors, which would
   otherwise have to name this row's tags itself. *)
let kernel ?vector ?(numerics = Loop_numerics.Reference_f64) ?precision
    ?mark_base ~table_alloc p =
  Err.map_error
    (fun (e : error) ->
      match e with
      | `Index_constant_out_of_range _ as e -> e
      | `Local_arrays_too_large _ as e -> e
      | `Unsupported_precision _ as e -> e)
    (kernel_exact ~vector ~numerics ~precision ~mark_base ~table_alloc p)

let align16 x = Int64.logand (Int64.add x 15L) (Int64.lognot 15L)

let lower ?vector ?(numerics = Loop_numerics.Reference_f64) ?precision
    ?(count_marks = false) (p : Loop_program.t) : (t, error) Err.t =
  (* The module's own bytes: the error record, then constant tables as the
     kernel asks for them, then the local region, then the host's buffers. *)
  let top = ref (align16 (Int64.of_int W.record_bytes)) in
  let mark_base =
    if count_marks then (
      let at = Int64.to_int !top in
      top :=
        align16 (Int64.add !top (Int64.of_int (4 * List.length Loop_mark.all)));
      Some at)
    else None
  in
  let table_alloc ~bytes =
    let off = align8 !top in
    top := Int64.add off (Int64.of_int bytes);
    Int64.to_int off
  in
  Err.map
    (fun (k : kernel) ->
      let imports, funcs, base =
        Loop_wasm_link.link ~callees:k.callees [ k.func ]
      in
      let local_base = Int64.to_int (align16 !top) in
      let heap_base =
        Int64.to_int
          (align16 (Int64.add (Int64.of_int local_base) k.local_bytes))
      in
      let pages = max 1 ((heap_base + 65535) / 65536) in
      let m =
        {
          Wasm.Module.imports;
          funcs;
          globals = [];
          memory = Some { Wasm.Memory.min_pages = pages; max_pages = None };
          exports =
            [
              { Wasm.Export.name = "memory"; kind = Wasm.Export.Memory };
              { Wasm.Export.name = function_name; kind = Wasm.Export.Func base };
            ];
          data = k.data;
          customs = [ { Wasm.Custom.name = "abi"; payload = "loop-wasm/1" } ];
        }
      in
      let manifest =
        Loop_wasm_link.manifest ~numerics ~precisions:[ k.precision ]
          ~callees:k.callees m
      in
      {
        module_ =
          {
            m with
            Wasm.Module.customs =
              m.Wasm.Module.customs
              @ [ { Wasm.Custom.name = "manifest"; payload = manifest } ];
          };
        local_base;
        heap_base;
        mark_base;
        sites = k.sites;
        precision = k.precision;
      })
    (kernel_exact ~vector ~numerics ~precision ~mark_base ~table_alloc p)

let with_pages t ~pages =
  match t.module_.Wasm.Module.memory with
  | Some { Wasm.Memory.min_pages; _ } when pages >= min_pages ->
      {
        t.module_ with
        Wasm.Module.memory =
          Some { Wasm.Memory.min_pages = pages; max_pages = None };
      }
  | _ -> invalid_arg "Loop_wasm.with_pages: fewer pages than the static region"

let encode t = Wasm_encode.module_ t.module_
