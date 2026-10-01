open Loop_wasm_ctx
open Loop_wasm_value

(* A failure is a returned status, never a trap: the record is written, then
   the kernel returns nonzero. Each field is an [i64] expression evaluated only
   here, once the check has fired. *)
let fail st kind (fields : (int * I.t list) list) =
  [ i32 (W.kind_index kind); call st R.Callee.Fail_set ]
  @ List.concat_map
      (fun (slot, value) ->
        [ i32 error_address ]
        @ value
        @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset slot)) ])
      fields
  @ [ i32 1; I.Return ]

let i64_of_int k = [ I.I64_const (Int64.of_int k) ]

let scan_failure st which ~site ~local ~row ~lane ~extent =
  let row = wide_index st row in
  let lane = wide_index st lane in
  fail st F.Kind.Scan_projection
    [
      ( 0,
        i64_of_int
          (match which with F.Projection.Lane -> 0 | F.Projection.Row -> 1) );
      (1, i64_of_int (if Option.is_some local then 1 else 0));
      (2, row);
      (3, lane);
      (4, i64_of_int extent);
      (5, i64_of_int site);
    ]

(* The first axis, in [Expr.Axis.all] order, whose component is outside the
   buffer's shape names the failure ([Expr_bridge.bound_in_range]); none is a
   defect of the program, recorded as such. *)
let coord_failure st (b : Loop_buffer.t) (c : Loop_index.coord) =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  let comps =
    List.map
      (fun a -> (a, fresh st Wasm_type.I32, Dim.to_int (Vec6.get shape a)))
      Expr.Axis.all
  in
  let eval =
    List.concat_map
      (fun (a, l, _) -> index st (Expr.Coord.get c a) @ [ set l ])
      comps
  in
  let wide l = [ get l; n Wasm_op.I64_extend_i32_s ] in
  let rec chain k = function
    | [] -> [ i32 (W.kind_index F.Kind.Defect); call st R.Callee.Fail_set ]
    | (_, l, extent) :: rest ->
        [
          get l;
          int_const st extent;
          n Wasm_op.I32_ge_u;
          I.If
            ( None,
              [
                i32 error_address;
                I.I64_const (Int64.of_int k);
                I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 1));
                i32 error_address;
              ]
              @ wide l
              @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 2)) ],
              chain (k + 1) rest );
        ]
  in
  eval
  @ [ i32 (W.kind_index F.Kind.Coord_out_of_range); call st R.Callee.Fail_set ]
  @ [
      i32 error_address;
      I.I64_const (Int64.of_int (Tensor_id.to_int b.Loop_buffer.id));
      I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 0));
    ]
  @ List.concat
      (List.mapi
         (fun k (_, l, _) ->
           [ i32 error_address ]
           @ wide l
           @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset (3 + k))) ])
         comps)
  @ chain 0 comps
  @ [ i32 1; I.Return ]

(* [Value.i64_of_float]'s three rejections: NaN, an infinity, or a finite value
   beyond the [int64] range. *)
let from_float_failure st value =
  let x = fresh st Wasm_type.F64 in
  let only kind =
    [ i32 (W.kind_index kind); call st R.Callee.Fail_set; i32 1; I.Return ]
  in
  num st value
  @ [ set x; get x; get x; n Wasm_op.F64_ne ]
  @ [
      I.If
        ( None,
          only F.Kind.I64_from_float_nan,
          [
            get x;
            n Wasm_op.F64_abs;
            f64 Float.infinity;
            n Wasm_op.F64_eq;
            I.If
              ( None,
                only F.Kind.I64_from_float_infinite,
                fail st F.Kind.I64_from_float_out_of_range
                  [ (0, [ get x; n Wasm_op.I64_reinterpret_f64 ]) ] );
          ] );
    ]

let failure st ~site : Loop_failure.t -> I.t list = function
  | Loop_failure.Load_out_of_range { buffer = b; coord = c } ->
      coord_failure st b c
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      fail st F.Kind.Gather_index_out_of_range
        [ (0, big st raw); (1, i64_of_int extent) ]
  | Loop_failure.I64_division_by_zero -> fail st F.Kind.I64_division_by_zero []
  | Loop_failure.I64_division_overflow ->
      fail st F.Kind.I64_division_overflow []
  | Loop_failure.I64_from_float { value } -> from_float_failure st value
  | Loop_failure.Index_overflow _ ->
      invalid_arg "Loop_wasm.failure: an index overflow is written per node"
  | Loop_failure.Local_out_of_range _ ->
      fail st F.Kind.Unbound_local [ (0, i64_of_int site) ]
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      scan_failure st F.Projection.Lane ~site ~local ~row ~lane ~extent
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      scan_failure st F.Projection.Row ~site ~local ~row ~lane ~extent
