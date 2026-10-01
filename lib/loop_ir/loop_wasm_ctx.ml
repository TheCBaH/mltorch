module I = Wasm.Instr
module R = Loop_wasm_runtime
module F = Loop_js_failure
module W = Loop_wasm_failure

let function_name = "loop_kernel"
let error_address = 0

type error =
  [ `Index_constant_out_of_range of int | `Local_arrays_too_large of int64 ]

let pp_error ppf : [< error ] -> unit = function
  | `Index_constant_out_of_range n ->
      Fmt.pf ppf "index constant %d is not a valid 32-bit index operand" n
  | `Local_arrays_too_large n ->
      Fmt.pf ppf "local arrays need %Ld bytes, beyond the static region" n

(* The static region (error record, per-channel tables, local arrays) must stay
   far below the 2 GiB memory policy, so every offset below is a positive
   [int32]. *)
let static_cap = 0x4000_0000L

type st = {
  esc : error Err.Escape.t;
  mutable extra : Wasm_type.t list;  (** locals past the buffer parameters *)
  n_params : int;
  vars : (int, int) Hashtbl.t;
  bounds : (int, int) Hashtbl.t;
  floats : (int, int) Hashtbl.t;
  int64s : (int, int) Hashtbl.t;
  indices : (int, int) Hashtbl.t;
  buffers : (int, int) Hashtbl.t;  (** tensor id -> parameter position *)
  tables : (int, int * int) Hashtbl.t;  (** per-channel scale and zero tables *)
  arrays : (int, int) Hashtbl.t;
      (** array id -> byte offset from the [local] base pointer *)
  mutable local_top : int64;
  table_alloc : bytes:int -> int;
      (** reserves [bytes] of constant data, returning its absolute address *)
  mutable used : R.Callee.t list;
  mutable meter : (int * int) option;  (** [scan_remaining], [scan_live] *)
  sites : Loop_failure.t array;
  mutable next_site : int;
}

let n op = I.Numeric op
let i32 k = I.I32_const (Int32.of_int k)
let get i = I.Local_get i
let set i = I.Local_set i
let f64 x = I.F64_const (Int64.bits_of_float x)
let arg align offset = { Wasm.Mem_arg.align; offset }
let fits32 k = Int32.to_int (Int32.of_int k) = k
let refuse st e = Err.Escape.throw st.esc e

(* A constant that becomes an [i32] immediate. *)
let int_const st k =
  if fits32 k then i32 k else refuse st (`Index_constant_out_of_range k)

let call st c =
  if not (List.mem c st.used) then st.used <- c :: st.used;
  R.call c

let fresh st ty =
  let id = st.n_params + List.length st.extra in
  st.extra <- ty :: st.extra;
  id

let local table st key ty =
  match Hashtbl.find_opt table key with
  | Some l -> l
  | None ->
      let l = fresh st ty in
      Hashtbl.add table key l;
      l

let var st v = local st.vars st (Loop_var.to_int v) Wasm_type.I32
let bound st v = local st.bounds st (Loop_var.to_int v) Wasm_type.I32
let ftemp st t = local st.floats st (Loop_temp.to_int t) Wasm_type.F64
let itemp st t = local st.int64s st (Loop_temp.to_int t) Wasm_type.I64
let xtemp st t = local st.indices st (Loop_temp.to_int t) Wasm_type.I32

(* Parameter 0 is the [local] base pointer, so buffer [k] is parameter [k + 1]. *)
let buffer_param st (b : Loop_buffer.t) =
  1 + Hashtbl.find st.buffers (Tensor_id.to_int b.Loop_buffer.id)

let align8 x = Int64.logand (Int64.add x 7L) (Int64.lognot 7L)

(* Reserves [bytes] of the kernel's local region, 8-aligned, relative to the
   [local] base the caller passes: invocations run one after another, so every
   kernel's arrays overlay the same region. *)
let reserve_local st bytes =
  let off = align8 st.local_top in
  let top = Int64.add off bytes in
  if Int64.compare top static_cap > 0 then
    refuse st (`Local_arrays_too_large top);
  st.local_top <- top;
  Int64.to_int off

(* The per-axis coordinates of [Loop_ir]'s accesses become one row-major offset,
   folded as it is built exactly as [Loop_c] does. *)
let small k = k > -0x4000_0000 && k < 0x4000_0000

let scale_i k (a : Loop_index.t) : Loop_index.t =
  match a with
  | _ when k = 1 -> a
  | Loop_index.Const m when small m && small k && small (k * m) ->
      Loop_index.Const (k * m)
  | _ -> Loop_index.Scale (k, a)

let add_i (a : Loop_index.t) (b : Loop_index.t) : Loop_index.t =
  match (a, b) with
  | Loop_index.Const 0, x | x, Loop_index.Const 0 -> x
  | Loop_index.Const x, Loop_index.Const y when small x && small y ->
      Loop_index.Const (x + y)
  | _ -> Loop_index.Add (a, b)

let offset_index (b : Loop_buffer.t) (c : Loop_index.coord) : Loop_index.t =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  List.fold_left
    (fun acc a ->
      let extent = Dim.to_int (Vec6.get shape a) in
      let i = Expr.Coord.get c a in
      match acc with
      | None -> Some i
      | Some acc -> Some (add_i (scale_i extent acc) i))
    None Expr.Axis.all
  |> Option.get

let fmt_of (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

(* log2 of the cell width in bytes. *)
let cell_log2 b =
  match fmt_of b with
  | "bool" | "i8" -> 0
  | "bf16" | "f16" | "i16" -> 1
  | "f32" | "i32" -> 2
  | "f64" | "i64" -> 3
  | f -> invalid_arg ("Loop_wasm: no cell width for format " ^ f)
